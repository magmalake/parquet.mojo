"""`ParquetReader` — open a file, pick columns and row groups, read batches.

```mojo
from parquet import ParquetReader

var r = ParquetReader.open("part-0.parquet")
r.select_columns(["id", "name"])
var t = r.read_table()
print(t.num_rows, "rows")
```

The reader is parametrised on a `CodecSet` (see `parquet.codec`); the default
covers `UNCOMPRESSED`, `SNAPPY` and `GZIP` with no FFI, and
`parquet.ext_full.AllCodecs` adds `ZSTD` and `LZ4`.

Projection is by name, by dotted leaf path, or — for Iceberg — by Parquet
**field id**. Row groups can be chosen explicitly, or pruned automatically
from their statistics against simple `column op literal` predicates.
"""

from parquet.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_UINT16,
    AT_UINT32,
    AT_UINT64,
    AT_UINT8,
    AT_FLOAT16,
    AT_FLOAT32,
    AT_FLOAT64,
    AT_MAP,
    AT_UTF8,
    ArrayArena,
    ArrayData,
    ArrowType,
    bit_get,
    load_f32,
    load_f64,
    load_i32,
    load_i64,
)
from parquet.assemble import LeafSlice, build_field, first_leaf
from parquet.carrow import ExportedArray, export_c
from parquet.codec import CodecSet, DefaultCodecs
from parquet.page import ColumnData, read_column_chunk
from parquet.schema import ArrowField, LeafColumn, ParquetSchema, build_schema
from parquet.stats import (
    SV_BYTES,
    SV_NONE,
    ScalarValue,
    TypedStats,
    compare_scalars,
    decode_statistic,
    decode_stats,
)
from std.memory import bitcast
from threads import num_cpus, parallel_for
from thrift import (
    ColumnIndex,
    ColumnMetaData,
    FileMetaData,
    OffsetIndex,
    RowGroup,
    TCompactProtocolReader,
    read_footer,
    read_parquet_file,
)

comptime OP_EQ = 0
comptime OP_NE = 1
comptime OP_LT = 2
comptime OP_LE = 3
comptime OP_GT = 4
comptime OP_GE = 5


@fieldwise_init
struct Predicate(Copyable, Movable):
    """`column op literal`, for row-group and page pruning."""

    var column: String
    var op: Int
    var value: ScalarValue

    def __init__(out self, *, copy: Self):
        self.column = copy.column.copy()
        self.op = copy.op
        self.value = copy.value.copy()

    def __init__(out self, *, deinit move: Self):
        self.column = move.column^
        self.op = move.op
        self.value = move.value^


def op_name(op: Int) -> String:
    var names: List[String] = [
        String("="),
        String("!="),
        String("<"),
        String("<="),
        String(">"),
        String(">="),
    ]
    if op < 0 or op >= len(names):
        return String("?")
    return names[op].copy()


def range_can_match(
    op: Int, lo: ScalarValue, hi: ScalarValue, v: ScalarValue
) raises -> Bool:
    """Could any value in `[lo, hi]` satisfy `x op v`?"""
    if lo.kind != v.kind:
        return True  # different kinds — do not prune on a guess
    var cl = compare_scalars(lo, v)
    var ch = compare_scalars(hi, v)
    if op == OP_EQ:
        return cl <= 0 and ch >= 0
    if op == OP_NE:
        # Only a row group where every value equals `v` can be pruned.
        return not (cl == 0 and ch == 0)
    if op == OP_LT:
        return cl < 0
    if op == OP_LE:
        return cl <= 0
    if op == OP_GT:
        return ch > 0
    if op == OP_GE:
        return ch >= 0
    return True


def _merge_ranges(var ranges: List[Tuple[Int, Int]]) -> List[Tuple[Int, Int]]:
    """Sorted, disjoint, adjacent ranges joined."""
    var out = List[Tuple[Int, Int]]()
    for r in ranges:
        if r[1] <= r[0]:
            continue
        if len(out) and out[len(out) - 1][1] >= r[0]:
            var last = out[len(out) - 1]
            var hi = last[1] if last[1] > r[1] else r[1]
            out[len(out) - 1] = (last[0], hi)
        else:
            out.append(r)
    return out^


def _intersect_ranges(
    a: List[Tuple[Int, Int]], b: List[Tuple[Int, Int]]
) -> List[Tuple[Int, Int]]:
    var out = List[Tuple[Int, Int]]()
    var i = 0
    var j = 0
    while i < len(a) and j < len(b):
        var lo = a[i][0] if a[i][0] > b[j][0] else b[j][0]
        var hi = a[i][1] if a[i][1] < b[j][1] else b[j][1]
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out^


struct RecordBatch(Copyable, Defaultable, Movable):
    """A contiguous run of rows as Arrow arrays, one per selected column."""

    var arena: ArrayArena
    var roots: List[Int]
    var num_rows: Int

    def __init__(out self):
        self.arena = ArrayArena()
        self.roots = List[Int]()
        self.num_rows = 0

    def __init__(out self, *, copy: Self):
        self.arena = copy.arena.copy()
        self.roots = copy.roots.copy()
        self.num_rows = copy.num_rows

    def __init__(out self, *, deinit move: Self):
        self.arena = move.arena^
        self.roots = move.roots^
        self.num_rows = move.num_rows

    def num_columns(self) -> Int:
        return len(self.roots)

    def column(ref self, i: Int) -> ref[self.arena.nodes[0]] ArrayData:
        return self.arena.nodes[self.roots[i]]

    def child(
        ref self, node: Int, k: Int
    ) -> ref[self.arena.nodes[0]] ArrayData:
        return self.arena.nodes[self.arena.nodes[node].children[k]]

    def name(self, i: Int) -> String:
        return self.arena.nodes[self.roots[i]].name.copy()

    def type(self, i: Int) -> ArrowType:
        return self.arena.nodes[self.roots[i]].type.copy()

    def export_c(self, i: Int) raises -> ExportedArray:
        """Column `i` over the Arrow C Data Interface. The result owns copies
        of every buffer, so it outlives this batch."""
        return export_c(self.arena, self.roots[i])

    def column_i64(self, i: Int) raises -> Tuple[List[Int64], List[Bool]]:
        return array_i64(self.column(i))

    def column_f64(self, i: Int) raises -> Tuple[List[Float64], List[Bool]]:
        return array_f64(self.column(i))

    def column_bool(self, i: Int) raises -> Tuple[List[Bool], List[Bool]]:
        return array_bool(self.column(i))

    def column_str(self, i: Int) raises -> Tuple[List[String], List[Bool]]:
        return array_str(self.column(i))


def _append_validity(a: ArrayData, mut out: List[Bool]):
    for i in range(a.length):
        out.append(bit_get(Span(a.validity), i))


def array_i64_into(
    a: ArrayData, mut vals: List[Int64], mut valid: List[Bool]
) raises:
    """Widen any integer, date, time or timestamp array to `Int64`."""
    var w = a.type.fixed_width()
    if w == 0 or a.type.id == AT_FLOAT32 or a.type.id == AT_FLOAT64:
        raise Error(
            String(
                "parquet: column of type ", String(a.type), " is not an integer"
            )
        )
    var signed = not (
        a.type.id == AT_UINT8
        or a.type.id == AT_UINT16
        or a.type.id == AT_UINT32
        or a.type.id == AT_UINT64
    )
    for i in range(a.length):
        var u: UInt64 = 0
        for k in range(w):
            u |= UInt64(a.values[i * w + k]) << UInt64(8 * k)
        if signed and w < 8:
            var sign_bit = UInt64(1) << UInt64(8 * w - 1)
            if (u & sign_bit) != 0:
                u |= ~((UInt64(1) << UInt64(8 * w)) - 1)
        vals.append(bitcast[DType.int64](u))
    _append_validity(a, valid)


def array_f64_into(
    a: ArrayData, mut vals: List[Float64], mut valid: List[Bool]
) raises:
    if a.type.id == AT_FLOAT64:
        for i in range(a.length):
            vals.append(load_f64(Span(a.values), i))
    elif a.type.id == AT_FLOAT32:
        for i in range(a.length):
            vals.append(Float64(load_f32(Span(a.values), i)))
    elif a.type.id == AT_FLOAT16:
        for i in range(a.length):
            var bits = UInt16(a.values[i * 2]) | (
                UInt16(a.values[i * 2 + 1]) << 8
            )
            vals.append(Float64(bitcast[DType.float16](bits)))
    else:
        raise Error(
            String(
                "parquet: column of type ",
                String(a.type),
                " is not floating point",
            )
        )
    _append_validity(a, valid)


def array_bool_into(
    a: ArrayData, mut vals: List[Bool], mut valid: List[Bool]
) raises:
    if a.type.id != AT_BOOL:
        raise Error(
            String(
                "parquet: column of type ", String(a.type), " is not boolean"
            )
        )
    for i in range(a.length):
        vals.append(bit_get(Span(a.values), i))
    _append_validity(a, valid)


def array_str_into(
    a: ArrayData, mut vals: List[String], mut valid: List[Bool]
) raises:
    if a.type.id != AT_UTF8 and a.type.id != AT_BINARY:
        raise Error(
            String(
                "parquet: column of type ",
                String(a.type),
                " is not a byte array",
            )
        )
    for i in range(a.length):
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        vals.append(String(StringSlice(unsafe_from_utf8=Span(a.values)[lo:hi])))
    _append_validity(a, valid)


def array_i64(a: ArrayData) raises -> Tuple[List[Int64], List[Bool]]:
    var vals = List[Int64]()
    var valid = List[Bool]()
    array_i64_into(a, vals, valid)
    return (vals^, valid^)


def array_f64(a: ArrayData) raises -> Tuple[List[Float64], List[Bool]]:
    var vals = List[Float64]()
    var valid = List[Bool]()
    array_f64_into(a, vals, valid)
    return (vals^, valid^)


def array_bool(a: ArrayData) raises -> Tuple[List[Bool], List[Bool]]:
    var vals = List[Bool]()
    var valid = List[Bool]()
    array_bool_into(a, vals, valid)
    return (vals^, valid^)


def array_str(a: ArrayData) raises -> Tuple[List[String], List[Bool]]:
    var vals = List[String]()
    var valid = List[Bool]()
    array_str_into(a, vals, valid)
    return (vals^, valid^)


comptime FileBytes = Span[UInt8, ImmUntrackedOrigin]
"""How `ParquetReader` holds the file, whether or not it owns the bytes.

The origin is untracked because a borrowing reader has no lifetime relation
the compiler can express, and a real origin parameter here would infect every
type that holds a reader.
"""


def _as_file_bytes(ref data: List[UInt8]) -> FileBytes:
    return rebind[FileBytes](Span(data))


def _no_chunk_meta(rg: Int, leaf: Int) -> Error:
    return Error(
        String(
            "parquet: column chunk ",
            leaf,
            " of row group ",
            rg,
            " has no metadata (an encrypted column?)",
        )
    )


def _bad_chunk_count(rg: Int, got: Int, nleaves: Int) -> String:
    return String(
        "parquet: row group ",
        rg,
        " has ",
        got,
        " column chunk(s) for a schema with ",
        nleaves,
        " leaves",
    )


def _short_levels(leaf: LeafColumn, which: StringSlice) -> String:
    return String(
        "parquet: column '",
        leaf.dotted(),
        "' decoded fewer ",
        which,
        " levels than it has value slots",
    )


# ── the unit of a load, and its two axes of fan-out ─────────────────────────
#
# `_load` decodes one column chunk per leaf and then walks its levels once to
# build the per-row indices `_assemble` slices with. That whole per-leaf body
# is `_decode_leaf`, which reads only its inputs and writes only its own four
# outputs — which is what makes it safe to run several of them at once.
# `ParquetReader.num_workers` says how many; 1, the default, calls it straight
# from `_load` on the calling thread with no threading machinery in the way.
#
# There are two independent things to run at once, and they share one work
# list rather than nesting. A task is a *(row group, leaf)* pair:
#
#   * across leaves of one row group — the axis whose ceiling is the slowest
#     column chunk, because once the widest column has a thread the rest run
#     out of work;
#   * across row groups — which has width exactly where the other one does
#     not, and is only available where every row group is being read anyway.
#
# `_decode_groups` flattens every pair it is asked for into one flat list and
# hands it to one `parallel_for`, so `num_workers` is a single budget for both
# axes and nesting can never oversubscribe it. The streaming path asks for one
# row group and gets the first axis alone; `read_table` asks for a window of
# them and gets both.
#
# A third axis, Arrow assembly, sits after this one and is written up above
# `_BatchPlan`. It is a separate fan-out rather than more tasks in this one
# because a batch cannot be assembled until every leaf of its row group has
# been decoded, and `parallel_for` has no barrier in the middle.


comptime _RECORD_STRIDE = 1024
"""Records between checkpoints in a nested leaf's row index.

The index exists to answer a handful of questions per row group — one per
batch boundary — so storing an answer per row is storing a thousand answers
nobody asks. A checkpoint every `_RECORD_STRIDE` records costs `rows / 1024`
`Int`s instead of `rows + 1`, and a lookup finishes by walking at most this
many records' worth of levels from the nearest one. Neither reference builds
an index at all — parquet-cpp's `TypedRecordReader::DelimitRecords` and
arrow-rs's `read_records` delimit as they read and keep only a position — so
this is the shape that keeps the sampling honest without keeping the array.
"""


struct _RowIndex(Copyable, Defaultable, Movable):
    """How to find a row inside one decoded column chunk.

    Three cases, and only the last one stores anything per row group:

    * `flat` and `dense` — one slot per row and every slot holds a value, so
      the row index *is* the slot index *is* the value index. Nothing stored.
    * `flat` — one slot per row, but some are null, so the value index is a
      running count of the non-null ones. Answered from `ColumnData.page_slot`
      / `page_value`, which the page walk fills in for free.
    * nested — a row spans as many slots as the repetition levels say, so
      finding one means counting `rep == 0`. Checkpointed every
      `_RECORD_STRIDE` records; a lookup walks forward from the last one.
    """

    var flat: Bool
    """`max_rep == 0`: one slot per row, so the slot index is the row index."""
    var dense: Bool
    """Every slot of the chunk holds a value, so the value index is the slot
    index and no null-counting is needed anywhere."""
    var max_def: Int
    var rows: Int
    var total_slots: Int
    var total_values: Int
    var slot_cp: List[Int]
    """Nested only: the first slot of record `k * _RECORD_STRIDE`."""
    var value_cp: List[Int]
    """Nested only, and only when the chunk has nulls: the first value of
    record `k * _RECORD_STRIDE`."""

    def __init__(out self):
        self.flat = False
        self.dense = True
        self.max_def = 0
        self.rows = 0
        self.total_slots = 0
        self.total_values = 0
        self.slot_cp = List[Int]()
        self.value_cp = List[Int]()

    def __init__(out self, *, copy: Self):
        self.flat = copy.flat
        self.dense = copy.dense
        self.max_def = copy.max_def
        self.rows = copy.rows
        self.total_slots = copy.total_slots
        self.total_values = copy.total_values
        self.slot_cp = copy.slot_cp.copy()
        self.value_cp = copy.value_cp.copy()

    def __init__(out self, *, deinit move: Self):
        self.flat = move.flat
        self.dense = move.dense
        self.max_def = move.max_def
        self.rows = move.rows
        self.total_slots = move.total_slots
        self.total_values = move.total_values
        self.slot_cp = move.slot_cp^
        self.value_cp = move.value_cp^

    @always_inline
    def record_at(self, cd: ColumnData, r: Int) -> Tuple[Int, Int]:
        """The first slot and the first value of record `r`.

        `r == rows` is the end of the chunk, which is a stored answer; anything
        else walks the levels forward from the nearest checkpoint, which is at
        most `_RECORD_STRIDE` records back.
        """
        if r >= self.rows:
            return (self.total_slots, self.total_values)
        var c = r // _RECORD_STRIDE
        var k = self.slot_cp[c]
        var rec = c * _RECORD_STRIDE
        var reps = cd.reps.unsafe_ptr()
        if self.dense:
            while rec < r:
                k += 1
                if k >= self.total_slots:
                    break
                if reps.unsafe_load(k) == 0:
                    rec += 1
            # Dense: every slot holds a value, so the value index is the slot.
            return (k, k)
        var v = self.value_cp[c]
        var defs = cd.defs.unsafe_ptr()
        var md = UInt16(self.max_def)
        while rec < r:
            if defs.unsafe_load(k) == md:
                v += 1
            k += 1
            if k >= self.total_slots:
                break
            if reps.unsafe_load(k) == 0:
                rec += 1
        return (k, v)


def _decode_leaf[
    Codecs: CodecSet
](
    file: FileBytes,
    cm: ColumnMetaData,
    leaf: LeafColumn,
    verify_crc: Bool,
    rows: Int,
    mut data: ColumnData,
    mut index: _RowIndex,
) raises:
    """Decode one column chunk of one row group, and index it by row.

    A pure function of its inputs: it reads the file bytes, which nothing
    mutates for the life of a reader, and writes only into its own two
    outputs. Both the sequential and the threaded path in `_load` call this and
    nothing else, so what a column decodes to cannot depend on which one ran,
    or on how many workers it ran with.

    The outputs are written rather than returned so that a threaded task can
    aim them straight at its own slots in `_LoadCtx`, with nothing to move
    afterwards.
    """
    data = read_column_chunk[Codecs](file, cm, leaf, verify_crc)
    var max_def = leaf.max_def
    var max_rep = leaf.max_rep
    var nvals = 0
    index = _RowIndex()
    index.max_def = max_def
    index.dense = data.all_present
    index.rows = rows
    index.total_slots = data.num_slots
    if max_rep == 0:
        # One slot per row, so the slot index *is* the row index, and the value
        # index of any row is a count of non-null slots the page walk has
        # already checkpointed. Nothing to build here at all.
        index.flat = True
        if data.num_slots != rows:
            raise Error(
                String(
                    "parquet: column '",
                    leaf.dotted(),
                    "' has ",
                    data.num_slots,
                    " value(s) in a row group of ",
                    rows,
                    " rows",
                )
            )
        if not data.all_present:
            # Levels are either one bitmap or one `UInt16` per slot; whichever
            # this chunk has, it has to reach as far as the row group claims.
            var have = len(data.mask) * 8 if data.masked() else len(data.defs)
            if rows > have:
                raise Error(_short_levels(leaf, "definition"))
        return
    var nslots = data.num_slots
    var all_present = data.all_present
    if nslots > len(data.reps):
        raise Error(_short_levels(leaf, "repetition"))
    if not all_present and nslots > len(data.defs):
        raise Error(_short_levels(leaf, "definition"))
    # The scan below is not optional — `records != rows` is a corruption check
    # every nested chunk has to pass — but what it *stores* is: one checkpoint
    # per `_RECORD_STRIDE` records rather than an entry per row.
    var ncp = rows // _RECORD_STRIDE + 1
    index.slot_cp.resize(ncp, 0)
    if not all_present:
        index.value_cp.resize(ncp, 0)
    var sp = index.slot_cp.unsafe_ptr()
    var vp = index.value_cp.unsafe_ptr()
    var reps = data.reps.unsafe_ptr()
    var defs = data.defs.unsafe_ptr()
    var records = 0
    var next_cp = 0
    var cp = 0
    for k in range(nslots):
        if reps.unsafe_load(k) == 0:
            # A chunk that starts more records than the row group claims is
            # caught right after the loop; up to then, only write inside the
            # sized buffer.
            if records == next_cp and cp < ncp:
                sp.unsafe_store(cp, k)
                if not all_present:
                    vp.unsafe_store(cp, nvals)
                cp += 1
                next_cp += _RECORD_STRIDE
            records += 1
        if all_present or Int(defs.unsafe_load(k)) == max_def:
            nvals += 1
    if records != rows:
        raise Error(
            String(
                "parquet: column '",
                leaf.dotted(),
                "' assembles ",
                records,
                " record(s) in a row group of ",
                rows,
                " rows",
            )
        )
    index.total_values = nvals


@always_inline
def _leaf_slices(
    needed: List[Bool],
    chunks: List[ColumnData],
    index: List[_RowIndex],
    r0: Int,
    r1: Int,
) -> List[LeafSlice]:
    """Where rows `[r0, r1)` sit in each leaf's decoded chunk.

    One entry per leaf of the file schema, `LeafSlice()` for the ones this read
    does not want. A pure function of the decoded chunk and its row index, so
    it does not matter which leaf ran on which thread, or whether the row group
    came out of `_load` or out of a prefetched window: the same row range gives
    the same slices. Both paths call this and nothing else, which is what makes
    that true rather than merely intended.

    `@always_inline` is not decoration: handing the row indices to an
    out-of-line call cost **9.6% of `bench_read_big` on a sequential read** —
    measured, and the entire cost of pulling this body out of `_assemble`.
    """
    var slices = List[LeafSlice]()
    for i in range(len(needed)):
        if not needed[i]:
            slices.append(LeafSlice())
            continue
        ref ix = index[i]
        var s0 = r0
        var s1 = r1
        var v0 = r0
        if not ix.flat:
            if r1 > ix.rows:
                slices.append(LeafSlice())
                continue
            var at = ix.record_at(chunks[i], r0)
            s0 = at[0]
            v0 = at[1]
            s1 = ix.record_at(chunks[i], r1)[0]
        elif r1 > chunks[i].num_slots:
            slices.append(LeafSlice())
            continue
        elif not ix.dense:
            # Every slot is a row here, so the value index is how many of the
            # rows before this one held a value.
            v0 = chunks[i].value_at(r0, ix.max_def)
        slices.append(LeafSlice(s0, s1, v0))
    return slices^


struct _RowGroupData(Defaultable, Movable):
    """One row group's decode state — what a loaded reader holds.

    One slot per leaf of the *file* schema, not per projected leaf, so a leaf's
    index here is the same number it has everywhere else and a projection
    leaves empty slots rather than renumbering anything.
    """

    var rg: Int
    """The row group this holds, or -1 once it has been handed to the reader."""
    var chunks: List[ColumnData]
    var index: List[_RowIndex]

    def __init__(out self):
        self.rg = -1
        self.chunks = List[ColumnData]()
        self.index = List[_RowIndex]()

    def __init__(out self, *, deinit move: Self):
        self.rg = move.rg
        self.chunks = move.chunks^
        self.index = move.index^

    # The two outputs, moved out; the value keeps an empty list in place. A
    # field cannot be transferred out of the middle of a live value, and this
    # one is still live at that point, so each one swaps rather than moves.

    def take_chunks(mut self) -> List[ColumnData]:
        var taken = self.chunks^
        self.chunks = List[ColumnData]()
        return taken^

    def take_index(mut self) -> List[_RowIndex]:
        var taken = self.index^
        self.index = List[_RowIndex]()
        return taken^


struct _LoadCtx(Movable):
    """Everything a threaded decode reads, and where it writes.

    A task is one *(row group, leaf)* pair, and both axes of parallelism go
    through this one flat list: the streaming path asks for a single row group
    and gets the per-column fan-out, `read_table` asks for a window of them and
    gets that fan-out across every row group in the window at once. Either way
    there is one pool drawing from one queue, so composing the axes cannot
    oversubscribe `num_workers`.

    The outputs are sized `ngroups * nleaves` and written at
    `group * nleaves + leaf` — the task's own `dest[k]`, which no other task
    touches. So the tasks need no lock, the file bytes they all read are
    immutable for the life of the reader, and the lists come out group-major
    and leaf-minor: exactly the order the sequential loop produces them in,
    which is what makes row order independent of which worker finished when.
    """

    var file: FileBytes
    var verify_crc: Bool
    var nleaves: Int
    var rgs: List[Int]
    """The row group each group slot holds, in visit order."""
    var metas: List[ColumnMetaData]
    """The chunk metadata of each queued leaf, copied so no worker reads
    through the reader while the caller holds it mutably."""
    var leaves: List[LeafColumn]
    var rows: List[Int]
    var dest: List[Int]
    """Task `k` writes the output slots at `dest[k]`."""
    var chunks: List[ColumnData]
    var index: List[_RowIndex]
    var errors: List[String]
    """Per output slot, empty when it decoded. A task cannot raise — pthread
    has no exception channel — so a failure lands in its own slot and the
    caller re-raises it in group-then-leaf order after the join, which is why
    the error a corrupt file gives does not depend on which worker lost."""
    var fatal: List[String]
    """Per group, a failure that stops a whole row group before any of its
    leaves is queued. Recorded rather than raised on the spot so that the
    *first* failure in visit order is the one that reaches the caller, which is
    what the sequential path does."""

    def __init__(
        out self,
        file: FileBytes,
        verify_crc: Bool,
        nleaves: Int,
        ngroups: Int,
    ):
        self.file = file
        self.verify_crc = verify_crc
        self.nleaves = nleaves
        self.rgs = List[Int](length=ngroups, fill=-1)
        self.metas = List[ColumnMetaData]()
        self.leaves = List[LeafColumn]()
        self.rows = List[Int]()
        self.dest = List[Int]()
        self.chunks = List[ColumnData]()
        self.index = List[_RowIndex]()
        self.errors = List[String]()
        self.fatal = List[String]()
        for _ in range(ngroups * nleaves):
            self.chunks.append(ColumnData())
            self.index.append(_RowIndex())
            self.errors.append(String())
        for _ in range(ngroups):
            self.fatal.append(String())

    def __init__(out self, *, deinit move: Self):
        self.file = move.file
        self.verify_crc = move.verify_crc
        self.nleaves = move.nleaves
        self.rgs = move.rgs^
        self.metas = move.metas^
        self.leaves = move.leaves^
        self.rows = move.rows^
        self.dest = move.dest^
        self.chunks = move.chunks^
        self.index = move.index^
        self.errors = move.errors^
        self.fatal = move.fatal^

    def take_group(mut self, g: Int) -> _RowGroupData:
        """Group `g`'s outputs, lifted out of the flat lists.

        Swapped out one slot at a time, leaving empties behind: the context is
        still live, and a `ColumnData` holds the decoded buffers, so this must
        move rather than copy.
        """
        var out = _RowGroupData()
        out.rg = self.rgs[g]
        var base = g * self.nleaves
        for i in range(self.nleaves):
            var cd = ColumnData()
            swap(cd, self.chunks[base + i])
            out.chunks.append(cd^)
            var ix = _RowIndex()
            swap(ix, self.index[base + i])
            out.index.append(ix^)
        return out^


def _decode_leaf_task[Codecs: CodecSet](k: Int, mut ctx: _LoadCtx) -> None:
    """One column chunk of one row group, on whichever worker drew task `k`.

    Writes only the two output slots at `dest[k]` — or, on a failure, that
    slot's error cell — neither of which another task touches, and reads only
    fields nothing writes while the fan-out runs.
    """
    var d = ctx.dest[k]
    try:
        _decode_leaf[Codecs](
            ctx.file,
            ctx.metas[k],
            ctx.leaves[k],
            ctx.verify_crc,
            ctx.rows[k],
            ctx.chunks[d],
            ctx.index[d],
        )
    except e:
        ctx.errors[d] = String(e)


# ── the third axis: Arrow assembly ─────────────────────────────────────────
#
# Decoding a column chunk turns file bytes into values and levels; *assembly*
# turns those into Arrow buffers, one array per Arrow field. Once the decode
# fan-out landed, assembly was the whole serial remainder of a read — 32% of
# the mixed benchmark and 16% of the wide one — and Amdahl's law on those two
# numbers predicted the scaling that was actually observed at eight workers to
# within a few percent. So it is the ceiling, and it is the axis below.
#
# A task is one *(batch, top-level field)* pair. Top-level fields are already
# independent: `build_field` reads the decoded chunks and the row slices and
# writes only into the arena it is handed, and a nested field recurses into
# that same arena. Batches are independent for the same reason, and a window
# of row groups has several of them in flight, so both go into one flat work
# list handed to one `parallel_for` — one pool per window for assembly, next to
# the one the same window already used for decode, rather than one per batch.
#
# **Arena layout does not move.** `ArrayData` refers to its children by arena
# index, so assembling fields in a different order would shift every index and
# produce a structurally different batch out of identical values — the kind of
# wrong a value-comparing test cannot see. Each task therefore builds into its
# *own* arena, starting at index 0, and the calling thread grafts those arenas
# into the batch in field order afterwards (`ArrayArena.graft`), shifting every
# child index by the graft point. A subtree built alone and grafted at `B` is
# byte for byte the subtree built directly into a shared arena that already
# held `B` nodes, so the batch that comes out is the sequential one at every
# worker count — not merely one holding the same values.


@fieldwise_init
struct _BatchPlan(Copyable, Defaultable, Movable):
    """One batch a window will cut, decided before any of it is built."""

    var rg: Int
    var group: Int
    """Which decoded row group holds it, or -1 for a row group with no rows —
    which `_prefetch` skips and which is built on the calling thread."""
    var r0: Int
    var r1: Int

    def __init__(out self):
        self.rg = -1
        self.group = -1
        self.r0 = 0
        self.r1 = 0


struct _AssembleCtx(Movable):
    """Everything the assembly fan-out reads, and where each task writes.

    Task `t` builds field `field[t]` of batch `batch[t]` into `arenas[t]` — its
    own arena, which no other task touches — and records the root's index
    inside it in `local_root[t]`, or its failure in `errors[t]`. Everything
    else is read-only for the length of the call: the schema and the include
    mask are copies, the row groups were moved in whole, and the slices were
    computed on the calling thread before any worker started.

    Tasks are queued batch-major and field-minor, which is visit order, so
    scanning `errors` in task order gives the caller the same first failure the
    sequential path gives.
    """

    var schema: ParquetSchema
    var include: List[Bool]
    var groups: List[_RowGroupData]
    """The window's decoded row groups, moved out of `_prefetched`."""
    var slices: List[List[LeafSlice]]
    """Per planned batch, where its rows sit in each leaf's chunk."""
    var batch: List[Int]
    var group_of: List[Int]
    var field: List[Int]
    var arenas: List[ArrayArena]
    var local_root: List[Int]
    var errors: List[String]

    def __init__(
        out self,
        var schema: ParquetSchema,
        var include: List[Bool],
        var groups: List[_RowGroupData],
    ):
        self.schema = schema^
        self.include = include^
        self.groups = groups^
        self.slices = List[List[LeafSlice]]()
        self.batch = List[Int]()
        self.group_of = List[Int]()
        self.field = List[Int]()
        self.arenas = List[ArrayArena]()
        self.local_root = List[Int]()
        self.errors = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.schema = move.schema^
        self.include = move.include^
        self.groups = move.groups^
        self.slices = move.slices^
        self.batch = move.batch^
        self.group_of = move.group_of^
        self.field = move.field^
        self.arenas = move.arenas^
        self.local_root = move.local_root^
        self.errors = move.errors^


def _assemble_task(t: Int, mut ctx: _AssembleCtx) -> None:
    """One top-level field of one batch, on whichever worker drew task `t`.

    Writes `arenas[t]`, `local_root[t]` and `errors[t]` and nothing else; reads
    the decoded chunks, which nothing mutates while the fan-out runs.
    """
    try:
        ctx.local_root[t] = build_field(
            ctx.schema,
            ctx.field[t],
            0,
            ctx.groups[ctx.group_of[t]].chunks,
            ctx.slices[ctx.batch[t]],
            ctx.arenas[t],
            ctx.include,
        )
    except e:
        ctx.errors[t] = String(e)


struct ParquetReader[Codecs: CodecSet = DefaultCodecs](Movable):
    """A Parquet file, projected and batched."""

    var _owned: List[UInt8]
    """The file bytes when this reader owns them; empty when it borrows."""
    var data: FileBytes
    """The file bytes, however they are held. Points into `_owned` when this
    reader owns them."""
    var meta: FileMetaData
    var schema: ParquetSchema
    var batch_size: Int
    var verify_crc: Bool
    var num_workers: Int
    """How many OS threads turn file bytes into Arrow at once.

    `1`, the default, does all of it on the calling thread and is what every
    existing caller gets: the sequential path is the same code it always was,
    with no pool started and no task context built. `0` means one worker per
    core. Anything else is that many workers, clamped down to the number of
    tasks there actually are — a two-column read gains nothing from ten
    threads.

    A column chunk is the unit of *decoding* because it is where most of the
    time is: on the wide benchmark, dictionary gather, value decoding,
    decompression and the per-row index together are about 85% of a read, and
    they are all inside it. Chunks are independent through decode, so the tasks
    share nothing but the immutable file bytes and the output is identical
    whatever the worker count — asserted over the whole fixture corpus by
    `test_num_workers_is_bit_identical`.

    **Three axes, one budget.** This number is the whole thread budget, and
    every axis draws from it:

    * every read fans out across the leaves of the row group it is loading;
    * `read_table` *additionally* fans out across row groups, because it is
      the one entry point that was always going to visit all of them;
    * and it fans out Arrow *assembly* across the *(batch, top-level field)*
      pairs of a window, because once decoding was threaded, assembly on the
      calling thread was the whole serial remainder of a read.

    Per-leaf parallelism alone is bounded by the slowest column chunk — one
    fat dictionary-encoded string column is most of a wide read on its own, so
    past two or four threads the others idle. Row groups are the axis that
    still has width there. The axes never nest: a decode task is one *(row
    group, leaf)* pair and an assembly task is one *(batch, top-level field)*
    pair, each flattened into a single work list for a single pool, so
    `num_workers = 4` means four threads and not four times however many
    columns.

    **Memory.** The streaming path — `read_batch`, `has_next`, and the reader's
    own iteration — is untouched and still holds exactly one row group's decode
    state, whatever this is set to. `read_table` decodes up to `num_workers`
    row groups at a time (see `read_table`), so past `1` it trades a bounded
    amount of peak memory for the second axis. The third axis trades none: an
    assembly task builds the arrays the batch was going to hold anyway, into a
    small arena that is grafted into the batch and dropped. Nothing changes at
    `1`.
    """
    var _row_groups: List[Int]
    var _roots: List[Int]
    """Selected top-level Arrow field indices."""
    var _needed: List[Bool]
    """Per leaf: is it under a selected root?"""
    var _include: List[Bool]
    """Per Arrow field: is it built? Empty means "everything under a root"."""
    var _rg_pos: Int
    var _range_pos: Int
    var _row_pos: Int
    var _did_empty: Bool
    """Whether the empty row group we are sitting on has been handed out."""
    var _ranges: List[List[Tuple[Int, Int]]]
    """Per selected row group, the row ranges left after page pruning. An
    empty list means the whole row group."""
    var _loaded_rg: Int
    var _chunks: List[ColumnData]
    var _row_index: List[_RowIndex]
    """Per leaf, how to turn a row number into a slot and a value index."""
    var _prefetched: List[_RowGroupData]
    """Row groups `read_table` decoded ahead, in visit order.

    Only ever non-empty inside a threaded `read_table`: `_load` takes a group
    out of here instead of decoding it, and everything downstream of `_load` —
    batching, assembly, row order — is the code it always was. Empty for every
    other caller, which is what keeps the streaming memory contract intact.
    """

    def __init__(out self, var data: List[UInt8]) raises:
        """Take ownership of the file bytes."""
        var owned = data^
        # Build over the buffer first, then hand it to the reader. Moving the
        # `List` afterwards moves the handle, not the heap buffer the span
        # points at, so `self.data` stays valid.
        self = Self(unsafe_borrowing=_as_file_bytes(owned))
        self._owned = owned^

    @staticmethod
    def from_span(data: Span[UInt8, _]) raises -> Self:
        """Read a file out of a buffer this reader does **not** own.

        The caller must keep `data` alive for as long as the reader and must
        not mutate it; nothing checks either. Use it when the bytes already
        exist somewhere that outlives the read -- a mapped file, a shared
        arena, a benchmark timing many reads over one buffer. Prefer the
        owning constructor otherwise.
        """
        return Self(unsafe_borrowing=rebind[FileBytes](data))

    def __init__(out self, *, unsafe_borrowing: FileBytes) raises:
        """Initialise every field over a buffer owned by someone else.

        `from_span` is the public spelling. This exists so the owning
        constructor can share it rather than keeping a second copy in sync.
        """
        self._owned = List[UInt8]()
        self.data = unsafe_borrowing
        self.meta = read_footer(self.data)
        self.schema = build_schema(self.meta.schema)
        self.batch_size = 65536
        self.verify_crc = True
        self.num_workers = 1
        self._row_groups = List[Int]()
        for i in range(len(self.meta.row_groups)):
            self._row_groups.append(i)
        self._roots = self.schema.roots.copy()
        self._needed = List[Bool](length=len(self.schema.leaves), fill=True)
        self._include = List[Bool]()
        self._rg_pos = 0
        self._range_pos = 0
        self._row_pos = 0
        self._did_empty = False
        self._ranges = List[List[Tuple[Int, Int]]]()
        self._loaded_rg = -1
        self._chunks = List[ColumnData]()
        self._row_index = List[_RowIndex]()
        self._prefetched = List[_RowGroupData]()

    def __init__(out self, *, deinit move: Self):
        self._owned = move._owned^
        # Still valid: the heap buffer did not move, only the handle to it.
        self.data = move.data
        self.meta = move.meta^
        self.schema = move.schema^
        self.batch_size = move.batch_size
        self.verify_crc = move.verify_crc
        self.num_workers = move.num_workers
        self._row_groups = move._row_groups^
        self._roots = move._roots^
        self._needed = move._needed^
        self._include = move._include^
        self._rg_pos = move._rg_pos
        self._range_pos = move._range_pos
        self._row_pos = move._row_pos
        self._did_empty = move._did_empty
        self._ranges = move._ranges^
        self._loaded_rg = move._loaded_rg
        self._chunks = move._chunks^
        self._row_index = move._row_index^
        self._prefetched = move._prefetched^

    @staticmethod
    def open(path: StringSlice) raises -> Self:
        return Self(read_parquet_file(String(path)))

    # ── metadata ───────────────────────────────────────────────────────────

    def num_rows(self) -> Int:
        return Int(self.meta.num_rows)

    def num_row_groups(self) -> Int:
        return len(self.meta.row_groups)

    def created_by(self) -> String:
        if self.meta.created_by:
            return self.meta.created_by.value().copy()
        return String()

    def key_value_metadata(self) raises -> List[Tuple[String, String]]:
        var out = List[Tuple[String, String]]()
        if self.meta.key_value_metadata:
            for kv in self.meta.key_value_metadata.value():
                var v = String()
                if kv.value:
                    v = kv.value.value().copy()
                out.append((kv.key.copy(), v^))
        return out^

    def split_offsets(self) -> List[Int64]:
        """Iceberg's `data_file.split_offsets`: where each row group starts."""
        var out = List[Int64](capacity=len(self.meta.row_groups))
        for rg in self.meta.row_groups:
            if rg.file_offset:
                out.append(rg.file_offset.value())
            elif len(rg.columns) and rg.columns[0].meta_data:
                ref cm = rg.columns[0].meta_data.value()
                var start = Int64(cm.data_page_offset)
                if cm.dictionary_page_offset:
                    var d = cm.dictionary_page_offset.value()
                    if d > 0 and d < start:
                        start = d
                out.append(start)
            else:
                out.append(0)
        return out^

    def statistics(self, rg: Int, leaf: Int) raises -> TypedStats:
        """The decoded statistics of one column chunk, empty if absent."""
        ref chunk = self.meta.row_groups[rg].columns[leaf]
        if not chunk.meta_data:
            return TypedStats()
        ref cm = chunk.meta_data.value()
        if not cm.statistics:
            return TypedStats()
        return decode_stats(self.schema.leaves[leaf], cm.statistics.value())

    def offset_index(self, rg: Int, leaf: Int) raises -> Optional[OffsetIndex]:
        ref chunk = self.meta.row_groups[rg].columns[leaf]
        if not chunk.offset_index_offset or not chunk.offset_index_length:
            return None
        var off = Int(chunk.offset_index_offset.value())
        var n = Int(chunk.offset_index_length.value())
        if off < 0 or n < 0 or off + n > len(self.data):
            raise Error("parquet: offset index runs past the end of the file")
        var p = TCompactProtocolReader(self.data[off : off + n])
        var oi = OffsetIndex()
        oi.read(p)
        return oi^

    def column_index(self, rg: Int, leaf: Int) raises -> Optional[ColumnIndex]:
        ref chunk = self.meta.row_groups[rg].columns[leaf]
        if not chunk.column_index_offset or not chunk.column_index_length:
            return None
        var off = Int(chunk.column_index_offset.value())
        var n = Int(chunk.column_index_length.value())
        if off < 0 or n < 0 or off + n > len(self.data):
            raise Error("parquet: column index runs past the end of the file")
        var p = TCompactProtocolReader(self.data[off : off + n])
        var ci = ColumnIndex()
        ci.read(p)
        return ci^

    # ── projection ─────────────────────────────────────────────────────────

    def _mark(mut self, fi: Int):
        if self.schema.fields[fi].leaf >= 0:
            self._needed[self.schema.fields[fi].leaf] = True
            return
        var kids = self.schema.fields[fi].children.copy()
        for c in kids:
            self._mark(c)

    def _set_roots(mut self, var roots: List[Int]):
        self._roots = roots^
        self._include = List[Bool]()
        for i in range(len(self._needed)):
            self._needed[i] = False
        var rs = self._roots.copy()
        for r in rs:
            self._mark(r)
        self._loaded_rg = -1
        self._drop_prefetch()

    def _include_subtree(self, fi: Int, mut inc: List[Bool]):
        inc[fi] = True
        ref kids = self.schema.fields[fi].children
        if len(kids) == 0:
            return
        for k in range(len(kids)):
            self._include_subtree(kids[k], inc)

    def select_fields(mut self, fields: List[Int]) raises:
        """Project by Arrow field index, sub-fields included.

        A selected field brings its whole sub-tree; every ancestor is kept as a
        wrapper, so the value still arrives inside the struct it belongs to,
        and nothing else is decoded. Roots come out in the order they were
        asked for, which is the contract `select_columns` already has.
        `select_columns` is the special case where every selection is a root.
        """
        var n = len(self.schema.fields)
        var parent = List[Int](length=n, fill=-1)
        for f in range(n):
            ref kids = self.schema.fields[f].children
            for k in range(len(kids)):
                parent[kids[k]] = f
        var inc = List[Bool](length=n, fill=False)
        for i in range(len(fields)):
            var f = fields[i]
            if f < 0 or f >= n:
                raise Error(String("parquet: no field at index ", f))
            self._include_subtree(f, inc)
            var p = parent[f]
            while p >= 0:
                inc[p] = True
                p = parent[p]
        # A map is only a map with both halves: an included map keeps its key.
        for f in range(n):
            if not inc[f] or self.schema.fields[f].type.id != AT_MAP:
                continue
            ref kids = self.schema.fields[f].children
            if len(kids) == 0:
                continue
            var entries = kids[0]
            inc[entries] = True
            ref pair = self.schema.fields[entries].children
            if len(pair) > 0:
                self._include_subtree(pair[0], inc)
        var roots = List[Int]()
        for i in range(len(fields)):
            var top = fields[i]
            while parent[top] >= 0:
                top = parent[top]
            var seen = False
            for k in range(len(roots)):
                if roots[k] == top:
                    seen = True
                    break
            if not seen:
                roots.append(top)
        self._roots = roots^
        for i in range(len(self._needed)):
            self._needed[i] = False
        for f in range(n):
            if inc[f] and self.schema.fields[f].leaf >= 0:
                self._needed[self.schema.fields[f].leaf] = True
        self._include = inc^
        self._loaded_rg = -1
        self._drop_prefetch()

    def select_field_ids_deep(mut self, ids: List[Int32]) raises:
        """`select_fields`, addressing the fields by Parquet field id."""
        var fields = List[Int]()
        for i in range(len(ids)):
            var fi = self.schema.field_by_id(ids[i])
            if fi < 0:
                raise Error(String("parquet: no field with id ", ids[i]))
            fields.append(fi)
        self.select_fields(fields)

    def select_all(mut self):
        self._set_roots(self.schema.roots.copy())

    def select_columns(mut self, names: List[String]) raises:
        """Project by top-level column name."""
        var roots = List[Int]()
        for n in names:
            var fi = self.schema.field_by_name(n)
            if fi < 0:
                raise Error(String("parquet: no column named '", n, "'"))
            roots.append(fi)
        self._set_roots(roots^)

    def select_field_ids(mut self, ids: List[Int32]) raises:
        """Project by Parquet field id — what Iceberg schema evolution needs."""
        var roots = List[Int]()
        for id in ids:
            var fi = self.schema.field_by_id(id)
            if fi < 0:
                raise Error(String("parquet: no field with id ", id))
            roots.append(fi)
        self._set_roots(roots^)

    def select_row_groups(mut self, var groups: List[Int]) raises:
        for g in groups:
            if g < 0 or g >= len(self.meta.row_groups):
                raise Error(String("parquet: row group ", g, " does not exist"))
        self._row_groups = groups^
        self._ranges = List[List[Tuple[Int, Int]]]()
        self.rewind()

    def prune_row_groups(mut self, predicates: List[Predicate]) raises -> Int:
        """Drop row groups whose statistics prove no row can match. Returns
        the number of row groups left."""
        var keep = List[Int]()
        for g in self._row_groups:
            if self._rg_matches(g, predicates):
                keep.append(g)
        self._row_groups = keep^
        self._ranges = List[List[Tuple[Int, Int]]]()
        self.rewind()
        return len(self._row_groups)

    def _rg_matches(self, rg: Int, predicates: List[Predicate]) raises -> Bool:
        for p in predicates:
            var leaf = self.schema.leaf_by_path(p.column)
            if leaf < 0:
                continue
            var st = self.statistics(rg, leaf)
            if not st.has_min_max:
                continue
            if not range_can_match(p.op, st.min, st.max, p.value):
                return False
        return True

    # ── reading ────────────────────────────────────────────────────────────

    def rewind(mut self):
        self._rg_pos = 0
        self._range_pos = 0
        self._row_pos = 0
        self._did_empty = False
        self._loaded_rg = -1
        self._drop_prefetch()

    def _resolved_workers(self) -> Int:
        """`num_workers` with `0` spelt out as the core count."""
        var w = self.num_workers
        if w == 0:
            w = num_cpus()
        if w < 1:
            w = 1
        return w

    def _drop_prefetch(mut self):
        self._prefetched = List[_RowGroupData]()

    def _take_prefetched(mut self, rg: Int) -> Bool:
        """Adopt a row group `read_table` already decoded, if it is there.

        A miss is not an error: the streaming path never prefetches, a row
        group can appear twice in an explicit `select_row_groups`, and a group
        is only ever handed out once. Whoever misses decodes it the ordinary
        way.
        """
        for g in range(len(self._prefetched)):
            if self._prefetched[g].rg != rg:
                continue
            self._prefetched[g].rg = -1
            self._chunks = self._prefetched[g].take_chunks()
            self._row_index = self._prefetched[g].take_index()
            self._loaded_rg = rg
            return True
        return False

    def _load(mut self, rg: Int) raises:
        if self._loaded_rg == rg:
            return
        var nleaves = len(self.schema.leaves)
        # Drop the row group being replaced before anything else, so that at
        # most one loaded group is alive at a time on this side of the reader.
        self._chunks = List[ColumnData]()
        self._row_index = List[_RowIndex]()
        if self._take_prefetched(rg):
            return
        var rows = Int(self.meta.row_groups[rg].num_rows)
        if len(self.meta.row_groups[rg].columns) != nleaves:
            raise Error(
                _bad_chunk_count(
                    rg, len(self.meta.row_groups[rg].columns), nleaves
                )
            )
        var workers = self._resolved_workers()
        if workers > 1:
            var projected = 0
            for i in range(nleaves):
                if self._needed[i]:
                    projected += 1
            if workers > projected:
                workers = projected
        if workers > 1:
            self._load_threaded(rg, nleaves, workers)
            return
        for i in range(nleaves):
            if not self._needed[i]:
                self._chunks.append(ColumnData())
                self._row_index.append(_RowIndex())
                continue
            ref chunk = self.meta.row_groups[rg].columns[i]
            if not chunk.meta_data:
                raise _no_chunk_meta(rg, i)
            var cd = ColumnData()
            var ix = _RowIndex()
            _decode_leaf[Self.Codecs](
                self.data,
                chunk.meta_data.value(),
                self.schema.leaves[i],
                self.verify_crc,
                rows,
                cd,
                ix,
            )
            self._chunks.append(cd^)
            self._row_index.append(ix^)
        self._loaded_rg = rg

    def _decode_groups(
        self, rgs: List[Int], nleaves: Int, workers: Int
    ) raises -> List[_RowGroupData]:
        """Decode every projected leaf of every row group in `rgs`, threaded.

        This is the whole fan-out, and the only one. Every *(row group, leaf)*
        pair becomes one task in one flat list handed to one `parallel_for`, so
        a window of row groups and the columns inside them draw from a single
        queue of `workers` threads: the axes compose by flattening, never by
        nesting a pool inside a pool.

        The metadata each task needs is copied into the context on the calling
        thread, before any worker starts, so no worker reads through the
        reader while it runs.

        Tasks are queued in group-then-leaf order, which is also file order,
        and that turns out to matter: `parallel_for` draws from one counter, so
        the queue order *is* the schedule. Queueing the largest chunks first —
        the textbook longest-processing-time-first fix for the tail — is worth
        12% at two workers on the wide fixture and costs 22% at eight on the
        mixed one, because it starts every fat dictionary column at once and
        they contend for memory bandwidth rather than for cores. File order
        spreads them out. This is measured, not assumed; see the PR.

        Failures do not race to the caller. A task writes into its own error
        cell, a row group that cannot be queued at all writes into its own
        `fatal` cell, and both are scanned in group-then-leaf order after the
        join — so a corrupt file raises with the first failure in *visit*
        order, the same message the sequential path gives, at any worker count.

        The result is one `_RowGroupData` per entry of `rgs`, in that order.
        """
        var ngroups = len(rgs)
        var ctx = _LoadCtx(self.data, self.verify_crc, nleaves, ngroups)
        for g in range(ngroups):
            var rg = rgs[g]
            ctx.rgs[g] = rg
            var cols = len(self.meta.row_groups[rg].columns)
            if cols != nleaves:
                ctx.fatal[g] = _bad_chunk_count(rg, cols, nleaves)
                continue
            var rows = Int(self.meta.row_groups[rg].num_rows)
            for i in range(nleaves):
                if not self._needed[i]:
                    continue
                ref chunk = self.meta.row_groups[rg].columns[i]
                if not chunk.meta_data:
                    ctx.errors[g * nleaves + i] = String(_no_chunk_meta(rg, i))
                    continue
                ctx.metas.append(chunk.meta_data.value().copy())
                ctx.leaves.append(self.schema.leaves[i].copy())
                ctx.rows.append(rows)
                ctx.dest.append(g * nleaves + i)
        var n = len(ctx.dest)
        # The typed `parallel_for` holds `ctx` for the whole call, joins
        # included, so nothing has to mention it afterwards to keep it alive.
        parallel_for[_decode_leaf_task[Self.Codecs]](
            n, ctx, num_workers=workers
        )
        for g in range(ngroups):
            if ctx.fatal[g]:
                raise Error(ctx.fatal[g])
            for i in range(nleaves):
                if ctx.errors[g * nleaves + i]:
                    raise Error(ctx.errors[g * nleaves + i])
        var out = List[_RowGroupData]()
        for g in range(ngroups):
            out.append(ctx.take_group(g))
        return out^

    def _load_threaded(mut self, rg: Int, nleaves: Int, workers: Int) raises:
        """`_load`'s body on `workers` threads, one task per projected leaf.

        A window of exactly one row group: the second axis has no width here,
        because the streaming path has no licence to decode a row group the
        caller has not asked for yet.
        """
        var one: List[Int] = [rg]
        var got = self._decode_groups(one, nleaves, workers)
        self._chunks = got[0].take_chunks()
        self._row_index = got[0].take_index()
        self._loaded_rg = rg

    def _prefetched_group(self, rg: Int) -> Int:
        """Where row group `rg` sits in the current window, or -1."""
        for g in range(len(self._prefetched)):
            if self._prefetched[g].rg == rg:
                return g
        return -1

    def _plan_window(mut self, past: Int) -> List[_BatchPlan]:
        """Cut every batch this window hands out, without building any of them.

        The batch walk of `read_batch`, with `_load` and `_assemble` taken out:
        it advances exactly the same iterator state — `_rg_pos`, `_range_pos`,
        `_row_pos` and `_did_empty` — over exactly the same row ranges, so the
        batches a window plans are the batches the sequential loop would have
        cut, in the same order and with the same boundaries. Nothing here reads
        a decoded chunk, so it can run before a single array is built.
        """
        var plans = List[_BatchPlan]()
        while self._seek():
            if self._rg_pos >= past:
                break
            var rg = self._row_groups[self._rg_pos]
            if Int(self.meta.row_groups[rg].num_rows) == 0:
                # One empty batch, so a zero-row file keeps its columns.
                # `_prefetch` skips these, hence no group to build it from.
                self._did_empty = True
                plans.append(_BatchPlan(rg, -1, 0, 0))
                continue
            var ranges = self._ranges_for(self._rg_pos)
            var span = ranges[self._range_pos]
            var r0 = self._row_pos
            var r1 = r0 + self.batch_size
            if r1 > span[1]:
                r1 = span[1]
            plans.append(_BatchPlan(rg, self._prefetched_group(rg), r0, r1))
            self._row_pos = r1
        return plans^

    def _assemble_window(
        mut self, plans: List[_BatchPlan], workers: Int, mut t: Table
    ) raises:
        """Build every batch of a planned window, and append them in order.

        One `parallel_for` over *(batch, top-level field)* pairs, then a graft
        of each task's private arena into its batch on this thread — see the
        note above `_BatchPlan` for why the graft, and not the fan-out, is what
        makes the arena layout independent of the worker count.

        The window's decoded row groups are moved out of `_prefetched` and into
        the context, so the tasks read them through a value nothing else holds
        and the reader is left with the empty `_prefetched` the caller expects
        after a window.
        """
        if len(plans) == 0:
            return
        var groups = self._prefetched^
        self._prefetched = List[_RowGroupData]()
        var ctx = _AssembleCtx(
            self.schema.copy(), self._include.copy(), groups^
        )
        # Slices first, on this thread: they are pure arithmetic over the
        # per-row indices, and computing them here keeps the tasks to the one
        # thing that is worth spreading.
        for p in range(len(plans)):
            var g = plans[p].group
            if g < 0:
                ctx.slices.append(List[LeafSlice]())
                continue
            ctx.slices.append(
                _leaf_slices(
                    self._needed,
                    ctx.groups[g].chunks,
                    ctx.groups[g].index,
                    plans[p].r0,
                    plans[p].r1,
                )
            )
        # Batch-major, field-minor: task order is visit order.
        for p in range(len(plans)):
            if plans[p].group < 0:
                continue
            for k in range(len(self._roots)):
                ctx.batch.append(p)
                ctx.group_of.append(plans[p].group)
                ctx.field.append(self._roots[k])
                ctx.arenas.append(ArrayArena())
                ctx.local_root.append(0)
                ctx.errors.append(String())
        parallel_for[_assemble_task](len(ctx.batch), ctx, num_workers=workers)
        var task = 0
        for p in range(len(plans)):
            if plans[p].group < 0:
                # A row group with no rows: decoded and assembled here, in its
                # place in visit order, exactly as the sequential loop does it.
                self._load(plans[p].rg)
                var empty = self._assemble(0, 0)
                t.num_rows += empty.num_rows
                t.batches.append(empty^)
                continue
            var b = RecordBatch()
            b.num_rows = plans[p].r1 - plans[p].r0
            for _k in range(len(self._roots)):
                # Checked as the batches are stitched rather than all at once,
                # so a failure reaches the caller at the point in visit order
                # where the sequential path would have raised it.
                if ctx.errors[task]:
                    raise Error(ctx.errors[task])
                var at = b.arena.graft(ctx.arenas[task], ctx.local_root[task])
                b.roots.append(at)
                task += 1
            t.num_rows += b.num_rows
            t.batches.append(b^)

    def _prefetch(mut self, first: Int, past: Int, workers: Int) raises:
        """Decode the row groups at selection slots `[first, past)` in one go.

        Skips what the iterator will never load — a row group with no rows, one
        page pruning emptied, and a repeat of one already in this window — so
        that a threaded `read_table` decodes exactly the row groups a
        sequential one does and no more.
        """
        var rgs = List[Int]()
        for s in range(first, past):
            var rg = self._row_groups[s]
            if Int(self.meta.row_groups[rg].num_rows) == 0:
                continue
            if len(self._ranges_for(s)) == 0:
                continue
            var seen = False
            for k in range(len(rgs)):
                if rgs[k] == rg:
                    seen = True
                    break
            if seen:
                continue
            rgs.append(rg)
        if len(rgs) == 0:
            return
        var got = self._decode_groups(rgs, len(self.schema.leaves), workers)
        self._prefetched = got^

    def _ranges_for(self, slot: Int) -> List[Tuple[Int, Int]]:
        """The row ranges to read from the row group at `slot`."""
        var rows = Int(self.meta.row_groups[self._row_groups[slot]].num_rows)
        if slot < len(self._ranges) and len(self._ranges[slot]):
            return self._ranges[slot].copy()
        if slot < len(self._ranges):
            # Pruning ran and left this row group with nothing.
            return List[Tuple[Int, Int]]()
        var whole = List[Tuple[Int, Int]]()
        whole.append((0, rows))
        return whole^

    def _seek(mut self) -> Bool:
        """Move to the next non-empty row range; False when there is none."""
        while self._rg_pos < len(self._row_groups):
            var rows = Int(
                self.meta.row_groups[self._row_groups[self._rg_pos]].num_rows
            )
            if rows == 0:
                # A row group with no rows still yields one empty batch, so a
                # zero-row file keeps its columns.
                if not self._did_empty:
                    return True
                self._rg_pos += 1
                self._range_pos = 0
                self._row_pos = 0
                self._did_empty = False
                continue
            var ranges = self._ranges_for(self._rg_pos)
            while self._range_pos < len(ranges):
                var span = ranges[self._range_pos]
                if self._row_pos < span[0]:
                    self._row_pos = span[0]
                if self._row_pos < span[1]:
                    return True
                self._range_pos += 1
                self._row_pos = 0
            self._rg_pos += 1
            self._range_pos = 0
            self._row_pos = 0
            self._did_empty = False
        return False

    def has_next(mut self) -> Bool:
        return self._seek()

    def read_batch(mut self) raises -> RecordBatch:
        """The next batch of at most `batch_size` rows. A batch never spans two
        row groups, and never crosses a gap left by page pruning."""
        if not self._seek():
            return RecordBatch()
        var rg = self._row_groups[self._rg_pos]
        self._load(rg)
        if Int(self.meta.row_groups[rg].num_rows) == 0:
            self._did_empty = True
            return self._assemble(0, 0)
        var ranges = self._ranges_for(self._rg_pos)
        var span = ranges[self._range_pos]
        var r0 = self._row_pos
        var r1 = r0 + self.batch_size
        if r1 > span[1]:
            r1 = span[1]
        var batch = self._assemble(r0, r1)
        self._row_pos = r1
        return batch^

    def _assemble(mut self, r0: Int, r1: Int) raises -> RecordBatch:
        var slices = _leaf_slices(
            self._needed,
            self._chunks,
            self._row_index,
            r0,
            r1,
        )
        var batch = RecordBatch()
        batch.num_rows = r1 - r0
        var roots = self._roots.copy()
        for r in roots:
            batch.roots.append(
                build_field(
                    self.schema,
                    r,
                    0,
                    self._chunks,
                    slices,
                    batch.arena,
                    self._include,
                )
            )
        return batch^

    # ── page-level pruning ─────────────────────────────────────────────────

    def page_row_ranges(
        self, rg: Int, predicates: List[Predicate]
    ) raises -> List[Tuple[Int, Int]]:
        """The rows of row group `rg` whose pages could satisfy `predicates`.

        Uses the `ColumnIndex` bounds and the `OffsetIndex` first-row indices.
        A column without a page index puts no restriction on the answer, so a
        file written without one behaves exactly as it did before.
        """
        var rows = Int(self.meta.row_groups[rg].num_rows)
        var out = List[Tuple[Int, Int]]()
        out.append((0, rows))
        for p in predicates:
            var leaf = self.schema.leaf_by_path(p.column)
            if leaf < 0:
                continue
            if self.schema.leaves[leaf].max_rep > 0:
                # A repeated column's pages do not line up with rows one to
                # one in a way a simple predicate can use.
                continue
            var oi = self.offset_index(rg, leaf)
            var ci = self.column_index(rg, leaf)
            if not oi or not ci:
                continue
            var keep = List[Tuple[Int, Int]]()
            ref locs = oi.value().page_locations
            ref idx = ci.value()
            if len(idx.null_pages) != len(locs):
                continue
            for k in range(len(locs)):
                var start = Int(locs[k].first_row_index)
                var end = rows
                if k + 1 < len(locs):
                    end = Int(locs[k + 1].first_row_index)
                if start >= end:
                    continue
                if idx.null_pages[k]:
                    continue
                if k >= len(idx.min_values) or k >= len(idx.max_values):
                    keep.append((start, end))
                    continue
                var lo = decode_statistic(
                    self.schema.leaves[leaf], Span(idx.min_values[k])
                )
                var hi = decode_statistic(
                    self.schema.leaves[leaf], Span(idx.max_values[k])
                )
                if range_can_match(p.op, lo, hi, p.value):
                    keep.append((start, end))
            out = _intersect_ranges(out, keep)
        return _merge_ranges(out^)

    def prune_pages(mut self, predicates: List[Predicate]) raises -> Int:
        """Restrict reading to the pages whose bounds could match. Returns the
        number of rows left."""
        self._ranges = List[List[Tuple[Int, Int]]]()
        var total = 0
        for i in range(len(self._row_groups)):
            var ranges = self.page_row_ranges(self._row_groups[i], predicates)
            for r in ranges:
                total += r[1] - r[0]
            self._ranges.append(ranges^)
        self.rewind()
        return total

    def read_table(mut self) raises -> Table:
        """Every selected row group, as a list of batches.

        This is the one entry point that parallelises across **row groups** as
        well as columns, and the reason is a memory contract rather than a
        performance one. `read_table` was always going to hold every row group
        of the selection at once — that is what returning a `Table` means — so
        decoding several of them at once asks for no lifetime the caller was
        not already granting. The streaming path makes the opposite promise,
        and keeps it: `read_batch` holds one row group and is untouched.

        What it does add is intermediate state. A loaded row group holds
        decompressed pages, level arrays and per-row indices *on top of* the
        Arrow batches built from them, and sequentially only one row group's
        worth is ever alive. So the window is capped at `num_workers` row
        groups: peak decode state grows by at most `num_workers - 1` row groups
        over the sequential path, never with the size of the file, and at
        `num_workers = 1` — the default — this is the same loop it always was,
        byte for byte and allocation for allocation.

        Within a window the batches are cut by the ordinary iterator walk —
        `_plan_window`, which is `read_batch` with the building taken out — and
        then built by a second fan-out over *(batch, top-level field)* pairs,
        because sequential Arrow assembly was the whole serial remainder of a
        threaded read. They are appended in file order on this thread whatever
        order the workers finished in, and each field is grafted into its batch
        at the index it would have had sequentially, so neither row order nor
        arena layout depends on the worker count.
        """
        self.rewind()
        var t = Table()
        var workers = self._resolved_workers()
        var nsel = len(self._row_groups)
        if workers <= 1 or nsel <= 1:
            while self.has_next():
                var b = self.read_batch()
                t.num_rows += b.num_rows
                t.batches.append(b^)
            return t^
        var window = workers
        if window > nsel:
            window = nsel
        var first = 0
        while first < nsel:
            var past = first + window
            if past > nsel:
                past = nsel
            self._prefetch(first, past, workers)
            var plans = self._plan_window(past)
            self._assemble_window(plans, workers, t)
            self._drop_prefetch()
            # `_plan_window` may have walked past the window looking for a row
            # to hand out; the next window starts wherever it stopped.
            first = past
            if self._rg_pos > first:
                first = self._rg_pos
        return t^


struct Table(Copyable, Defaultable, Movable):
    """Every batch of a read, with accessors that run across all of them."""

    var batches: List[RecordBatch]
    var num_rows: Int

    def __init__(out self):
        self.batches = List[RecordBatch]()
        self.num_rows = 0

    def __init__(out self, *, copy: Self):
        self.batches = copy.batches.copy()
        self.num_rows = copy.num_rows

    def __init__(out self, *, deinit move: Self):
        self.batches = move.batches^
        self.num_rows = move.num_rows

    def num_columns(self) -> Int:
        if len(self.batches) == 0:
            return 0
        return self.batches[0].num_columns()

    def name(self, i: Int) -> String:
        return self.batches[0].name(i)

    def type(self, i: Int) -> ArrowType:
        return self.batches[0].type(i)

    def column_i64(self, i: Int) raises -> Tuple[List[Int64], List[Bool]]:
        var vals = List[Int64]()
        var valid = List[Bool]()
        for b in self.batches:
            array_i64_into(b.column(i), vals, valid)
        return (vals^, valid^)

    def column_f64(self, i: Int) raises -> Tuple[List[Float64], List[Bool]]:
        var vals = List[Float64]()
        var valid = List[Bool]()
        for b in self.batches:
            array_f64_into(b.column(i), vals, valid)
        return (vals^, valid^)

    def column_bool(self, i: Int) raises -> Tuple[List[Bool], List[Bool]]:
        var vals = List[Bool]()
        var valid = List[Bool]()
        for b in self.batches:
            array_bool_into(b.column(i), vals, valid)
        return (vals^, valid^)

    def column_str(self, i: Int) raises -> Tuple[List[String], List[Bool]]:
        var vals = List[String]()
        var valid = List[Bool]()
        for b in self.batches:
            array_str_into(b.column(i), vals, valid)
        return (vals^, valid^)
