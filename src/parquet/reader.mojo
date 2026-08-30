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
from thrift import (
    ColumnIndex,
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


def range_can_match(op: Int, lo: ScalarValue, hi: ScalarValue, v: ScalarValue) raises -> Bool:
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


struct RecordBatch(Copyable, Movable, Defaultable):
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

    def column(ref self, i: Int) -> ref [self.arena.nodes[0]] ArrayData:
        return self.arena.nodes[self.roots[i]]

    def child(ref self, node: Int, k: Int) -> ref [self.arena.nodes[0]] ArrayData:
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
            String("parquet: column of type ", String(a.type), " is not an integer")
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
            var bits = UInt16(a.values[i * 2]) | (UInt16(a.values[i * 2 + 1]) << 8)
            vals.append(Float64(bitcast[DType.float16](bits)))
    else:
        raise Error(
            String("parquet: column of type ", String(a.type), " is not floating point")
        )
    _append_validity(a, valid)


def array_bool_into(
    a: ArrayData, mut vals: List[Bool], mut valid: List[Bool]
) raises:
    if a.type.id != AT_BOOL:
        raise Error(
            String("parquet: column of type ", String(a.type), " is not boolean")
        )
    for i in range(a.length):
        vals.append(bit_get(Span(a.values), i))
    _append_validity(a, valid)


def array_str_into(
    a: ArrayData, mut vals: List[String], mut valid: List[Bool]
) raises:
    if a.type.id != AT_UTF8 and a.type.id != AT_BINARY:
        raise Error(
            String("parquet: column of type ", String(a.type), " is not a byte array")
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


struct ParquetReader[Codecs: CodecSet = DefaultCodecs](Movable):
    """A Parquet file, projected and batched."""

    var data: List[UInt8]
    var meta: FileMetaData
    var schema: ParquetSchema
    var batch_size: Int
    var verify_crc: Bool
    var _row_groups: List[Int]
    var _roots: List[Int]
    """Selected top-level Arrow field indices."""
    var _needed: List[Bool]
    """Per leaf: is it under a selected root?"""
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
    var _row_slot: List[List[Int]]
    var _row_value: List[List[Int]]

    def __init__(out self, var data: List[UInt8]) raises:
        self.data = data^
        self.meta = read_footer(Span(self.data))
        self.schema = build_schema(self.meta.schema)
        self.batch_size = 65536
        self.verify_crc = True
        self._row_groups = List[Int]()
        for i in range(len(self.meta.row_groups)):
            self._row_groups.append(i)
        self._roots = self.schema.roots.copy()
        self._needed = List[Bool](length=len(self.schema.leaves), fill=True)
        self._rg_pos = 0
        self._range_pos = 0
        self._row_pos = 0
        self._did_empty = False
        self._ranges = List[List[Tuple[Int, Int]]]()
        self._loaded_rg = -1
        self._chunks = List[ColumnData]()
        self._row_slot = List[List[Int]]()
        self._row_value = List[List[Int]]()

    def __init__(out self, *, deinit move: Self):
        self.data = move.data^
        self.meta = move.meta^
        self.schema = move.schema^
        self.batch_size = move.batch_size
        self.verify_crc = move.verify_crc
        self._row_groups = move._row_groups^
        self._roots = move._roots^
        self._needed = move._needed^
        self._rg_pos = move._rg_pos
        self._range_pos = move._range_pos
        self._row_pos = move._row_pos
        self._did_empty = move._did_empty
        self._ranges = move._ranges^
        self._loaded_rg = move._loaded_rg
        self._chunks = move._chunks^
        self._row_slot = move._row_slot^
        self._row_value = move._row_value^

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
        var p = TCompactProtocolReader(Span(self.data)[off : off + n])
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
        var p = TCompactProtocolReader(Span(self.data)[off : off + n])
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
        for i in range(len(self._needed)):
            self._needed[i] = False
        var rs = self._roots.copy()
        for r in rs:
            self._mark(r)
        self._loaded_rg = -1

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

    def _load(mut self, rg: Int) raises:
        if self._loaded_rg == rg:
            return
        var nleaves = len(self.schema.leaves)
        self._chunks = List[ColumnData]()
        self._row_slot = List[List[Int]]()
        self._row_value = List[List[Int]]()
        var rows = Int(self.meta.row_groups[rg].num_rows)
        if len(self.meta.row_groups[rg].columns) != nleaves:
            raise Error(
                String(
                    "parquet: row group ",
                    rg,
                    " has ",
                    len(self.meta.row_groups[rg].columns),
                    " column chunk(s) for a schema with ",
                    nleaves,
                    " leaves",
                )
            )
        for i in range(nleaves):
            var cd = ColumnData()
            var slot = List[Int]()
            var value = List[Int]()
            if self._needed[i]:
                ref chunk = self.meta.row_groups[rg].columns[i]
                if not chunk.meta_data:
                    raise Error(
                        String(
                            "parquet: column chunk ",
                            i,
                            " of row group ",
                            rg,
                            " has no metadata (an encrypted column?)",
                        )
                    )
                cd = read_column_chunk[Self.Codecs](
                    Span(self.data),
                    chunk.meta_data.value(),
                    self.schema.leaves[i],
                    self.verify_crc,
                )
                var max_def = self.schema.leaves[i].max_def
                var max_rep = self.schema.leaves[i].max_rep
                slot.reserve(rows + 1)
                value.reserve(rows + 1)
                var nvals = 0
                if max_rep == 0:
                    if cd.num_slots != rows:
                        raise Error(
                            String(
                                "parquet: column '",
                                self.schema.leaves[i].dotted(),
                                "' has ",
                                cd.num_slots,
                                " value(s) in a row group of ",
                                rows,
                                " rows",
                            )
                        )
                    for k in range(rows):
                        slot.append(k)
                        value.append(nvals)
                        if max_def == 0 or Int(cd.defs[k]) == max_def:
                            nvals += 1
                    slot.append(rows)
                    value.append(nvals)
                else:
                    for k in range(cd.num_slots):
                        if Int(cd.reps[k]) == 0:
                            slot.append(k)
                            value.append(nvals)
                        if Int(cd.defs[k]) == max_def:
                            nvals += 1
                    slot.append(cd.num_slots)
                    value.append(nvals)
                    if len(slot) != rows + 1:
                        raise Error(
                            String(
                                "parquet: column '",
                                self.schema.leaves[i].dotted(),
                                "' assembles ",
                                len(slot) - 1,
                                " record(s) in a row group of ",
                                rows,
                                " rows",
                            )
                        )
            self._chunks.append(cd^)
            self._row_slot.append(slot^)
            self._row_value.append(value^)
        self._loaded_rg = rg

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
            var rows = Int(self.meta.row_groups[self._row_groups[self._rg_pos]].num_rows)
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
        var slices = List[LeafSlice]()
        for i in range(len(self.schema.leaves)):
            if self._needed[i] and len(self._row_slot[i]) > r1:
                slices.append(
                    LeafSlice(
                        self._row_slot[i][r0],
                        self._row_slot[i][r1],
                        self._row_value[i][r0],
                    )
                )
            else:
                slices.append(LeafSlice())
        var batch = RecordBatch()
        batch.num_rows = r1 - r0
        var roots = self._roots.copy()
        for r in roots:
            batch.roots.append(
                build_field(self.schema, r, 0, self._chunks, slices, batch.arena)
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
        """Every selected row group, as a list of batches."""
        self.rewind()
        var t = Table()
        while self.has_next():
            var b = self.read_batch()
            t.num_rows += b.num_rows
            t.batches.append(b^)
        return t^


struct Table(Copyable, Movable, Defaultable):
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
