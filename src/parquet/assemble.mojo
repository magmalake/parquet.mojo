"""Dremel record assembly: definition and repetition levels back into Arrow.

Every leaf of a Parquet file carries, for each of its value slots, the
definition level (how deep the path to the value is actually present) and the
repetition level (at which level of nesting the value repeats). Those two
numbers are enough to rebuild the whole nested structure, and that is what
`build_field` does — one Arrow array per Arrow field, recursively.

The key idea is a *presence threshold*. A field is asked to build only the
slots whose definition level is at least `require_def`, which its parent sets:

* the root asks for `require_def = 0`, so every slot is a row;
* a `LIST` or `MAP` asks its child for `require_def = elem_def_level`, the
  definition level at which the repeated group actually has an entry — so the
  child array comes back already containing exactly the list's elements, with
  the empty and null containers dropped;
* a `STRUCT` passes its own threshold straight through, because a struct does
  not change granularity.

Levels come from the field's first descendant leaf. Every leaf under a field
agrees about the levels at and above that field, so any one of them will do —
which is what makes it possible to build a struct's children independently and
still have them line up.
"""

from parquet.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_DECIMAL128,
    AT_LIST,
    AT_MAP,
    AT_STRUCT,
    AT_TIMESTAMP,
    AT_UTF8,
    ArrayArena,
    ArrayData,
    bit_fill_valid,
    bit_set,
)
from parquet.convert import append_null, append_value
from parquet.encoding import PK_BOOL, PK_FIXED, PK_VAR, PhysBuffer
from parquet.page import ColumnData
from parquet.schema import LeafColumn, ParquetSchema
from thrift import Type


@fieldwise_init
struct LeafSlice(Copyable, Defaultable, Movable):
    """The slot range of one leaf that a batch covers, and where its values
    start in the chunk's value buffer."""

    var s0: Int
    var s1: Int
    var v0: Int

    def __init__(out self):
        self.s0 = 0
        self.s1 = 0
        self.v0 = 0


def first_leaf(s: ParquetSchema, fi: Int) raises -> Int:
    """The first descendant leaf of Arrow field `fi` — its level source."""
    return first_included_leaf(s, fi, List[Bool]())


def first_included_leaf(
    s: ParquetSchema, fi: Int, include: List[Bool]
) raises -> Int:
    """`first_leaf`, restricted to the fields `include` selects.

    A pruned read skips whole sub-trees, so the level source of a struct can no
    longer be its first child: it has to be the first child actually being
    built. Every leaf under a field agrees about the levels at and above that
    field, so any included one will do.
    """
    var lf = _first_included_leaf(s, fi, include)
    if lf < 0:
        raise Error(
            String(
                "parquet.assemble: field '",
                s.fields[fi].name,
                "' has no leaf under it",
            )
        )
    return lf


def _first_included_leaf(s: ParquetSchema, fi: Int, include: List[Bool]) -> Int:
    if len(include) > 0 and not include[fi]:
        return -1
    if s.fields[fi].leaf >= 0:
        return s.fields[fi].leaf
    ref kids = s.fields[fi].children
    if len(kids) == 0:
        return -1
    for k in range(len(kids)):
        var got = _first_included_leaf(s, kids[k], include)
        if got >= 0:
            return got
    return -1


def build_field(
    s: ParquetSchema,
    fi: Int,
    require_def: Int,
    chunks: List[ColumnData],
    slices: List[LeafSlice],
    mut arena: ArrayArena,
    include: List[Bool],
) raises -> Int:
    """Build the Arrow array for field `fi`; return its index in `arena`.

    `include`, when it is not empty, is one flag per entry of `s.fields`: a
    struct builds only the children it selects, so a projection that wants
    `a.b` and nothing else decodes exactly the leaves under `a.b`. A list and
    a map always build their element (and a map its key), because their
    offsets mean nothing without one.
    """
    var lf = first_included_leaf(s, fi, include)
    var kind = s.fields[fi].type.id

    if s.fields[fi].leaf >= 0:
        return _build_leaf(s, fi, require_def, chunks, slices, arena)
    if kind == AT_LIST or kind == AT_MAP:
        return _build_list(
            s, fi, require_def, chunks, slices, arena, lf, include
        )
    if kind == AT_STRUCT:
        return _build_struct(
            s, fi, require_def, chunks, slices, arena, lf, include
        )
    raise Error(
        String(
            "parquet.assemble: field '",
            s.fields[fi].name,
            "' has no leaf and is not a list, map or struct",
        )
    )


def _mark(mut a: ArrayData, i: Int, valid: Bool):
    """Record the validity of slot `i`, building the bitmap only if needed."""
    if valid:
        if len(a.validity):
            bit_set(a.validity, i, True)
        return
    if len(a.validity) == 0 and i > 0:
        bit_fill_valid(a.validity, i)
    bit_set(a.validity, i, False)
    a.null_count += 1


def _new_array(s: ParquetSchema, fi: Int) -> ArrayData:
    var a = ArrayData(s.fields[fi].type.copy(), s.fields[fi].name.copy())
    a.nullable = s.fields[fi].nullable
    a.field_id = s.fields[fi].field_id
    return a^


def _build_leaf(
    s: ParquetSchema,
    fi: Int,
    require_def: Int,
    chunks: List[ColumnData],
    slices: List[LeafSlice],
    mut arena: ArrayArena,
) raises -> Int:
    var lf = s.fields[fi].leaf
    ref cd = chunks[lf]
    ref sl = slices[lf]
    ref leaf = s.leaves[lf]
    var max_def = leaf.max_def
    var out = _new_array(s, fi)
    if require_def == 0 or (cd.all_present and max_def >= require_def):
        # Every slot in the range becomes a row, so the array can be built one
        # buffer at a time rather than one row at a time.
        if _fill_leaf(out, leaf, cd, sl.s0, sl.v0, sl.s1 - sl.s0, max_def):
            return arena.add(out^)
    var vi = sl.v0
    for i in range(sl.s0, sl.s1):
        var d = cd.def_at(i, max_def)
        var present = d == max_def
        if d >= require_def:
            _mark(out, out.length, present)
            if present:
                append_value(out, leaf, cd.values, vi)
            else:
                append_null(out, leaf)
        if present:
            vi += 1
    if out.null_count == 0:
        out.validity.clear()
    return arena.add(out^)


def _fill_leaf(
    mut out: ArrayData,
    leaf: LeafColumn,
    cd: ColumnData,
    s0: Int,
    v0: Int,
    n: Int,
    max_def: Int,
) raises -> Bool:
    """Build a leaf array for `n` consecutive slots in one pass per buffer.

    The caller has established that every one of those slots becomes a row, so
    the only question left per slot is whether it holds a value. That makes
    the validity bitmap one pass over the definition levels and the values one
    pass over the value buffer, instead of a call to `append_value` per row.

    Returns `False` — leaving `out` untouched — when the Arrow bytes are not
    the Parquet bytes: a decimal, an `INT96` timestamp, or a narrow integer
    carried in a wider physical type. Those still go value by value.
    """
    if n < 0:
        return False
    var id = out.type.id
    if id == AT_DECIMAL128:
        return False
    if id == AT_TIMESTAMP and leaf.physical == Type.INT96.value:
        return False
    ref vals = cd.values
    var w = out.type.fixed_width()
    var var_len = id == AT_UTF8 or id == AT_BINARY
    if var_len:
        if vals.kind != PK_VAR:
            return False
    elif id == AT_BOOL:
        if vals.kind != PK_BOOL:
            return False
    elif w == 0 or vals.kind != PK_FIXED or vals.width != w:
        return False

    # ── validity ──────────────────────────────────────────────────────────
    var dense = cd.all_present
    var present = n
    if not dense:
        if s0 + n > len(cd.defs):
            return False
        var defs = cd.defs.unsafe_ptr()
        out.validity.resize((n + 7) // 8, 0)
        var vb = out.validity.unsafe_ptr()
        var byte = UInt8(0)
        present = 0
        for k in range(n):
            if Int(defs.unsafe_load(s0 + k)) == max_def:
                byte |= UInt8(1) << UInt8(k & 7)
                present += 1
            if (k & 7) == 7:
                vb.unsafe_store(k >> 3, byte)
                byte = 0
        if (n & 7) != 0:
            vb.unsafe_store(n >> 3, byte)
    out.length = n
    out.null_count = n - present
    if out.null_count == 0:
        out.validity.clear()
        dense = True

    # ── values ────────────────────────────────────────────────────────────
    if var_len:
        if v0 + present + 1 > len(vals.offsets):
            raise Error(_short_values(leaf))
        var voff = vals.offsets.unsafe_ptr()
        var base = Int(voff.unsafe_load(v0))
        var last = Int(voff.unsafe_load(v0 + present))
        out.values.extend(Span(vals.bytes)[base:last])
        out.offsets.resize(n + 1, 0)
        var ooff = out.offsets.unsafe_ptr()
        if dense:
            for k in range(n + 1):
                ooff.unsafe_store(k, voff.unsafe_load(v0 + k) - Int32(base))
        else:
            var defs = cd.defs.unsafe_ptr()
            var vi = v0
            var run = 0
            for k in range(n):
                ooff.unsafe_store(k, Int32(run))
                if Int(defs.unsafe_load(s0 + k)) == max_def:
                    run += Int(voff.unsafe_load(vi + 1)) - Int(
                        voff.unsafe_load(vi)
                    )
                    vi += 1
            ooff.unsafe_store(n, Int32(run))
        return True

    if id == AT_BOOL:
        out.values.resize((n + 7) // 8, 0)
        var vp = out.values.unsafe_ptr()
        var byte = UInt8(0)
        var vi = v0
        if dense:
            for k in range(n):
                if vals.bool_at(v0 + k):
                    byte |= UInt8(1) << UInt8(k & 7)
                if (k & 7) == 7:
                    vp.unsafe_store(k >> 3, byte)
                    byte = 0
        else:
            var defs = cd.defs.unsafe_ptr()
            for k in range(n):
                if Int(defs.unsafe_load(s0 + k)) == max_def:
                    if vals.bool_at(vi):
                        byte |= UInt8(1) << UInt8(k & 7)
                    vi += 1
                if (k & 7) == 7:
                    vp.unsafe_store(k >> 3, byte)
                    byte = 0
        if (n & 7) != 0:
            vp.unsafe_store(n >> 3, byte)
        return True

    if (v0 + present) * w > len(vals.bytes):
        raise Error(_short_values(leaf))
    if dense:
        out.values.extend(Span(vals.bytes)[v0 * w : (v0 + n) * w])
        return True
    out.values.resize(n * w, 0)
    var dst = out.values.unsafe_ptr()
    var src = vals.bytes.unsafe_ptr()
    var defs = cd.defs.unsafe_ptr()
    var vi = v0
    if w == 8:
        var d8 = dst.unsafe_bitcast[UInt64]()
        var s8 = src.unsafe_bitcast[UInt64]()
        for k in range(n):
            if Int(defs.unsafe_load(s0 + k)) == max_def:
                d8.unsafe_store(k, s8.unsafe_load[alignment=1](vi))
                vi += 1
    elif w == 4:
        var d4 = dst.unsafe_bitcast[UInt32]()
        var s4 = src.unsafe_bitcast[UInt32]()
        for k in range(n):
            if Int(defs.unsafe_load(s0 + k)) == max_def:
                d4.unsafe_store(k, s4.unsafe_load[alignment=1](vi))
                vi += 1
    else:
        for k in range(n):
            if Int(defs.unsafe_load(s0 + k)) == max_def:
                for b in range(w):
                    dst.unsafe_store(k * w + b, src.unsafe_load(vi * w + b))
                vi += 1
    return True


def _short_values(leaf: LeafColumn) -> String:
    return String(
        "parquet.assemble: column '",
        leaf.dotted(),
        "' has fewer values than its definition levels call for",
    )


def _build_struct(
    s: ParquetSchema,
    fi: Int,
    require_def: Int,
    chunks: List[ColumnData],
    slices: List[LeafSlice],
    mut arena: ArrayArena,
    lf: Int,
    include: List[Bool],
) raises -> Int:
    var kids = List[Int]()
    var child_fields = s.fields[fi].children.copy()
    for c in child_fields:
        if len(include) > 0 and not include[c]:
            continue
        kids.append(
            build_field(s, c, require_def, chunks, slices, arena, include)
        )
    ref cd = chunks[lf]
    ref sl = slices[lf]
    var def_level = s.fields[fi].def_level
    var rep_level = s.fields[fi].rep_level
    var out = _new_array(s, fi)
    var leaf_max_def = s.leaves[lf].max_def
    for i in range(sl.s0, sl.s1):
        var d = cd.def_at(i, leaf_max_def)
        var r = cd.rep_at(i)
        if d >= require_def and r <= rep_level:
            _mark(out, out.length, d >= def_level)
            out.length += 1
    if out.null_count == 0:
        out.validity.clear()
    for k in kids:
        if arena.nodes[k].length != out.length:
            raise Error(
                String(
                    "parquet.assemble: struct '",
                    s.fields[fi].name,
                    "' has ",
                    out.length,
                    " row(s) but child '",
                    arena.nodes[k].name,
                    "' has ",
                    arena.nodes[k].length,
                )
            )
    out.children = kids^
    return arena.add(out^)


def _build_list(
    s: ParquetSchema,
    fi: Int,
    require_def: Int,
    chunks: List[ColumnData],
    slices: List[LeafSlice],
    mut arena: ArrayArena,
    lf: Int,
    include: List[Bool],
) raises -> Int:
    var elem_def = s.fields[fi].elem_def_level
    var child = build_field(
        s, s.fields[fi].children[0], elem_def, chunks, slices, arena, include
    )
    ref cd = chunks[lf]
    ref sl = slices[lf]
    var rep_level = s.fields[fi].rep_level
    var elem_rep = s.fields[fi].elem_rep_level
    var def_level = s.fields[fi].def_level
    var out = _new_array(s, fi)
    out.offsets.append(0)
    var elems = 0
    var leaf_max_def = s.leaves[lf].max_def
    for i in range(sl.s0, sl.s1):
        var d = cd.def_at(i, leaf_max_def)
        var r = cd.rep_at(i)
        if d < require_def:
            continue
        if r <= rep_level:
            if out.length > 0:
                out.offsets.append(Int32(elems))
            _mark(out, out.length, d >= def_level)
            out.length += 1
        elif out.length == 0:
            raise Error(
                String(
                    "parquet.assemble: list '",
                    s.fields[fi].name,
                    "' starts with repetition level ",
                    r,
                    ", which continues a list that never began",
                )
            )
        if d >= elem_def and r <= elem_rep:
            elems += 1
    if out.length > 0:
        out.offsets.append(Int32(elems))
    if out.null_count == 0:
        out.validity.clear()
    if arena.nodes[child].length != elems:
        raise Error(
            String(
                "parquet.assemble: list '",
                s.fields[fi].name,
                "' counted ",
                elems,
                " element(s) but its child array has ",
                arena.nodes[child].length,
            )
        )
    out.children.append(child)
    return arena.add(out^)
