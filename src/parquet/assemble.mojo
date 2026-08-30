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
    AT_LIST,
    AT_MAP,
    AT_STRUCT,
    ArrayArena,
    ArrayData,
    bit_set,
)
from parquet.convert import append_null, append_value
from parquet.page import ColumnData
from parquet.schema import ParquetSchema


@fieldwise_init
struct LeafSlice(Copyable, Movable, Defaultable):
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
    var f = fi
    while s.fields[f].leaf < 0:
        if len(s.fields[f].children) == 0:
            raise Error(
                String(
                    "parquet.assemble: field '",
                    s.fields[f].name,
                    "' has no leaf under it",
                )
            )
        f = s.fields[f].children[0]
    return s.fields[f].leaf


def build_field(
    s: ParquetSchema,
    fi: Int,
    require_def: Int,
    chunks: List[ColumnData],
    slices: List[LeafSlice],
    mut arena: ArrayArena,
) raises -> Int:
    """Build the Arrow array for field `fi`; return its index in `arena`."""
    var lf = first_leaf(s, fi)
    var kind = s.fields[fi].type.id

    if s.fields[fi].leaf >= 0:
        return _build_leaf(s, fi, require_def, chunks, slices, arena)
    if kind == AT_LIST or kind == AT_MAP:
        return _build_list(s, fi, require_def, chunks, slices, arena, lf)
    if kind == AT_STRUCT:
        return _build_struct(s, fi, require_def, chunks, slices, arena, lf)
    raise Error(
        String(
            "parquet.assemble: field '",
            s.fields[fi].name,
            "' has no leaf and is not a list, map or struct",
        )
    )


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
    var vi = sl.v0
    for i in range(sl.s0, sl.s1):
        var d = Int(cd.defs[i]) if len(cd.defs) else 0
        var present = d == max_def
        if d >= require_def:
            if present:
                append_value(out, leaf, cd.values, vi)
            else:
                append_null(out, leaf)
        if present:
            vi += 1
    if out.null_count == 0:
        out.validity.clear()
    return arena.add(out^)


def _build_struct(
    s: ParquetSchema,
    fi: Int,
    require_def: Int,
    chunks: List[ColumnData],
    slices: List[LeafSlice],
    mut arena: ArrayArena,
    lf: Int,
) raises -> Int:
    var kids = List[Int]()
    var child_fields = s.fields[fi].children.copy()
    for c in child_fields:
        kids.append(build_field(s, c, require_def, chunks, slices, arena))
    ref cd = chunks[lf]
    ref sl = slices[lf]
    var def_level = s.fields[fi].def_level
    var rep_level = s.fields[fi].rep_level
    var out = _new_array(s, fi)
    for i in range(sl.s0, sl.s1):
        var d = Int(cd.defs[i]) if len(cd.defs) else 0
        var r = Int(cd.reps[i]) if len(cd.reps) else 0
        if d >= require_def and r <= rep_level:
            var valid = d >= def_level
            bit_set(out.validity, out.length, valid)
            if not valid:
                out.null_count += 1
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
) raises -> Int:
    var elem_def = s.fields[fi].elem_def_level
    var child = build_field(
        s, s.fields[fi].children[0], elem_def, chunks, slices, arena
    )
    ref cd = chunks[lf]
    ref sl = slices[lf]
    var rep_level = s.fields[fi].rep_level
    var elem_rep = s.fields[fi].elem_rep_level
    var def_level = s.fields[fi].def_level
    var out = _new_array(s, fi)
    out.offsets.append(0)
    var elems = 0
    for i in range(sl.s0, sl.s1):
        var d = Int(cd.defs[i]) if len(cd.defs) else 0
        var r = Int(cd.reps[i]) if len(cd.reps) else 0
        if d < require_def:
            continue
        if r <= rep_level:
            if out.length > 0:
                out.offsets.append(Int32(elems))
            var valid = d >= def_level
            bit_set(out.validity, out.length, valid)
            if not valid:
                out.null_count += 1
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
