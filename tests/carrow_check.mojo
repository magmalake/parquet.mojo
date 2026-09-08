"""What this repository's Arrow C Data Interface round-trip tests compare with.

The Arrow layer itself is arrow-mlake.mojo, and so are its own unit tests and
the pyarrow gate. What stays here is the *integration* check: every column of
every Parquet fixture, exported and imported back, which is coverage of the
decoder as much as of the interface. These are the helpers it needs.

`assert_same_array` is the round-trip comparator: two `ArrayData` trees, in two
arenas that need not be numbered alike, walked in lockstep and compared on
everything that survives a trip through C — type, name, nullability, length,
null count, per-element validity, offsets, and the *meaningful extent* of the
values buffer rather than its capacity, because an imported buffer is exactly
as long as the elements need and the reader's need not be.

`assert_preorder` is the half a lockstep walk cannot see. `tests/fingerprint`
records the trap: a check that starts at the roots and follows child indices
folds a consistently permuted arena to the same answer, because it never looks
at the numbers themselves. So the import's arena layout is asserted directly —
the nodes of an imported tree are its pre-order DFS, `root`, `root + 1`, … with
nothing else in between — and `permuted_arena` is the negative control that
renumbers an arena without touching a value, so a test can show that
`assert_same_array` is blind to it and `assert_preorder` is not.

The rest is plumbing over the two C structs: `exported_pair` hands the raw
addresses over unowned, and `word` / `set_word` / `buffer_address` /
`child_array` are how a test reaches in and malforms one.
"""

# Deliberately imported through `parquet` rather than `arrow_mlake`: these are
# the re-export paths consumers write, so building this file is what keeps the
# `parquet.arrow` / `parquet.carrow` shims from rotting.
from parquet import (
    AT_BINARY,
    AT_BOOL,
    AT_LARGE_BINARY,
    AT_LARGE_UTF8,
    AT_UTF8,
    ArrayArena,
    ArrayData,
    export_c,
)
from parquet.arrow import bit_get


# ── the round-trip comparator ──────────────────────────────────────────────


def _fail(label: StringSlice, what: StringSlice) raises:
    raise Error(String("carrow round trip at ", label, ": ", what))


def _values_extent(a: ArrayData) raises -> Int:
    """How many bytes of `a.values` actually carry elements.

    An imported array's values buffer is sized to its window; the reader's can
    be longer than the elements it holds, because it was filled by a decoder
    that appended. Comparing the two buffers whole would report a difference
    that is not one, so both sides are cut to the extent the type defines.
    """
    var i = a.type.id
    if i == AT_BOOL:
        return (a.length + 7) // 8
    if i == AT_UTF8 or i == AT_BINARY:
        return Int(a.offsets[a.length]) if len(a.offsets) > a.length else 0
    if i == AT_LARGE_UTF8 or i == AT_LARGE_BINARY:
        return (
            Int(a.large_offsets[a.length]) if len(a.large_offsets)
            > a.length else 0
        )
    return a.type.fixed_width() * a.length


def assert_same_array(
    left: ArrayArena,
    lroot: Int,
    right: ArrayArena,
    rroot: Int,
    label: StringSlice,
) raises:
    """Two array trees, compared on everything the C interface carries.

    `field_id` is deliberately not compared: the export writes no
    `PARQUET:field_id` metadata, so a round trip through our own exporter
    drops it. The importer reads the key when a foreign producer writes it,
    which is asserted separately.
    """
    var stack_l: List[Int] = [lroot]
    var stack_r: List[Int] = [rroot]
    var visited = 0
    while len(stack_l):
        var li = stack_l.pop()
        var ri = stack_r.pop()
        visited += 1
        ref a = left.nodes[li]
        ref b = right.nodes[ri]
        var at = String(label, " node ", visited)
        if a.type.format() != b.type.format():
            _fail(
                at,
                String("type ", a.type.format(), " became ", b.type.format()),
            )
        if String(a.type) != String(b.type):
            _fail(at, String("type ", a.type, " became ", b.type))
        if a.name != b.name:
            _fail(at, String("name ", a.name, " became ", b.name))
        if a.nullable != b.nullable:
            _fail(at, "nullability changed")
        if a.length != b.length:
            _fail(at, String("length ", a.length, " became ", b.length))
        if a.null_count != b.null_count:
            _fail(
                at,
                String("null_count ", a.null_count, " became ", b.null_count),
            )
        for i in range(a.length):
            if bit_get(Span(a.validity), i) != bit_get(Span(b.validity), i):
                _fail(at, String("validity differs at element ", i))
        if len(a.offsets) != len(b.offsets) or len(a.large_offsets) != len(
            b.large_offsets
        ):
            _fail(at, "offset buffer lengths differ")
        for i in range(len(a.offsets)):
            if a.offsets[i] != b.offsets[i]:
                _fail(at, String("offset ", i, " differs"))
        for i in range(len(a.large_offsets)):
            if a.large_offsets[i] != b.large_offsets[i]:
                _fail(at, String("large offset ", i, " differs"))
        var extent = _values_extent(a)
        if extent != _values_extent(b):
            _fail(at, "values extents differ")
        if len(a.values) < extent or len(b.values) < extent:
            _fail(at, "a values buffer is shorter than its own elements")
        for i in range(extent):
            if a.values[i] != b.values[i]:
                _fail(at, String("values byte ", i, " differs"))
        if len(a.children) != len(b.children):
            _fail(at, "child counts differ")
        for k in range(len(a.children) - 1, -1, -1):
            stack_l.append(a.children[k])
            stack_r.append(b.children[k])


def preorder_size(arena: ArrayArena, root: Int) raises -> Int:
    """How many nodes the tree rooted at `root` holds."""
    var n = 0
    var stack: List[Int] = [root]
    while len(stack):
        var node = stack.pop()
        n += 1
        ref kids = arena.nodes[node].children
        for k in range(len(kids)):
            stack.append(kids[k])
    return n


def assert_preorder(arena: ArrayArena, root: Int, label: StringSlice) raises:
    """The tree at `root` occupies `root, root + 1, …` in DFS pre-order.

    This is the property `import_c` promises and the one a walk from the root
    cannot check for itself, so it is checked against the arena's own indices.
    """
    var expect = root
    var stack: List[Int] = [root]
    while len(stack):
        var node = stack.pop()
        if node != expect:
            raise Error(
                String(
                    "carrow arena layout at ",
                    label,
                    ": expected node ",
                    expect,
                    " in pre-order, found ",
                    node,
                )
            )
        expect += 1
        ref kids = arena.nodes[node].children
        for k in range(len(kids) - 1, -1, -1):
            stack.append(kids[k])


def permuted_arena(
    arena: ArrayArena, root: Int
) raises -> Tuple[ArrayArena, Int]:
    """`arena` renumbered back to front, with not one value changed.

    The negative control for `assert_preorder`, and the same damage
    `fingerprint.permuted_arenas` models: every array holds the bytes it held
    and every tree is the tree that went in, but the arena's own numbering is
    reversed.
    """
    var n = len(arena.nodes)
    var out = ArrayArena()
    for i in range(n):
        var a = arena.nodes[n - 1 - i].copy()
        for c in range(len(a.children)):
            a.children[c] = n - 1 - a.children[c]
        _ = out.add(a^)
    return (out^, n - 1 - root)


def _words(addr: Int) -> Pointer[Int64, MutUntrackedOrigin]:
    return Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)


def exported_pair(arena: ArrayArena, root: Int) raises -> Tuple[Int, Int]:
    """Export one array and hand the two raw addresses over, unowned."""
    var e = export_c(arena, root)
    var raw = e.into_raw()
    return (raw[0], raw[1])


def buffer_address(array: Int, i: Int) -> Int:
    """Buffer `i` of an exported `ArrowArray`, so a test can corrupt it."""
    var bufs = Int(_words(array)[unsafe_offset=5])
    return Int(_words(bufs)[unsafe_offset=i])


def set_word(addr: Int, i: Int, v: Int64):
    """Overwrite one word of a C struct, so a test can malform it."""
    _words(addr)[unsafe_offset=i] = v


def word(addr: Int, i: Int) -> Int64:
    return _words(addr)[unsafe_offset=i]


def child_array(array: Int, k: Int) -> Int:
    return Int(_words(Int(_words(array)[unsafe_offset=6]))[unsafe_offset=k])
