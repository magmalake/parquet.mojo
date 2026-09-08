"""What the Arrow C Data Interface import tests compare with, and produce from.

Three things live here.

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

The rest is a **producer**: `make_struct_arena` builds a small struct array by
hand, and `StreamSource` wraps a list of exported ones in a real C
`ArrowArrayStream` — `get_schema`, `get_next` until a released array, and a
`get_last_error` that can be made to fire. Testing a consumer against our own
exporter would only prove the two agree; a hand-written producer that moves
structs out and hands over ownership the way the specification says is what
exercises the ownership rules. pyarrow, over in `tools/produce_c_data.py`, is
the producer we did not write at all.
"""

from parquet import (
    AT_BOOL,
    AT_BINARY,
    AT_INT64,
    AT_LARGE_BINARY,
    AT_LARGE_UTF8,
    AT_STRUCT,
    AT_UTF8,
    ArrayArena,
    ArrayData,
    ArrowType,
    CArrowArray,
    CArrowArrayStream,
    CArrowSchema,
    export_c,
)
from parquet.arrow import bit_get, bit_set
from parquet.carrow_import import (
    StreamGetLastErrorFn,
    StreamGetNextFn,
    StreamGetSchemaFn,
    StreamReleaseFn,
    release_c_array,
    release_c_schema,
)
from std.memory.alloc import unsafe_alloc


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


# ── a hand-built struct array, for the stream ──────────────────────────────


def _put_i64(mut buf: List[UInt8], v: Int64):
    var u = UInt64(v)
    for k in range(8):
        buf.append(UInt8((u >> UInt64(8 * k)) & 0xFF))


def make_struct_arena(rows: Int, base: Int64) raises -> Tuple[ArrayArena, Int]:
    """A `struct<n: int64>` of `rows` rows holding `base + i`, every third null.

    Built by hand rather than read out of a fixture so that a stream's batches
    can differ from one another and the consumer can be caught handing them
    back in the wrong order.
    """
    var arena = ArrayArena()
    var child = ArrayData(ArrowType(AT_INT64), String("n"))
    child.length = rows
    for i in range(rows):
        var null = i % 3 == 2
        _put_i64(child.values, 0 if null else base + Int64(i))
        bit_set(child.validity, i, not null)
        if null:
            child.null_count += 1
    var ci = arena.add(child^)
    var st = ArrayData(ArrowType(AT_STRUCT), String("row"))
    st.length = rows
    st.children = [ci]
    var root = arena.add(st^)
    return (arena^, root)


# ── a C ArrowArrayStream producer ──────────────────────────────────────────
#
# `private_data` is one `Int64` block: cursor, count, the schema address, the
# call index at which `get_next` should fail (-1 for never), the address of
# the error message, then one address per pre-exported `ArrowArray`, then the
# message bytes. Handing a struct over is a *move* — the 72 or 80 bytes are
# copied into the consumer's storage and our own copy's `release` is zeroed —
# which is what makes the consumer, and only the consumer, responsible for
# freeing it.

comptime _W_CURSOR = 0
comptime _W_COUNT = 1
comptime _W_SCHEMA = 2
comptime _W_FAIL_AT = 3
comptime _W_MESSAGE = 4
comptime _W_ARRAYS = 5


def _words(addr: Int) -> Pointer[Int64, MutUntrackedOrigin]:
    return Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)


def _move_struct(src: Int, dst: Int, size: Int, release_word: Int):
    """Copy `size` bytes and mark the source released — the interface's move."""
    var s = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src)
    var d = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst)
    for i in range(size):
        d[unsafe_offset=i] = s[unsafe_offset=i]
    _words(src)[unsafe_offset=release_word] = 0


def _src_get_schema(
    s: Pointer[CArrowArrayStream, MutUntrackedOrigin],
    dest: Pointer[CArrowSchema, MutUntrackedOrigin],
) abi("C") -> Int32:
    var blk = s[].private_data
    var schema = Int(_words(blk)[unsafe_offset=_W_SCHEMA])
    if schema == 0:
        return 22
    _move_struct(schema, Int(dest), 72, 7)
    _words(blk)[unsafe_offset=_W_SCHEMA] = 0
    return 0


def _src_get_next(
    s: Pointer[CArrowArrayStream, MutUntrackedOrigin],
    dest: Pointer[CArrowArray, MutUntrackedOrigin],
) abi("C") -> Int32:
    var blk = s[].private_data
    var w = _words(blk)
    var cursor = Int(w[unsafe_offset=_W_CURSOR])
    if Int(w[unsafe_offset=_W_FAIL_AT]) == cursor:
        return 5
    if cursor >= Int(w[unsafe_offset=_W_COUNT]):
        # The end of the stream is a *released* array: all zeroes, and in
        # particular a null `release`.
        for i in range(10):
            _words(Int(dest))[unsafe_offset=i] = 0
        return 0
    var arr = Int(w[unsafe_offset=_W_ARRAYS + cursor])
    _move_struct(arr, Int(dest), 80, 8)
    w[unsafe_offset=_W_CURSOR] = Int64(cursor + 1)
    return 0


def _src_get_last_error(
    s: Pointer[CArrowArrayStream, MutUntrackedOrigin]
) abi("C") -> Int:
    return Int(_words(s[].private_data)[unsafe_offset=_W_MESSAGE])


def _src_release(
    s: Pointer[CArrowArrayStream, MutUntrackedOrigin]
) abi("C") -> None:
    """Free whatever was never handed over, then the block itself."""
    var blk = s[].private_data
    if blk != 0:
        var w = _words(blk)
        release_c_schema(Int(w[unsafe_offset=_W_SCHEMA]))
        var n = Int(w[unsafe_offset=_W_COUNT])
        for i in range(Int(w[unsafe_offset=_W_CURSOR]), n):
            release_c_array(Int(w[unsafe_offset=_W_ARRAYS + i]))
        Pointer[Int64, MutUntrackedOrigin](
            unsafe_from_address=blk
        ).unsafe_free()
    s[].private_data = 0
    s[].release = 0


struct StreamSource(Movable):
    """A C `ArrowArrayStream` over arrays this process exported.

    The consumer takes ownership of the stream; `address()` is what to hand to
    `ImportedStream`. Nothing here is freed by this object — a stream that has
    been given away must be released by whoever took it, which is exactly the
    behaviour the consumer is being tested for.
    """

    var _stream: Int
    var _block: Int

    def __init__(
        out self, batches: Int, rows_of_first: Int, fail_at: Int
    ) raises:
        """`batches` struct arrays of growing length, and where to fail."""
        var msg = String("the producer stopped early")
        var mlen = msg.byte_length() + 1
        var words = _W_ARRAYS + batches + (mlen + 7) // 8
        var blk = unsafe_alloc[Int64](words)
        for i in range(words):
            blk[unsafe_offset=i] = 0
        self._block = Int(blk)
        var msg_at = self._block + 8 * (_W_ARRAYS + batches)
        var mp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=msg_at)
        var mb = msg.as_bytes()
        for i in range(len(mb)):
            mp[unsafe_offset=i] = mb[i]
        mp[unsafe_offset=len(mb)] = 0
        blk[unsafe_offset=_W_COUNT] = Int64(batches)
        blk[unsafe_offset=_W_FAIL_AT] = Int64(fail_at)
        blk[unsafe_offset=_W_MESSAGE] = Int64(msg_at)
        for i in range(batches):
            var built = make_struct_arena(rows_of_first + i, Int64(100 * i))
            var e = export_c(built[0], built[1])
            var raw = e.into_raw()
            blk[unsafe_offset=_W_ARRAYS + i] = Int64(raw[0])
            if i == 0:
                blk[unsafe_offset=_W_SCHEMA] = Int64(raw[1])
            else:
                # One schema for the whole stream, so the rest are released
                # here rather than leaked.
                release_c_schema(raw[1])
        var st = unsafe_alloc[Int64](5)
        self._stream = Int(st)
        Pointer[StreamGetSchemaFn, MutUntrackedOrigin](
            unsafe_from_address=self._stream
        )[] = _src_get_schema
        Pointer[StreamGetNextFn, MutUntrackedOrigin](
            unsafe_from_address=self._stream + 8
        )[] = _src_get_next
        Pointer[StreamGetLastErrorFn, MutUntrackedOrigin](
            unsafe_from_address=self._stream + 16
        )[] = _src_get_last_error
        Pointer[StreamReleaseFn, MutUntrackedOrigin](
            unsafe_from_address=self._stream + 24
        )[] = _src_release
        st[unsafe_offset=4] = Int64(self._block)

    def __init__(out self, *, deinit move: Self):
        self._stream = move._stream
        self._block = move._block

    def address(self) -> Int:
        return self._stream

    def free_stream_storage(mut self):
        """Free the 40 bytes the `ArrowArrayStream` itself sits in.

        Separate from `release`, and only correct after the consumer has
        released the stream: the callback frees what the stream *owns*, and
        the struct it was written into belongs to whoever allocated it.
        """
        if self._stream != 0:
            Pointer[Int64, MutUntrackedOrigin](
                unsafe_from_address=self._stream
            ).unsafe_free()
            self._stream = 0


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
