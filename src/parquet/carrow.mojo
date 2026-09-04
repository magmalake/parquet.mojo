"""The Arrow C Data Interface — this library's public output contract.

`export_c` turns one `ArrayData` tree into the two C structs every Arrow
implementation understands:

```c
struct ArrowSchema { const char* format; const char* name; const char* metadata;
                     int64_t flags; int64_t n_children; ArrowSchema** children;
                     ArrowSchema* dictionary; void (*release)(ArrowSchema*);
                     void* private_data; };
struct ArrowArray  { int64_t length; int64_t null_count; int64_t offset;
                     int64_t n_buffers; int64_t n_children; const void** buffers;
                     ArrowArray** children; ArrowArray* dictionary;
                     void (*release)(ArrowArray*); void* private_data; };
```

Each export is **two** allocations — one for the whole schema tree, one for the
whole array tree — laid out as: the structs, then the pointer arrays, then the
format and name strings and the metadata blocks, then copies of every buffer.
The root's `private_data` is the base of its block and its `release` frees the
lot; children carry a release that only clears itself, because a consumer
releases the root and the root owns everything. Both callbacks are `abi("C")`,
so they can be called from C, Python or Rust.

Buffers are copied rather than borrowed, which is what makes the export
outlive the `RecordBatch` it came from and makes `ExportedArray` safe to hand
to a runtime that will release it whenever it likes.

```mojo
var e = export_c(batch.arena, batch.roots[0])
print(e.array, e.schema)   # two addresses, ready for _import_from_c
e.into_raw()               # hand ownership to the consumer
```
"""

from std.memory.alloc import unsafe_alloc

from parquet.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_LARGE_BINARY,
    AT_LARGE_LIST,
    AT_LARGE_UTF8,
    AT_LIST,
    AT_MAP,
    AT_NULL,
    AT_STRUCT,
    AT_UTF8,
    ArrayArena,
    ArrayData,
)

comptime ARROW_FLAG_NULLABLE: Int64 = 2


@fieldwise_init
struct CArrowSchema(Copyable, Movable):
    """The C `ArrowSchema`, field for field. Pointers are held as addresses."""

    var format: Int
    var name: Int
    var metadata: Int
    var flags: Int64
    var n_children: Int64
    var children: Int
    var dictionary: Int
    var release: Int
    var private_data: Int


@fieldwise_init
struct CArrowArray(Copyable, Movable):
    """The C `ArrowArray`, field for field. Pointers are held as addresses."""

    var length: Int64
    var null_count: Int64
    var offset: Int64
    var n_buffers: Int64
    var n_children: Int64
    var buffers: Int
    var children: Int
    var dictionary: Int
    var release: Int
    var private_data: Int


def release_array(
    p: Pointer[CArrowArray, MutUntrackedOrigin]
) abi("C") -> None:
    """Free the whole exported array tree. `private_data` is the block."""
    var block = p[].private_data
    p[].release = 0
    if block != 0:
        var base = Pointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=block
        )
        base.unsafe_free()


def release_schema(
    p: Pointer[CArrowSchema, MutUntrackedOrigin]
) abi("C") -> None:
    """Free the whole exported schema tree."""
    var block = p[].private_data
    p[].release = 0
    if block != 0:
        var base = Pointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=block
        )
        base.unsafe_free()


def release_child_array(
    p: Pointer[CArrowArray, MutUntrackedOrigin]
) abi("C") -> None:
    """A child is owned by its root; releasing one only marks it released."""
    p[].release = 0


def release_child_schema(
    p: Pointer[CArrowSchema, MutUntrackedOrigin]
) abi("C") -> None:
    p[].release = 0


def n_buffers_for(a: ArrayData) -> Int:
    var i = a.type.id
    if i == AT_NULL:
        return 0
    if i == AT_STRUCT:
        return 1
    if i == AT_LIST or i == AT_LARGE_LIST or i == AT_MAP:
        return 2
    if (
        i == AT_UTF8
        or i == AT_BINARY
        or i == AT_LARGE_UTF8
        or i == AT_LARGE_BINARY
    ):
        return 3
    return 2


def _align8(n: Int) -> Int:
    return (n + 7) & ~7


struct _Block(Movable):
    """A bump allocator over one `alloc[UInt64]` region."""

    var base: Pointer[UInt64, MutUntrackedOrigin]
    var size: Int
    var used: Int

    def __init__(out self, size: Int):
        self.size = _align8(size)
        self.base = unsafe_alloc[UInt64](self.size // 8)
        self.used = 0
        var p = self.base.unsafe_bitcast[UInt8]()
        for i in range(self.size):
            p[unsafe_offset=i] = 0

    def __init__(out self, *, deinit move: Self):
        self.base = move.base
        self.size = move.size
        self.used = move.used

    def address(self) -> Int:
        return Int(self.base)

    def take(mut self, n: Int) raises -> Int:
        """Reserve `n` bytes, 8-byte aligned; return their address."""
        var at = self.used
        self.used = _align8(self.used + n)
        if self.used > self.size:
            raise Error(
                String(
                    "parquet.carrow: export block overflow (",
                    self.used,
                    " > ",
                    self.size,
                    ")",
                )
            )
        return Int(self.base) + at

    def bytes_at(self, addr: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
        return Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=addr
        )

    def words_at(self, addr: Int) -> Pointer[Int64, MutUntrackedOrigin]:
        return Pointer[Int64, MutUntrackedOrigin](
            unsafe_from_address=addr
        )

    def put_bytes(mut self, data: Span[UInt8, _]) raises -> Int:
        var at = self.take(len(data) if len(data) else 1)
        var p = self.bytes_at(at)
        for i in range(len(data)):
            p[unsafe_offset=i] = data[i]
        return at

    def put_cstring(mut self, text: StringSlice) raises -> Int:
        var b = text.as_bytes()
        var at = self.take(len(b) + 1)
        var p = self.bytes_at(at)
        for i in range(len(b)):
            p[unsafe_offset=i] = b[i]
        p[unsafe_offset=len(b)] = 0
        return at


def _metadata_bytes(a: ArrayData) -> List[UInt8]:
    """Arrow's C metadata encoding: `n`, then (`len`, key, `len`, value)*."""
    var out = List[UInt8]()
    if not a.type.extension:
        return out^
    var keys: List[String] = [String("ARROW:extension:name")]
    var vals: List[String] = [a.type.extension.copy()]
    _put_i32(out, Int32(len(keys)))
    for i in range(len(keys)):
        _put_i32(out, Int32(keys[i].byte_length()))
        out.extend(keys[i].as_bytes())
        _put_i32(out, Int32(vals[i].byte_length()))
        out.extend(vals[i].as_bytes())
    return out^


def _put_i32(mut out: List[UInt8], v: Int32):
    var u = UInt32(v)
    for k in range(4):
        out.append(UInt8((u >> UInt32(8 * k)) & 0xFF))


def _validity_bytes(a: ArrayData) -> Int:
    if a.null_count == 0:
        return 0
    return (a.length + 7) // 8


def _values_bytes(a: ArrayData) -> Int:
    var i = a.type.id
    if i == AT_BOOL:
        return (a.length + 7) // 8
    if (
        i == AT_UTF8
        or i == AT_BINARY
        or i == AT_LARGE_UTF8
        or i == AT_LARGE_BINARY
    ):
        return len(a.values)
    if (
        i == AT_STRUCT
        or i == AT_LIST
        or i == AT_LARGE_LIST
        or i == AT_MAP
        or i == AT_NULL
    ):
        return 0
    return a.type.fixed_width() * a.length


def _offsets_bytes(a: ArrayData) -> Int:
    var i = a.type.id
    if i == AT_UTF8 or i == AT_BINARY or i == AT_LIST or i == AT_MAP:
        return 4 * (a.length + 1)
    if i == AT_LARGE_UTF8 or i == AT_LARGE_BINARY or i == AT_LARGE_LIST:
        return 8 * (a.length + 1)
    return 0


def _collect(arena: ArrayArena, root: Int, mut order: List[Int]):
    """Depth-first order of an array and its children, parents first."""
    var stack: List[Int] = [root]
    while len(stack):
        var node = stack.pop()
        order.append(node)
        ref kids = arena.nodes[node].children
        for k in range(len(kids)):
            stack.append(kids[len(kids) - 1 - k])


struct ExportedArray(Movable):
    """An exported pair of C structs, still owned by Mojo until `into_raw`."""

    var array: Int
    """Address of the `ArrowArray`."""
    var schema: Int
    """Address of the `ArrowSchema`."""
    var _owned: Bool

    def __init__(out self, array: Int, schema: Int):
        self.array = array
        self.schema = schema
        self._owned = True

    def __init__(out self, *, deinit move: Self):
        self.array = move.array
        self.schema = move.schema
        self._owned = move._owned

    def into_raw(mut self) -> Tuple[Int, Int]:
        """Give the two structs to the consumer; they must release them."""
        self._owned = False
        return (self.array, self.schema)

    def release(mut self):
        if not self._owned:
            return
        self._owned = False
        var a = Pointer[CArrowArray, MutUntrackedOrigin](
            unsafe_from_address=self.array
        )
        if a[].release != 0:
            release_array(a)
        var s = Pointer[CArrowSchema, MutUntrackedOrigin](
            unsafe_from_address=self.schema
        )
        if s[].release != 0:
            release_schema(s)

    def __deinit__(deinit self):
        self.release()


def _export_schema(
    arena: ArrayArena, root: Int, order: List[Int]
) raises -> Int:
    var n = len(order)
    var index = Dict[Int, Int]()
    for i in range(n):
        index[order[i]] = i
    # Size the block.
    var size = n * 72  # nine 8-byte words per ArrowSchema
    for node in order:
        ref a = arena.nodes[node]
        size += _align8(len(a.children) * 8)
        size += _align8(a.type.format().byte_length() + 1)
        size += _align8(a.name.byte_length() + 1)
        var md = _metadata_bytes(a)
        if len(md):
            size += _align8(len(md))
    var blk = _Block(size + 64)
    var structs = blk.take(n * 72)
    for i in range(n):
        ref a = arena.nodes[order[i]]
        var kids = len(a.children)
        var kidptr = 0
        if kids:
            kidptr = blk.take(kids * 8)
        var fmt = blk.put_cstring(a.type.format())
        var nm = blk.put_cstring(a.name)
        var md = _metadata_bytes(a)
        var mdp = 0
        if len(md):
            mdp = blk.put_bytes(Span(md))
        var w = blk.words_at(structs + i * 72)
        w[unsafe_offset=0] = Int64(fmt)
        w[unsafe_offset=1] = Int64(nm)
        w[unsafe_offset=2] = Int64(mdp)
        w[unsafe_offset=3] = ARROW_FLAG_NULLABLE if a.nullable else 0
        w[unsafe_offset=4] = Int64(kids)
        w[unsafe_offset=5] = Int64(kidptr)
        w[unsafe_offset=6] = 0
        w[unsafe_offset=7] = 0
        w[unsafe_offset=8] = Int64(blk.address()) if i == 0 else 0
        _store_schema_release(structs + i * 72, i == 0)
        if kids:
            var kp = blk.words_at(kidptr)
            for k in range(kids):
                kp[unsafe_offset=k] = Int64(structs + index[a.children[k]] * 72)
    var addr = structs
    _ = blk^
    return addr


comptime SchemaReleaseFn = def(
    Pointer[CArrowSchema, MutUntrackedOrigin]
) thin abi("C") -> None
comptime ArrayReleaseFn = def(
    Pointer[CArrowArray, MutUntrackedOrigin]
) thin abi("C") -> None


def _store_schema_release(struct_addr: Int, is_root: Bool):
    """Write the `release` function pointer into a C `ArrowSchema` (word 7)."""
    var slot = Pointer[SchemaReleaseFn, MutUntrackedOrigin](
        unsafe_from_address=struct_addr + 56
    )
    slot[] = release_schema if is_root else release_child_schema


def _store_array_release(struct_addr: Int, is_root: Bool):
    """Write the `release` function pointer into a C `ArrowArray` (word 8)."""
    var slot = Pointer[ArrayReleaseFn, MutUntrackedOrigin](
        unsafe_from_address=struct_addr + 64
    )
    slot[] = release_array if is_root else release_child_array


def _export_array(arena: ArrayArena, root: Int, order: List[Int]) raises -> Int:
    var n = len(order)
    var index = Dict[Int, Int]()
    for i in range(n):
        index[order[i]] = i
    var size = n * 80  # ten 8-byte words per ArrowArray
    for node in order:
        ref a = arena.nodes[node]
        size += _align8(len(a.children) * 8)
        size += _align8(n_buffers_for(a) * 8)
        size += _align8(_validity_bytes(a)) + 8
        size += _align8(_offsets_bytes(a)) + 8
        size += _align8(_values_bytes(a)) + 8
    var blk = _Block(size + 64)
    var structs = blk.take(n * 80)
    for i in range(n):
        ref a = arena.nodes[order[i]]
        var kids = len(a.children)
        var nbuf = n_buffers_for(a)
        var kidptr = 0
        if kids:
            kidptr = blk.take(kids * 8)
        var bufptr = 0
        if nbuf:
            bufptr = blk.take(nbuf * 8)
        var bp = blk.words_at(bufptr) if nbuf else blk.words_at(structs)
        if nbuf > 0:
            # buffer 0 is always validity, NULL when there are no nulls
            if a.null_count > 0:
                var vb = _validity_bytes(a)
                var at = blk.take(vb)
                var p = blk.bytes_at(at)
                for k in range(vb):
                    p[unsafe_offset=k] = (
                        a.validity[k] if k < len(a.validity) else 0
                    )
                bp[unsafe_offset=0] = Int64(at)
            else:
                bp[unsafe_offset=0] = 0
        var slot = 1
        var ob = _offsets_bytes(a)
        if ob > 0:
            var at = blk.take(ob)
            var p32 = blk.bytes_at(at)
            var wide = ob == 8 * (a.length + 1)
            for k in range(a.length + 1):
                var v: Int64
                if wide:
                    v = a.large_offsets[k] if k < len(a.large_offsets) else 0
                else:
                    v = Int64(a.offsets[k]) if k < len(a.offsets) else 0
                var width = 8 if wide else 4
                for b in range(width):
                    p32[unsafe_offset=k * width + b] = UInt8(
                        (UInt64(v) >> UInt64(8 * b)) & 0xFF
                    )
            bp[unsafe_offset=slot] = Int64(at)
            slot += 1
        var vb2 = _values_bytes(a)
        if slot < nbuf:
            if vb2 > 0:
                var at = blk.take(vb2)
                var p = blk.bytes_at(at)
                for k in range(vb2):
                    p[unsafe_offset=k] = a.values[k] if k < len(a.values) else 0
                bp[unsafe_offset=slot] = Int64(at)
            else:
                bp[unsafe_offset=slot] = Int64(blk.take(1))
            slot += 1
        var w = blk.words_at(structs + i * 80)
        w[unsafe_offset=0] = Int64(a.length)
        w[unsafe_offset=1] = Int64(a.null_count)
        w[unsafe_offset=2] = 0
        w[unsafe_offset=3] = Int64(nbuf)
        w[unsafe_offset=4] = Int64(kids)
        w[unsafe_offset=5] = Int64(bufptr)
        w[unsafe_offset=6] = Int64(kidptr)
        w[unsafe_offset=7] = 0
        w[unsafe_offset=8] = 0
        w[unsafe_offset=9] = Int64(blk.address()) if i == 0 else 0
        _store_array_release(structs + i * 80, i == 0)
        if kids:
            var kp = blk.words_at(kidptr)
            for k in range(kids):
                kp[unsafe_offset=k] = Int64(structs + index[a.children[k]] * 80)
    var addr = structs
    _ = blk^
    return addr


def export_c(arena: ArrayArena, root: Int) raises -> ExportedArray:
    """Export one array of an arena over the Arrow C Data Interface."""
    var order = List[Int]()
    _collect(arena, root, order)
    var schema = _export_schema(arena, root, order)
    var array = _export_array(arena, root, order)
    return ExportedArray(array, schema)
