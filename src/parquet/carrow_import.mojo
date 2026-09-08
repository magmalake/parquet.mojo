"""The Arrow C Data Interface, inbound — another runtime's arrays read into
this one.

`parquet.carrow` is the outbound half: an `ArrayData` tree becomes a C
`ArrowSchema`/`ArrowArray` pair that pyarrow, DuckDB or a Rust crate can
consume. This module is its inverse. `import_c` turns that same pair back into
an `ArrayData` in an `ArrayArena`, and `ImportedStream` drives the third struct
of the interface, `ArrowArrayStream`, which is how a producer hands over a
*sequence* of batches:

```c
struct ArrowArrayStream {
  int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);
  int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);
  const char* (*get_last_error)(struct ArrowArrayStream*);
  void (*release)(struct ArrowArrayStream*);
  void* private_data;
};
```

That is what a scan returns, so it is the entry point a LanceDB or ADBC
binding actually needs.

**Importing copies.** `ArrayData` owns its buffers as `List[UInt8]`, so every
buffer a producer hands over is copied into Mojo memory. The alternative — an
`ArrayData` that borrows a foreign pointer and keeps the producer's release
callback alive behind it — would be zero-copy but changes the type every other
module in this library reads, so it is deliberately not what this does. See
`docs`/the pull request for the measured cost; on a 16-column, 65k-row batch
the copy runs at memory-copy speed and is a rounding error next to decoding
the same data out of Parquet.

**Who releases what.** The C Data Interface makes the *consumer* responsible
for the structs it is handed: it calls the root's `release`, which frees the
whole tree, and a released struct is recognised by a null `release` pointer.
Children are never released on their own — the root owns them. This module
implements exactly that side of the convention:

* `import_c` **borrows**. It reads the pair and copies out of it, and it does
  not release anything. A caller that owns the pair — because a producer moved
  it into caller storage — decides when to release it.
* `ImportedArray` **owns**. Constructing one asserts "this pair is mine now";
  it releases the root array and the root schema exactly once, on `release()`
  or on destruction, and `release()` is idempotent because it checks for the
  null `release` pointer first.
* `ImportedStream` owns the stream, the schema it pulled out of `get_schema`,
  and every array `get_next` moves into it. Each array is released as soon as
  its batch has been copied, so a long scan holds one batch of producer memory
  at a time and not the whole stream.

**Nothing about a producer is trusted.** `n_buffers` is checked against the
format string, `n_children` against the type, offsets for monotonicity and
against the child length they index into, and every format string this module
cannot name is an error carrying the string itself. The interface does *not*
carry buffer sizes — a `const void**` and nothing more — so a truncated
`utf8` data buffer is undetectable by construction; what can be cross-checked
is checked, and the limit is stated rather than papered over.

```mojo
var arena = ArrayArena()
var root = import_c(arena, array_addr, schema_addr)   # borrowing
var batch = ImportedArray(array_addr, schema_addr).into_batch()  # owning
```
"""

from std.bit import pop_count
from std.memory.alloc import unsafe_alloc

from parquet.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_DATE32,
    AT_DECIMAL128,
    AT_FIXED_SIZE_BINARY,
    AT_FLOAT16,
    AT_FLOAT32,
    AT_FLOAT64,
    AT_INT8,
    AT_INT16,
    AT_INT32,
    AT_INT64,
    AT_LARGE_BINARY,
    AT_LARGE_LIST,
    AT_LARGE_UTF8,
    AT_LIST,
    AT_MAP,
    AT_NULL,
    AT_STRUCT,
    AT_TIME32,
    AT_TIME64,
    AT_TIMESTAMP,
    AT_UINT8,
    AT_UINT16,
    AT_UINT32,
    AT_UINT64,
    AT_UTF8,
    TU_MICRO,
    TU_MILLI,
    TU_NANO,
    TU_SECOND,
    ArrayArena,
    ArrayData,
    ArrowType,
)
from parquet.carrow import (
    ARROW_FLAG_NULLABLE,
    ArrayReleaseFn,
    CArrowArray,
    CArrowSchema,
    SchemaReleaseFn,
    n_buffers_for_type,
)
from parquet.reader import RecordBatch

comptime MAX_IMPORT_DEPTH = 64
"""How deep a foreign type may nest before we call it a malformed producer.

Children are reached through pointers a producer wrote, so a cycle is possible
and would otherwise be an unbounded walk. Real Arrow schemas are a handful of
levels deep; anything past this is a bug on the other side of the interface.
"""

comptime _MAX_METADATA_KEYS = 4096
comptime _MAX_METADATA_LEN = 1 << 24


# ── reading the C structs ──────────────────────────────────────────────────


def _c_array(addr: Int) -> Pointer[CArrowArray, MutUntrackedOrigin]:
    return Pointer[CArrowArray, MutUntrackedOrigin](unsafe_from_address=addr)


def _c_schema(addr: Int) -> Pointer[CArrowSchema, MutUntrackedOrigin]:
    return Pointer[CArrowSchema, MutUntrackedOrigin](unsafe_from_address=addr)


def _cstring(addr: Int) -> String:
    """A NUL-terminated `const char*`, or the empty string for NULL."""
    if addr == 0:
        return String()
    return String(
        unsafe_from_utf8_ptr=Pointer[UInt8, ImmUntrackedOrigin](
            unsafe_from_address=addr
        )
    )


def _u8(addr: Int) -> Pointer[UInt8, ImmUntrackedOrigin]:
    return Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=addr)


def release_c_array(addr: Int):
    """Call a producer's `release` on an `ArrowArray`, once.

    A released struct carries a null `release`, which is both how the producer
    tells us it is gone and how this stays idempotent. Only ever called on a
    *root*: the interface says children are freed by their parent, and calling
    a child's release does nothing but mark it.
    """
    if addr == 0:
        return
    var p = _c_array(addr)
    if p[].release == 0:
        return
    var slot = Pointer[ArrayReleaseFn, MutUntrackedOrigin](
        unsafe_from_address=addr + 64
    )
    slot[](p)


def release_c_schema(addr: Int):
    """Call a producer's `release` on an `ArrowSchema`, once."""
    if addr == 0:
        return
    var p = _c_schema(addr)
    if p[].release == 0:
        return
    var slot = Pointer[SchemaReleaseFn, MutUntrackedOrigin](
        unsafe_from_address=addr + 56
    )
    slot[](p)


# ── format strings ─────────────────────────────────────────────────────────


def _unsupported(fmt: StringSlice, why: StringSlice) raises -> ArrowType:
    raise Error(
        String(
            "parquet.carrow: ",
            why,
            ' — Arrow format string "',
            fmt,
            '"',
        )
    )


def _digits(b: Span[Int, _], mut i: Int, fmt: StringSlice) raises -> Int:
    var start = i
    var v = 0
    while i < len(b) and b[i] >= 0x30 and b[i] <= 0x39:
        v = v * 10 + (b[i] - 0x30)
        i += 1
    if i == start:
        _ = _unsupported(fmt, "expected a number in the format string")
    return v


def _time_unit(c: Int, fmt: StringSlice) raises -> Int:
    if c == ord("s"):
        return TU_SECOND
    if c == ord("m"):
        return TU_MILLI
    if c == ord("u"):
        return TU_MICRO
    if c == ord("n"):
        return TU_NANO
    _ = _unsupported(fmt, "unknown time unit")
    return TU_MICRO


def parse_format(fmt: StringSlice) raises -> ArrowType:
    """One Arrow C Data Interface format string as an `ArrowType`.

    Every string this library can *write* round-trips, plus the temporal and
    decimal spellings a foreign producer will send that our writer never
    emits. Everything else raises with the string in the message: guessing at
    an unrecognised format would mean reinterpreting somebody else's buffer,
    which is the one failure mode that produces wrong numbers silently rather
    than an error.
    """
    # Widened to `Int` up front: a format string is at most a couple of dozen
    # bytes, and comparing against `ord(...)` — which is an `Int` — beats
    # spelling `UInt8(ord(...))` at every one of the forty-odd branches below.
    var raw = fmt.as_bytes()
    var b = List[Int]()
    for i in range(len(raw)):
        b.append(Int(raw[i]))
    var n = len(b)
    if n == 0:
        return _unsupported(fmt, "empty format string")
    var c0 = b[0]

    if n == 1:
        if c0 == ord("n"):
            return ArrowType(AT_NULL)
        if c0 == ord("b"):
            return ArrowType(AT_BOOL)
        if c0 == ord("c"):
            return ArrowType(AT_INT8)
        if c0 == ord("C"):
            return ArrowType(AT_UINT8)
        if c0 == ord("s"):
            return ArrowType(AT_INT16)
        if c0 == ord("S"):
            return ArrowType(AT_UINT16)
        if c0 == ord("i"):
            return ArrowType(AT_INT32)
        if c0 == ord("I"):
            return ArrowType(AT_UINT32)
        if c0 == ord("l"):
            return ArrowType(AT_INT64)
        if c0 == ord("L"):
            return ArrowType(AT_UINT64)
        if c0 == ord("e"):
            return ArrowType(AT_FLOAT16)
        if c0 == ord("f"):
            return ArrowType(AT_FLOAT32)
        if c0 == ord("g"):
            return ArrowType(AT_FLOAT64)
        if c0 == ord("u"):
            return ArrowType(AT_UTF8)
        if c0 == ord("U"):
            return ArrowType(AT_LARGE_UTF8)
        if c0 == ord("z"):
            return ArrowType(AT_BINARY)
        if c0 == ord("Z"):
            return ArrowType(AT_LARGE_BINARY)
        return _unsupported(fmt, "unknown primitive format")

    # `w:<width>` — fixed size binary.
    if c0 == ord("w") and b[1] == ord(":"):
        var i = 2
        var width = _digits(b, i, fmt)
        if i != n or width <= 0:
            return _unsupported(fmt, "malformed fixed-size-binary width")
        var t = ArrowType(AT_FIXED_SIZE_BINARY)
        t.byte_width = width
        return t^

    # `d:<precision>,<scale>` or `d:<precision>,<scale>,<bitwidth>`.
    if c0 == ord("d") and b[1] == ord(":"):
        var i = 2
        var precision = _digits(b, i, fmt)
        if i >= n or b[i] != ord(","):
            return _unsupported(fmt, "malformed decimal format")
        i += 1
        var negative = i < n and b[i] == ord("-")
        if negative:
            i += 1
        var scale = _digits(b, i, fmt)
        if negative:
            scale = -scale
        var bits = 128
        if i < n and b[i] == ord(","):
            i += 1
            bits = _digits(b, i, fmt)
        if i != n:
            return _unsupported(fmt, "trailing text in a decimal format")
        if bits != 128:
            # `ArrayData` has no decimal32/64/256 buffer layout to put these
            # in, and reinterpreting a 256-bit value as 128 bits would be a
            # wrong answer rather than a missing one.
            return _unsupported(fmt, "only 128-bit decimals are supported")
        var t = ArrowType(AT_DECIMAL128)
        t.byte_width = 16
        t.precision = precision
        t.scale = scale
        return t^

    # Temporal types, all spelled `t` + a family letter + a unit.
    if c0 == ord("t"):
        var fam = b[1]
        if fam == ord("d"):
            if n == 3 and b[2] == ord("D"):
                return ArrowType(AT_DATE32)
            return _unsupported(fmt, "date64 has no ArrowType here")
        if fam == ord("t"):
            if n != 3:
                return _unsupported(fmt, "malformed time format")
            var unit = _time_unit(b[2], fmt)
            # `tts`/`ttm` are the 32-bit spellings and `ttu`/`ttn` the 64-bit
            # ones; the Arrow spec ties the storage width to the unit, so the
            # unit is what picks the type id.
            var t = ArrowType(AT_TIME32 if unit <= TU_MILLI else AT_TIME64)
            t.unit = unit
            return t^
        if fam == ord("s"):
            if n < 3:
                return _unsupported(fmt, "malformed timestamp format")
            var unit = _time_unit(b[2], fmt)
            if n > 3 and b[3] != ord(":"):
                return _unsupported(fmt, "malformed timestamp format")
            var tz = String()
            if n > 4:
                tz = String(StringSlice(unsafe_from_utf8=raw[4:]))
            var t = ArrowType(AT_TIMESTAMP)
            t.unit = unit
            t.tz = tz^
            return t^
        return _unsupported(fmt, "duration and interval types are not read")

    if c0 == ord("+"):
        var rest = Span(b)[1:]
        if len(rest) == 1:
            if rest[0] == ord("l"):
                return ArrowType(AT_LIST)
            if rest[0] == ord("L"):
                return ArrowType(AT_LARGE_LIST)
            if rest[0] == ord("s"):
                return ArrowType(AT_STRUCT)
            if rest[0] == ord("m"):
                return ArrowType(AT_MAP)
        return _unsupported(
            fmt,
            (
                "unions, run-end encoding, list views and fixed-size lists are"
                " not read"
            ),
        )

    return _unsupported(fmt, "unknown format")


def n_children_for_type(id: Int) -> Int:
    """How many children a type must have, or -1 where any number is legal."""
    if id == AT_LIST or id == AT_LARGE_LIST or id == AT_MAP:
        return 1
    if id == AT_STRUCT:
        return -1
    return 0


# ── metadata ───────────────────────────────────────────────────────────────


def _read_i32(p: Pointer[UInt8, ImmUntrackedOrigin], at: Int) -> Int:
    var u = UInt32(0)
    for k in range(4):
        u |= UInt32(p[unsafe_offset=at + k]) << UInt32(8 * k)
    return Int(Int32(u))


def extension_name(metadata: Int) raises -> String:
    """`ARROW:extension:name` out of a C schema's metadata block, if present.

    The encoding is Arrow's own: an `int32` count, then that many
    (`int32` length, bytes) pairs alternating key and value, native-endian and
    unaligned. Nothing in the interface bounds the block, so the counts are
    sanity-checked before they are used as loop bounds — a garbage `n` here
    would otherwise be an unbounded read of somebody else's address space.
    """
    if metadata == 0:
        return String()
    var p = _u8(metadata)
    var n = _read_i32(p, 0)
    if n < 0 or n > _MAX_METADATA_KEYS:
        raise Error(
            String("parquet.carrow: implausible metadata key count ", n)
        )
    var at = 4
    for _ in range(n):
        var klen = _read_i32(p, at)
        if klen < 0 or klen > _MAX_METADATA_LEN:
            raise Error(
                String("parquet.carrow: implausible metadata key length ", klen)
            )
        var key_at = at + 4
        at = key_at + klen
        var vlen = _read_i32(p, at)
        if vlen < 0 or vlen > _MAX_METADATA_LEN:
            raise Error(
                String(
                    "parquet.carrow: implausible metadata value length ", vlen
                )
            )
        var val_at = at + 4
        at = val_at + vlen
        var key = _copy_bytes(metadata + key_at, 0, klen)
        if StringSlice(unsafe_from_utf8=Span(key)) == "ARROW:extension:name":
            var val = _copy_bytes(metadata + val_at, 0, vlen)
            return String(StringSlice(unsafe_from_utf8=Span(val)))
    return String()


def parquet_field_id(metadata: Int) raises -> Int32:
    """`PARQUET:field_id` out of a C schema's metadata, or -1.

    Arrow's Parquet writers put the Iceberg/Parquet field id here as decimal
    text, and this library's `ArrayData` has a slot for it. Our own exporter
    does not write the key, so a round trip through it drops the id; a foreign
    producer that does write it keeps it.
    """
    if metadata == 0:
        return -1
    var p = _u8(metadata)
    var n = _read_i32(p, 0)
    if n < 0 or n > _MAX_METADATA_KEYS:
        return -1
    var at = 4
    for _ in range(n):
        var klen = _read_i32(p, at)
        if klen < 0 or klen > _MAX_METADATA_LEN:
            return -1
        var key_at = at + 4
        at = key_at + klen
        var vlen = _read_i32(p, at)
        if vlen < 0 or vlen > _MAX_METADATA_LEN:
            return -1
        var val_at = at + 4
        at = val_at + vlen
        var key = _copy_bytes(metadata + key_at, 0, klen)
        if StringSlice(unsafe_from_utf8=Span(key)) == "PARQUET:field_id":
            var val = _copy_bytes(metadata + val_at, 0, vlen)
            var v = 0
            var neg = False
            for k in range(len(val)):
                if k == 0 and Int(val[k]) == ord("-"):
                    neg = True
                elif val[k] >= 0x30 and val[k] <= 0x39:
                    v = v * 10 + Int(val[k] - 0x30)
                else:
                    return -1
            return Int32(-v if neg else v)
    return -1


# ── buffer copies ──────────────────────────────────────────────────────────


def _copy_bytes(src: Int, at: Int, n: Int) -> List[UInt8]:
    var out = List[UInt8]()
    if n <= 0 or src == 0:
        return out^
    out.resize(n, 0)
    var p = _u8(src + at)
    var d = out.unsafe_ptr()
    for i in range(n):
        d[unsafe_offset=i] = p[unsafe_offset=i]
    return out^


def _copy_bits(src: Int, bit_off: Int, n: Int, mut nulls: Int) -> List[UInt8]:
    """Copy `n` bits from an LSB-first bitmap into a fresh one starting at 0,
    and count the zeros.

    A foreign array's `offset` need not be a multiple of eight, so the general
    case is a shifted copy rather than a `memcpy`; the aligned case is the one
    that actually happens and gets the straight byte copy. The zeros are
    counted here rather than trusted from `null_count`, which the interface
    lets a producer leave at -1 for "not computed" and which in any case
    describes the whole array and not the window we are slicing out of it.
    """
    nulls = 0
    var out = List[UInt8]()
    if src == 0 or n <= 0:
        return out^
    var nbytes = (n + 7) // 8
    out.resize(nbytes, 0)
    var d = out.unsafe_ptr()
    var p = _u8(src)
    var shift = bit_off % 8
    var base = bit_off // 8
    var last = (bit_off + n - 1) // 8
    if shift == 0:
        for k in range(nbytes):
            d[unsafe_offset=k] = p[unsafe_offset=base + k]
    else:
        for k in range(nbytes):
            var lo = p[unsafe_offset=base + k] >> UInt8(shift)
            var hi = UInt8(0)
            if base + k + 1 <= last:
                hi = p[unsafe_offset=base + k + 1] << UInt8(8 - shift)
            d[unsafe_offset=k] = lo | hi
        _ = last
    # Bits past `n` in the final byte are whatever the neighbours held; mask
    # them off so the popcount below — and every later reader — sees zero.
    var tail = n % 8
    if tail:
        d[unsafe_offset=nbytes - 1] &= UInt8((1 << tail) - 1)
    var ones = 0
    for k in range(nbytes):
        ones += Int(pop_count(d[unsafe_offset=k]))
    nulls = n - ones
    return out^


# ── the import walk ────────────────────────────────────────────────────────


def _offset_at(buf: Int, i: Int, wide: Bool) -> Int:
    var p = _u8(buf)
    if wide:
        var u = UInt64(0)
        for k in range(8):
            u |= UInt64(p[unsafe_offset=i * 8 + k]) << UInt64(8 * k)
        return Int(Int64(u))
    var v = UInt32(0)
    for k in range(4):
        v |= UInt32(p[unsafe_offset=i * 4 + k]) << UInt32(8 * k)
    return Int(Int32(v))


def _check(cond: Bool, message: StringSlice) raises:
    if not cond:
        raise Error(String("parquet.carrow: ", message))


def _import_node(
    mut arena: ArrayArena,
    array_addr: Int,
    schema_addr: Int,
    rel_start: Int,
    count: Int,
    depth: Int,
) raises -> Int:
    """Copy one foreign array — and everything under it — into `arena`.

    `rel_start` and `count` are the *logical* window wanted out of this array,
    on top of the array's own `offset`. That parameter is what makes slices
    work: Arrow's `offset` is not a field `ArrayData` has, so an imported
    array is always materialised at position zero, and a list's child is
    materialised over exactly the range its parent's offsets name. Both are
    the same operation, one window narrowing another, which is why there is
    one function and not two.

    Nodes are appended in pre-order — this node's slot first, then its whole
    first subtree, then the next — which is the order `parquet.carrow`'s
    `_collect` walks on the way out. Import and export therefore agree on the
    arena's shape as well as its values, so `export → import → export` is the
    same arena and not merely the same numbers.
    """
    _check(depth <= MAX_IMPORT_DEPTH, "foreign schema nests implausibly deep")
    _check(array_addr != 0 and schema_addr != 0, "NULL ArrowArray/ArrowSchema")
    ref arr = _c_array(array_addr)[]
    ref sch = _c_schema(schema_addr)[]
    _check(arr.release != 0, "the producer handed over a released ArrowArray")
    _check(sch.release != 0, "the producer handed over a released ArrowSchema")

    var fmt = _cstring(sch.format)
    var type = parse_format(fmt)
    _check(
        sch.dictionary == 0 and arr.dictionary == 0,
        "dictionary-encoded arrays are not read",
    )

    var want_buffers = n_buffers_for_type(type.id)
    if Int(arr.n_buffers) != want_buffers:
        raise Error(
            String(
                "parquet.carrow: ",
                fmt,
                " needs ",
                want_buffers,
                " buffers, the producer gave ",
                arr.n_buffers,
            )
        )
    var want_children = n_children_for_type(type.id)
    if want_children >= 0 and Int(arr.n_children) != want_children:
        raise Error(
            String(
                "parquet.carrow: ",
                fmt,
                " needs ",
                want_children,
                " children, the producer gave ",
                arr.n_children,
            )
        )
    _check(
        Int(sch.n_children) == Int(arr.n_children),
        "ArrowSchema and ArrowArray disagree on the number of children",
    )
    _check(arr.length >= 0, "negative array length")
    _check(arr.offset >= 0, "negative array offset")
    _check(
        rel_start >= 0 and count >= 0 and rel_start + count <= Int(arr.length),
        "the window asked for runs past the end of the array",
    )
    if Int(arr.n_children) > 0:
        _check(arr.children != 0, "n_children > 0 with a NULL children array")
    if want_buffers > 0:
        _check(arr.buffers != 0, "n_buffers > 0 with a NULL buffers array")

    var phys = Int(arr.offset) + rel_start
    var bufs = arr.buffers
    var buf0 = _offset_at(bufs, 0, True) if want_buffers > 0 else 0

    # Reserve this node's slot before descending, so the arena comes out in
    # pre-order; `swap` fills it in once the children are in place.
    var me = arena.add(ArrayData())

    var kids = List[Int]()
    var offsets = List[Int32]()
    var large_offsets = List[Int64]()
    var values = List[UInt8]()

    var is_var_binary = (
        type.id == AT_UTF8
        or type.id == AT_BINARY
        or type.id == AT_LARGE_UTF8
        or type.id == AT_LARGE_BINARY
    )
    var is_list = type.id == AT_LIST or type.id == AT_MAP
    var is_large_list = type.id == AT_LARGE_LIST
    var wide = (
        type.id == AT_LARGE_UTF8
        or type.id == AT_LARGE_BINARY
        or type.id == AT_LARGE_LIST
    )

    if is_var_binary or is_list or is_large_list:
        var obuf = _offset_at(bufs, 1, True)
        _check(obuf != 0, "a variable-length array with a NULL offsets buffer")
        # Offsets are read once, checked for monotonicity, and rebased to
        # zero: the copy starts at the parent's window, so the first offset
        # has to become 0 for the values we actually copy.
        var first = _offset_at(obuf, phys, wide)
        _check(first >= 0, "negative first offset")
        var prev = first
        for i in range(count + 1):
            var v = _offset_at(obuf, phys + i, wide)
            _check(v >= prev, "offsets are not monotonically increasing")
            prev = v
            if wide:
                large_offsets.append(Int64(v - first))
            else:
                offsets.append(Int32(v - first))
        var span = prev - first
        if is_var_binary:
            var dbuf = _offset_at(bufs, 2, True)
            _check(
                dbuf != 0 or span == 0,
                "a variable-length array with a NULL data buffer",
            )
            values = _copy_bytes(dbuf, first, span)
        else:
            var child_a = _offset_at(arr.children, 0, True)
            var child_s = _offset_at(sch.children, 0, True)
            _check(child_a != 0 and child_s != 0, "NULL list child")
            _check(
                first + span <= Int(_c_array(child_a)[].length),
                (
                    "a list's offsets run past the end of its child — the"
                    " producer's buffers do not agree"
                ),
            )
            kids.append(
                _import_node(arena, child_a, child_s, first, span, depth + 1)
            )
    elif type.id == AT_STRUCT:
        for k in range(Int(arr.n_children)):
            var child_a = _offset_at(arr.children, k, True)
            var child_s = _offset_at(sch.children, k, True)
            _check(child_a != 0 and child_s != 0, "NULL struct child")
            # A struct's own offset and length narrow every child: Arrow's
            # rule is that the parent slices its children, and the child's own
            # offset is added under that.
            _check(
                phys + count <= Int(_c_array(child_a)[].length),
                (
                    "a struct child is shorter than the struct — the"
                    " producer's arrays do not agree"
                ),
            )
            kids.append(
                _import_node(arena, child_a, child_s, phys, count, depth + 1)
            )
    elif type.id != AT_NULL:
        var width = type.fixed_width()
        if type.id == AT_BOOL:
            var vbuf = _offset_at(bufs, 1, True)
            _check(vbuf != 0 or count == 0, "a bool array with NULL values")
            var ignored = 0
            values = _copy_bits(vbuf, phys, count, ignored)
        else:
            _check(width > 0, "a fixed-width type with no width")
            var vbuf = _offset_at(bufs, 1, True)
            _check(
                vbuf != 0 or count == 0, "a primitive array with NULL values"
            )
            values = _copy_bytes(vbuf, phys * width, count * width)

    var node = ArrayData(type^, _cstring(sch.name))
    node.nullable = (sch.flags & ARROW_FLAG_NULLABLE) != 0
    node.field_id = parquet_field_id(sch.metadata)
    node.type.extension = extension_name(sch.metadata)
    node.length = count
    if node.type.id == AT_NULL:
        # A null array carries no buffers at all; every element is null by
        # construction, and that is what `null_count` has to say.
        node.null_count = count
    else:
        var nulls = 0
        node.validity = _copy_bits(buf0, phys, count, nulls)
        node.null_count = nulls
    node.offsets = offsets^
    node.large_offsets = large_offsets^
    node.values = values^
    node.children = kids^
    swap(arena.nodes[me], node)
    return me


def import_c(mut arena: ArrayArena, array: Int, schema: Int) raises -> Int:
    """Copy a foreign C array into `arena`; return the index of its root.

    Borrowing: the pair is read, never released. The caller owns it — see
    `ImportedArray` for the owning form.
    """
    _check(array != 0 and schema != 0, "NULL ArrowArray/ArrowSchema")
    var length = Int(_c_array(array)[].length)
    return _import_node(arena, array, schema, 0, length, 0)


def import_batch_c(array: Int, schema: Int) raises -> RecordBatch:
    """A foreign C array as a `RecordBatch`.

    A stream's arrays are struct arrays by definition — the struct's fields
    are the batch's columns — so a `+s` root is unwrapped into one root per
    field. Anything else becomes a one-column batch, which is what a caller
    importing a bare column wants.

    A struct root with nulls of its own has no representation here, because a
    `RecordBatch` is a list of columns and not an array; that raises rather
    than dropping the row-level nulls on the floor.
    """
    var batch = RecordBatch()
    var arena = ArrayArena()
    var root = import_c(arena, array, schema)
    if arena.nodes[root].type.id == AT_STRUCT:
        _check(
            arena.nodes[root].null_count == 0,
            (
                "a struct array with nulls of its own cannot be a RecordBatch;"
                " import it as an array instead"
            ),
        )
        batch.num_rows = arena.nodes[root].length
        batch.roots = arena.nodes[root].children.copy()
    else:
        batch.num_rows = arena.nodes[root].length
        batch.roots = [root]
    batch.arena = arena^
    return batch^


# ── owning handles ─────────────────────────────────────────────────────────


struct ImportedArray(Movable):
    """One C `ArrowArray`/`ArrowSchema` pair, owned by us until released.

    Construct this when a producer has *moved* the two structs into storage
    you hold — which is what "consuming" means in the C Data Interface — and
    it will call the root release callbacks exactly once, on `release()` or at
    destruction. The addresses stay valid afterwards, but their contents do
    not, so import before releasing.
    """

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

    def into_arena(self, mut arena: ArrayArena) raises -> Int:
        return import_c(arena, self.array, self.schema)

    def into_batch(self) raises -> RecordBatch:
        return import_batch_c(self.array, self.schema)

    def release(mut self):
        if not self._owned:
            return
        self._owned = False
        release_c_array(self.array)
        release_c_schema(self.schema)

    def __deinit__(deinit self):
        self.release()


# ── ArrowArrayStream ───────────────────────────────────────────────────────


@fieldwise_init
struct CArrowArrayStream(Copyable, Movable):
    """The C `ArrowArrayStream`, field for field. Pointers are addresses."""

    var get_schema: Int
    var get_next: Int
    var get_last_error: Int
    var release: Int
    var private_data: Int


comptime StreamGetSchemaFn = def(
    Pointer[CArrowArrayStream, MutUntrackedOrigin],
    Pointer[CArrowSchema, MutUntrackedOrigin],
) thin abi("C") -> Int32
comptime StreamGetNextFn = def(
    Pointer[CArrowArrayStream, MutUntrackedOrigin],
    Pointer[CArrowArray, MutUntrackedOrigin],
) thin abi("C") -> Int32
comptime StreamGetLastErrorFn = def(
    Pointer[CArrowArrayStream, MutUntrackedOrigin]
) thin abi("C") -> Int
comptime StreamReleaseFn = def(
    Pointer[CArrowArrayStream, MutUntrackedOrigin]
) thin abi("C") -> None


struct ImportedStream(Movable):
    """A producer's `ArrowArrayStream`, consumed batch by batch.

    This is the shape a scan arrives in: one schema up front, then arrays
    until the producer signals the end by handing back a *released* array —
    a null `release` pointer and nothing else. `get_next` returning non-zero
    is a real error, and `get_last_error` says what it was; the two are
    distinguished here because conflating them turns a failed scan into a
    short one.

    Ownership runs three ways and all three are ours. The stream is released
    when this object dies. The schema is moved out of `get_schema` into
    storage this object allocates, and released with it. Each array `get_next`
    moves into us is released as soon as its batch has been copied, which is
    what keeps a long scan to one batch of producer memory at a time rather
    than the whole stream.
    """

    var stream: Int
    """Address of the caller's `ArrowArrayStream`."""
    var _schema: Int
    """Our own 72-byte `ArrowSchema`, filled by `get_schema`."""
    var _owned: Bool
    var _done: Bool

    def __init__(out self, stream: Int) raises:
        """Take ownership of `stream` and pull its schema."""
        _check(stream != 0, "NULL ArrowArrayStream")
        self.stream = stream
        self._owned = True
        self._done = False
        var s = Pointer[CArrowArrayStream, MutUntrackedOrigin](
            unsafe_from_address=stream
        )
        if s[].release == 0:
            self._schema = 0
            raise Error(
                "parquet.carrow: the producer handed over a released"
                " ArrowArrayStream"
            )
        self._schema = _alloc_zeroed(72)
        var slot = Pointer[StreamGetSchemaFn, MutUntrackedOrigin](
            unsafe_from_address=stream
        )
        var rc = slot[](s, _c_schema(self._schema))
        var why = String()
        if rc != 0:
            why = String(
                "parquet.carrow: ArrowArrayStream get_schema failed (",
                rc,
                "): ",
                _stream_error(stream),
            )
        elif _c_schema(self._schema)[].release == 0:
            why = String(
                "parquet.carrow: ArrowArrayStream get_schema returned nothing"
            )
        if why:
            # Raising out of `__init__` leaves nothing behind to destroy, so
            # the stream and the schema storage are given back here rather
            # than leaked on the one path where the producer misbehaves.
            self.release()
            raise Error(why)

    def __init__(out self, *, deinit move: Self):
        self.stream = move.stream
        self._schema = move._schema
        self._owned = move._owned
        self._done = move._done

    def format(self) -> String:
        """The root format string of the stream's schema, usually `+s`."""
        if self._schema == 0:
            return String()
        return _cstring(_c_schema(self._schema)[].format)

    def schema_address(self) -> Int:
        return self._schema

    def next(mut self) raises -> Optional[RecordBatch]:
        """The next batch, or nothing once the stream has ended."""
        if self._done or self._schema == 0:
            return None
        var s = Pointer[CArrowArrayStream, MutUntrackedOrigin](
            unsafe_from_address=self.stream
        )
        var storage = _alloc_zeroed(80)
        var slot = Pointer[StreamGetNextFn, MutUntrackedOrigin](
            unsafe_from_address=self.stream + 8
        )
        var rc = slot[](s, _c_array(storage))
        if rc != 0:
            _free(storage)
            self._done = True
            raise Error(
                String(
                    "parquet.carrow: ArrowArrayStream get_next failed (",
                    rc,
                    "): ",
                    _stream_error(self.stream),
                )
            )
        if _c_array(storage)[].release == 0:
            # The end of the stream: a released array, which is the only
            # signal the interface has for "no more".
            _free(storage)
            self._done = True
            return None
        # `owned` releases the array whatever happens next, including a raise
        # out of the import, so a malformed batch does not also leak one.
        var owned = ImportedArray(storage, 0)
        try:
            var batch = import_batch_c(storage, self._schema)
            owned.release()
            _free(storage)
            return Optional[RecordBatch](batch^)
        except e:
            owned.release()
            _free(storage)
            self._done = True
            raise e

    def release(mut self):
        if not self._owned:
            return
        self._owned = False
        release_c_schema(self._schema)
        if self._schema != 0:
            _free(self._schema)
            self._schema = 0
        if self.stream != 0:
            var s = Pointer[CArrowArrayStream, MutUntrackedOrigin](
                unsafe_from_address=self.stream
            )
            if s[].release != 0:
                var slot = Pointer[StreamReleaseFn, MutUntrackedOrigin](
                    unsafe_from_address=self.stream + 24
                )
                slot[](s)

    def __deinit__(deinit self):
        self.release()


def _stream_error(stream: Int) -> String:
    var s = Pointer[CArrowArrayStream, MutUntrackedOrigin](
        unsafe_from_address=stream
    )
    if s[].get_last_error == 0:
        return String("(no message)")
    var slot = Pointer[StreamGetLastErrorFn, MutUntrackedOrigin](
        unsafe_from_address=stream + 16
    )
    var msg = slot[](s)
    if msg == 0:
        return String("(no message)")
    return _cstring(msg)


def _alloc_zeroed(n: Int) -> Int:
    var p = unsafe_alloc[UInt64]((n + 7) // 8)
    var b = p.unsafe_bitcast[UInt8]()
    for i in range(((n + 7) // 8) * 8):
        b[unsafe_offset=i] = 0
    return Int(p)


def _free(addr: Int):
    if addr == 0:
        return
    Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=addr).unsafe_free()
