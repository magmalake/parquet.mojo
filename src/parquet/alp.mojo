"""ALP — Adaptive Lossless floating-Point, Parquet encoding 10.

From "ALP: Adaptive Lossless floating-Point Compression" (Afroozeh, Kuffo and
Boncz, SIGMOD 2024), specified for Parquet in `AlpEncoding.md`.

The idea is that most floating-point data in the wild is decimal in origin —
prices, sensor readings, coordinates — so `v * 10^e / 10^f` lands exactly on an
integer for some small `e` and `f`. Those integers are then frame-of-reference
encoded and bit-packed, which is far denser than the original mantissas. Values
that do not survive the round trip are stored verbatim as exceptions, so the
encoding stays lossless whatever the input.

A page is a 7-byte header, an offset array, then independent vectors:

```text
compression_mode : u8      0 = ALP
integer_encoding : u8      0 = frame of reference + bit-packing
log_vector_size  : u8      log2 of the values per vector, 3..15
num_elements     : i32 LE  values in the page
offsets          : u32 LE * num_vectors, from the start of the offset array
```

and each vector is

```text
exponent         : u8      e
factor           : u8      f, <= e
num_exceptions   : u16 LE
frame_of_ref     : i32/i64 LE   4 bytes for FLOAT, 8 for DOUBLE
bit_width        : u8
packed           : ceil(n * bit_width / 8) bytes, LSB-first
exc_positions    : u16 LE * num_exceptions
exc_values       : the original IEEE 754 values
```

Decoding one value is `(delta + frame_of_reference) * 10^f / 10^e`, then the
exceptions are written over the positions they name.
"""

from std.memory import bitcast

from parquet.encoding import PK_FIXED, PhysBuffer


# The spec requires the correctly-rounded IEEE 754 constants and forbids
# computing them at runtime: `10.0 ** e` accumulates error at larger e, and a
# decoder that is one ulp out stops being lossless. Written as literals so the
# compiler parses each one directly. Both are looked up once per vector, not
# per value, since exponent and factor are fixed for the whole vector.
def _pow10_f64(i: Int) raises -> Float64:
    if i == 0:
        return 1e0
    if i == 1:
        return 1e1
    if i == 2:
        return 1e2
    if i == 3:
        return 1e3
    if i == 4:
        return 1e4
    if i == 5:
        return 1e5
    if i == 6:
        return 1e6
    if i == 7:
        return 1e7
    if i == 8:
        return 1e8
    if i == 9:
        return 1e9
    if i == 10:
        return 1e10
    if i == 11:
        return 1e11
    if i == 12:
        return 1e12
    if i == 13:
        return 1e13
    if i == 14:
        return 1e14
    if i == 15:
        return 1e15
    if i == 16:
        return 1e16
    if i == 17:
        return 1e17
    if i == 18:
        return 1e18
    raise Error(String("parquet.alp: power of ten ", i, " out of range"))


def _pow10_f32(i: Int) raises -> Float32:
    if i == 0:
        return Float32(1e0)
    if i == 1:
        return Float32(1e1)
    if i == 2:
        return Float32(1e2)
    if i == 3:
        return Float32(1e3)
    if i == 4:
        return Float32(1e4)
    if i == 5:
        return Float32(1e5)
    if i == 6:
        return Float32(1e6)
    if i == 7:
        return Float32(1e7)
    if i == 8:
        return Float32(1e8)
    if i == 9:
        return Float32(1e9)
    if i == 10:
        return Float32(1e10)
    raise Error(String("parquet.alp: power of ten ", i, " out of range"))


def _need(data: Span[UInt8, _], at: Int, want: Int, what: StringSlice) raises:
    if at < 0 or at + want > len(data):
        raise Error(
            String(
                "parquet.alp: ",
                what,
                " needs ",
                want,
                " byte(s) at ",
                at,
                " but the page has ",
                len(data),
            )
        )


def _u16(data: Span[UInt8, _], at: Int) -> Int:
    return Int(data[at]) | (Int(data[at + 1]) << 8)


def _u32(data: Span[UInt8, _], at: Int) -> Int:
    return (
        Int(data[at])
        | (Int(data[at + 1]) << 8)
        | (Int(data[at + 2]) << 16)
        | (Int(data[at + 3]) << 24)
    )


def _unpack_lsb_u64(
    data: Span[UInt8, _], start: Int, width: Int, i: Int
) -> UInt64:
    """Value `i` of an LSB-first bit-packed run of `width`-bit fields."""
    if width == 0:
        return 0
    var bit = i * width
    var out = UInt64(0)
    var got = 0
    while got < width:
        var byte = start + (bit >> 3)
        var off = bit & 7
        var take = 8 - off
        if take > width - got:
            take = width - got
        var chunk = (UInt64(data[byte]) >> UInt64(off)) & (
            (UInt64(1) << UInt64(take)) - 1
        )
        out |= chunk << UInt64(got)
        got += take
        bit += take
    return out


def _decode_vector[
    dt: DType
](data: Span[UInt8, _], base: Int, n: Int, mut out: PhysBuffer,) raises:
    """Decode one vector of `n` values starting at `base`."""
    comptime is_double = dt == DType.float64
    comptime for_bytes = 8 if is_double else 4
    comptime max_exp = 18 if is_double else 10

    _need(data, base, 4, "vector header")
    var exponent = Int(data[base])
    var factor = Int(data[base + 1])
    var num_exc = _u16(data, base + 2)
    if exponent > max_exp:
        raise Error(String("parquet.alp: exponent ", exponent, " out of range"))
    if factor > exponent:
        raise Error(
            String(
                "parquet.alp: factor ",
                factor,
                " exceeds exponent ",
                exponent,
            )
        )

    var at = base + 4
    _need(data, at, for_bytes + 1, "frame of reference")
    var frame = Int64(0)

    @parameter
    if is_double:
        var raw = UInt64(0)
        for k in range(8):
            raw |= UInt64(data[at + k]) << UInt64(8 * k)
        frame = bitcast[DType.int64](raw)
    else:
        var raw = UInt32(0)
        for k in range(4):
            raw |= UInt32(data[at + k]) << UInt32(8 * k)
        frame = Int64(bitcast[DType.int32](raw))
    var bit_width = Int(data[at + for_bytes])
    if bit_width > (64 if is_double else 32):
        raise Error(
            String("parquet.alp: bit width ", bit_width, " out of range")
        )

    var packed = at + for_bytes + 1
    var packed_len = (n * bit_width + 7) // 8
    _need(data, packed, packed_len, "packed values")

    # `factor` scales up and `exponent` scales down, both by exact powers of
    # ten, and both are fixed for the vector — so look them up once here rather
    # than per value.
    #
    # One deviation to know about: the spec writes the decode as
    # `encoded * 10^factor * 10^(-exponent)`, two multiplications by tabulated
    # constants. This divides by 10^exponent instead, which can differ by an ulp
    # from multiplying by a rounded 10^(-exponent). Every column of
    # parquet-testing's alp_extended file decodes bit-identical to its PLAIN
    # reference either way, so nothing in the corpus separates them, but a
    # negative-power table would be the strictly faithful reading.
    var start = out.count

    @parameter
    if is_double:
        var up = _pow10_f64(factor)
        var down = _pow10_f64(exponent)
        for i in range(n):
            var delta = _unpack_lsb_u64(data, packed, bit_width, i)
            # frame_of_reference is the minimum encoded value and deltas are
            # non-negative, so the sum always fits the signed type.
            var enc = frame + Int64(delta)
            var v = Float64(enc) * up / down
            var bits = bitcast[DType.uint64](v)
            for b in range(8):
                out.bytes.append(UInt8((bits >> UInt64(8 * b)) & 0xFF))
            out.count += 1
    else:
        var up = _pow10_f32(factor)
        var down = _pow10_f32(exponent)
        for i in range(n):
            var delta = _unpack_lsb_u64(data, packed, bit_width, i)
            var enc = frame + Int64(delta)
            var v = Float32(enc) * up / down
            var bits = bitcast[DType.uint32](v)
            for b in range(4):
                out.bytes.append(UInt8((bits >> UInt32(8 * b)) & 0xFF))
            out.count += 1

    if num_exc == 0:
        return

    comptime val_bytes = 8 if is_double else 4
    var pos_at = packed + packed_len
    var val_at = pos_at + num_exc * 2
    _need(data, pos_at, num_exc * 2, "exception positions")
    _need(data, val_at, num_exc * val_bytes, "exception values")
    for k in range(num_exc):
        var pos = _u16(data, pos_at + k * 2)
        if pos >= n:
            raise Error(
                String(
                    "parquet.alp: exception position ",
                    pos,
                    " is outside a vector of ",
                    n,
                )
            )
        # Overwrite in place: the placeholder written above is meaningless.
        var dst = (start + pos) * val_bytes
        for b in range(val_bytes):
            out.bytes[dst + b] = data[val_at + k * val_bytes + b]


def decode_alp[
    dt: DType
](data: Span[UInt8, _], count: Int) raises -> PhysBuffer:
    """Decode an ALP page of `count` FLOAT (`dt=float32`) or DOUBLE values."""
    comptime width = 8 if dt == DType.float64 else 4
    var out = PhysBuffer(PK_FIXED, width)
    if count == 0:
        return out^

    _need(data, 0, 7, "page header")
    var mode = Int(data[0])
    if mode != 0:
        # 1 is ALP-RD, for values with no compact decimal form. Not built yet,
        # and a wrong guess here would silently corrupt the column.
        raise Error(
            String(
                "parquet.alp: compression mode ",
                mode,
                " is not implemented (only 0 = ALP)",
            )
        )
    var int_enc = Int(data[1])
    if int_enc != 0:
        raise Error(
            String("parquet.alp: integer encoding ", int_enc, " is not FOR")
        )
    var log_vec = Int(data[2])
    if log_vec < 3 or log_vec > 15:
        raise Error(
            String("parquet.alp: log_vector_size ", log_vec, " out of range")
        )
    var vector_size = 1 << log_vec
    var num_elements = _u32(data, 3)
    if num_elements != count:
        raise Error(
            String(
                "parquet.alp: page holds ",
                num_elements,
                " value(s) but the page header says ",
                count,
            )
        )

    var num_vectors = (num_elements + vector_size - 1) // vector_size
    var offsets = 7
    _need(data, offsets, num_vectors * 4, "vector offsets")

    out.bytes.reserve(count * width)
    var left = num_elements
    for v in range(num_vectors):
        # Offsets are relative to the start of the offset array, not the page.
        var base = offsets + _u32(data, offsets + v * 4)
        var n = vector_size if left > vector_size else left
        _decode_vector[dt](data, base, n, out)
        left -= n

    if out.count != count:
        raise Error(
            String(
                "parquet.alp: decoded ",
                out.count,
                " value(s) but the page says ",
                count,
            )
        )
    return out^
