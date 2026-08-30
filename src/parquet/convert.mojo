"""Physical Parquet values to Arrow buffers.

A Parquet leaf's physical type and its Arrow type are often different widths:
an `INT32` annotated `Int(8, true)` is one byte in Arrow, a `DECIMAL` over
`FIXED_LEN_BYTE_ARRAY(7)` is a sign-extended little-endian 16-byte integer, and
`INT96` is a nanosecond timestamp built from a Julian day and a time of day.
`append_value` and `append_null` do that translation one value at a time,
straight into the destination `ArrayData`'s buffers.
"""

from parquet.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_DATE32,
    AT_DECIMAL128,
    AT_FIXED_SIZE_BINARY,
    AT_FLOAT16,
    AT_FLOAT32,
    AT_FLOAT64,
    AT_INT16,
    AT_INT32,
    AT_INT64,
    AT_INT8,
    AT_TIME32,
    AT_TIME64,
    AT_TIMESTAMP,
    AT_UINT16,
    AT_UINT32,
    AT_UINT64,
    AT_UINT8,
    AT_UTF8,
    ArrayData,
    bit_set,
)
from parquet.encoding import PK_BOOL, PK_FIXED, PK_VAR, PhysBuffer
from parquet.schema import LeafColumn
from std.memory import bitcast
from thrift import Type

# Days between the Julian epoch and the Unix epoch — INT96's day field is a
# Julian day number.
comptime JULIAN_UNIX_EPOCH = 2440588
comptime NANOS_PER_DAY = 86400000000000


def _ensure_offsets(mut out: ArrayData):
    if len(out.offsets) == 0:
        out.offsets.append(0)


def _push_zero(mut out: ArrayData, n: Int):
    for _ in range(n):
        out.values.append(0)


def _push_le(mut out: ArrayData, v: UInt64, n: Int):
    for k in range(n):
        out.values.append(UInt8((v >> UInt64(8 * k)) & 0xFF))


def _le_u64(src: Span[UInt8, _], n: Int) -> UInt64:
    var v: UInt64 = 0
    for k in range(n):
        v |= UInt64(src[k]) << UInt64(8 * k)
    return v


def _push_decimal_be(mut out: ArrayData, src: Span[UInt8, _]) raises:
    """A big-endian two's-complement integer of any width up to 16 bytes,
    stored as Arrow's little-endian 16-byte decimal."""
    var n = len(src)
    if n == 0 or n > 16:
        raise Error(
            String("parquet.convert: decimal value of ", n, " bytes (max 16)")
        )
    var negative = (src[0] & 0x80) != 0
    var fill = UInt8(0xFF) if negative else UInt8(0)
    for k in range(n):
        out.values.append(src[n - 1 - k])
    for _ in range(16 - n):
        out.values.append(fill)


def _push_decimal_int(mut out: ArrayData, v: Int64):
    var u = bitcast[DType.uint64](v)
    for k in range(8):
        out.values.append(UInt8((u >> UInt64(8 * k)) & 0xFF))
    var fill = UInt8(0xFF) if v < 0 else UInt8(0)
    for _ in range(8):
        out.values.append(fill)


def int96_to_nanos(src: Span[UInt8, _]) -> Int64:
    """`INT96`: 8 little-endian bytes of nanoseconds within the day, then a
    little-endian Julian day number."""
    var nanos = bitcast[DType.int64](_le_u64(src, 8))
    var day = bitcast[DType.int32](UInt32(_le_u64(src[8:], 4)))
    return (Int64(day) - Int64(JULIAN_UNIX_EPOCH)) * Int64(
        NANOS_PER_DAY
    ) + nanos


def append_null(mut out: ArrayData, leaf: LeafColumn) raises:
    """Append a null slot, keeping every buffer the right length.

    Validity is the caller's business — `parquet.assemble` maintains it, and
    only materialises a bitmap once a null actually turns up.
    """
    var i = out.length
    out.length += 1
    var id = out.type.id
    if id == AT_BOOL:
        bit_set(out.values, i, False)
        return
    if id == AT_UTF8 or id == AT_BINARY:
        _ensure_offsets(out)
        out.offsets.append(Int32(len(out.values)))
        return
    var w = out.type.fixed_width()
    if w == 0:
        raise Error(
            String(
                "parquet.convert: cannot append a null of type ",
                String(out.type),
            )
        )
    _push_zero(out, w)


def append_value(
    mut out: ArrayData, leaf: LeafColumn, vals: PhysBuffer, vi: Int
) raises:
    """Append value `vi` of `vals`, translated to `out`'s Arrow type."""
    var i = out.length
    out.length += 1
    var id = out.type.id
    var phys = leaf.physical

    if id == AT_BOOL:
        bit_set(out.values, i, vals.bool_at(vi))
        return

    if id == AT_UTF8 or id == AT_BINARY:
        _ensure_offsets(out)
        out.values.extend(vals.value_span(vi))
        out.offsets.append(Int32(len(out.values)))
        return

    if id == AT_DECIMAL128:
        if phys == Type.INT32.value:
            var raw = UInt32(_le_u64(vals.value_span(vi), 4))
            _push_decimal_int(out, Int64(bitcast[DType.int32](raw)))
            return
        if phys == Type.INT64.value:
            _push_decimal_int(
                out, bitcast[DType.int64](_le_u64(vals.value_span(vi), 8))
            )
            return
        _push_decimal_be(out, vals.value_span(vi))
        return

    if id == AT_TIMESTAMP and phys == Type.INT96.value:
        _push_le(
            out, bitcast[DType.uint64](int96_to_nanos(vals.value_span(vi))), 8
        )
        return

    var w = out.type.fixed_width()
    if w == 0:
        raise Error(
            String(
                "parquet.convert: cannot append a value of type ",
                String(out.type),
            )
        )
    var src = vals.value_span(vi)
    if len(src) == w:
        out.values.extend(src)
        return
    if len(src) > w:
        # int8/int16/uint8/uint16 are carried in a physical INT32.
        for k in range(w):
            out.values.append(src[k])
        return
    raise Error(
        String(
            "parquet.convert: physical value of ",
            len(src),
            " byte(s) is too narrow for ",
            String(out.type),
        )
    )
