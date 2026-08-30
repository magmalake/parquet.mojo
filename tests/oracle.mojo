"""Compare a `Table` we decoded against the values pyarrow reads.

`tools/oracle_pyarrow.py` writes one `<fixture>.parquet.oracle.json` per
fixture. Scalars in it are JSON *strings* so nothing is lost to JSON's number
model; nulls are JSON `null`, lists are arrays, structs are objects and maps
are arrays of `[key, value]` pairs. The rendering rules are:

| type | rendering |
|---|---|
| boolean | `true` / `false` |
| integers | decimal, signed or unsigned as the type says |
| floats | the IEEE-754 bits of the value widened to double, 16 hex digits |
| utf8 | the string |
| binary, fixed-size binary | lowercase hex |
| decimal | the unscaled 128-bit integer, decimal |
| date, time, timestamp | the integer the column stores, in its own unit |

`check_fixture` walks every value of every column and asserts equality. For a
fixture too big to spell out, the oracle carries a CRC32 over a canonical
serialisation of every value instead, and the first 200 rows explicitly.
"""

from avro.json import JsonDoc, parse_json
from hashes import crc32
from parquet import (
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
    AT_LARGE_BINARY,
    AT_LARGE_LIST,
    AT_LARGE_UTF8,
    AT_LIST,
    AT_MAP,
    AT_STRUCT,
    AT_TIME32,
    AT_TIME64,
    AT_TIMESTAMP,
    AT_UINT16,
    AT_UINT32,
    AT_UINT64,
    AT_UINT8,
    AT_UTF8,
    ArrayArena,
    ArrayData,
    ParquetReader,
    Table,
    bit_get,
)
from std.memory import bitcast
from std.testing import assert_equal, assert_true

comptime HEX = "0123456789abcdef"


def hex_of(data: Span[UInt8, _]) -> String:
    var out = String()
    for b in data:
        out += HEX[byte = Int(b >> 4)]
        out += HEX[byte = Int(b & 15)]
    return out^


def read_text(path: StringSlice) raises -> String:
    var f = open(String(path), "r")
    var text = f.read()
    f.close()
    return text^


def load_oracle(path: StringSlice) raises -> JsonDoc:
    return parse_json(read_text(path))


def _u128_dec(lo_in: UInt64, hi_in: UInt64) -> String:
    """Decimal digits of an unsigned 128-bit integer."""
    if lo_in == 0 and hi_in == 0:
        return String("0")
    var lo = lo_in
    var hi = hi_in
    var digits = String()
    while hi != 0 or lo != 0:
        # Divide the 128-bit value by 10, high half first.
        var q_hi = hi // 10
        var r_hi = hi % 10
        # (r_hi * 2^64 + lo) / 10, computed in two 32-bit halves to stay exact.
        var top = (r_hi << 32) | (lo >> 32)
        var q_top = top // 10
        var r_top = top % 10
        var bot = (r_top << 32) | (lo & 0xFFFFFFFF)
        var q_bot = bot // 10
        var r = bot % 10
        hi = q_hi
        lo = (q_top << 32) | q_bot
        digits += HEX[byte = Int(r)]
    var out = String()
    var n = digits.byte_length()
    for i in range(n):
        out += digits[byte = n - 1 - i]
    return out^


def decimal_string(bytes: Span[UInt8, _]) raises -> String:
    """A little-endian 16-byte two's-complement integer, in decimal."""
    if len(bytes) != 16:
        raise Error("oracle: decimal value is not 16 bytes")
    var lo: UInt64 = 0
    var hi: UInt64 = 0
    for k in range(8):
        lo |= UInt64(bytes[k]) << UInt64(8 * k)
    for k in range(8):
        hi |= UInt64(bytes[8 + k]) << UInt64(8 * k)
    var negative = (hi >> 63) == 1
    if negative:
        lo = ~lo
        hi = ~hi
        if lo == 0xFFFFFFFFFFFFFFFF:
            lo = 0
            hi += 1
        else:
            lo += 1
        return String("-", _u128_dec(lo, hi))
    return _u128_dec(lo, hi)


def double_bits(v: Float64) -> String:
    var u = bitcast[DType.uint64](v)
    var out = String()
    for k in range(8):
        var b = UInt8((u >> UInt64(8 * (7 - k))) & 0xFF)
        out += HEX[byte = Int(b >> 4)]
        out += HEX[byte = Int(b & 15)]
    return out^


def _load_le(a: ArrayData, i: Int, w: Int) -> UInt64:
    var u: UInt64 = 0
    for k in range(w):
        u |= UInt64(a.values[i * w + k]) << UInt64(8 * k)
    return u


def render_scalar(a: ArrayData, i: Int) raises -> String:
    """Render value `i` of a primitive array exactly as the oracle would."""
    var id = a.type.id
    if id == AT_BOOL:
        return String("true") if bit_get(Span(a.values), i) else String("false")
    if id == AT_UTF8 or id == AT_LARGE_UTF8:
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        return String(StringSlice(unsafe_from_utf8=Span(a.values)[lo:hi]))
    if id == AT_BINARY or id == AT_LARGE_BINARY:
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        return hex_of(Span(a.values)[lo:hi])
    if id == AT_FIXED_SIZE_BINARY:
        var w = a.type.byte_width
        return hex_of(Span(a.values)[i * w : (i + 1) * w])
    if id == AT_DECIMAL128:
        return decimal_string(Span(a.values)[i * 16 : (i + 1) * 16])
    if id == AT_FLOAT64:
        return double_bits(bitcast[DType.float64](_load_le(a, i, 8)))
    if id == AT_FLOAT32:
        return double_bits(Float64(bitcast[DType.float32](UInt32(_load_le(a, i, 4)))))
    if id == AT_FLOAT16:
        return double_bits(Float64(bitcast[DType.float16](UInt16(_load_le(a, i, 2)))))
    if id == AT_UINT8 or id == AT_UINT16 or id == AT_UINT32 or id == AT_UINT64:
        return String(_load_le(a, i, a.type.fixed_width()))
    var w = a.type.fixed_width()
    if w == 0:
        raise Error(String("oracle: cannot render type ", String(a.type)))
    var u = _load_le(a, i, w)
    if w < 8:
        var sign_bit = UInt64(1) << UInt64(8 * w - 1)
        if (u & sign_bit) != 0:
            u |= ~((UInt64(1) << UInt64(8 * w)) - 1)
    return String(bitcast[DType.int64](u))


def _child(arena: ArrayArena, a: ArrayData, k: Int) -> Int:
    return a.children[k]


def canon_value(
    arena: ArrayArena, node: Int, i: Int, mut out: String
) raises:
    """The canonical text `tools/oracle_pyarrow.py` hashes, for a digest check."""
    ref a = arena.nodes[node]
    if not bit_get(Span(a.validity), i):
        out += "N"
        return
    var id = a.type.id
    if id == AT_LIST or id == AT_MAP or id == AT_LARGE_LIST:
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        if id == AT_MAP:
            out += String("L", hi - lo, ":")
            var entries = a.children[0]
            var kf = arena.nodes[entries].children[0]
            var vf = arena.nodes[entries].children[1]
            for k in range(lo, hi):
                out += "L2:"
                canon_value(arena, kf, k, out)
                canon_value(arena, vf, k, out)
        else:
            out += String("L", hi - lo, ":")
            for k in range(lo, hi):
                canon_value(arena, a.children[0], k, out)
        return
    if id == AT_STRUCT:
        out += String("O", len(a.children), ":")
        for c in a.children:
            ref ca = arena.nodes[c]
            out += String("S", ca.name.byte_length(), ":", ca.name)
            canon_value(arena, c, i, out)
        return
    var s = render_scalar(a, i)
    out += String("S", s.byte_length(), ":", s)


def match_value(
    doc: JsonDoc, expect: Int, arena: ArrayArena, node: Int, i: Int, where: String
) raises:
    ref a = arena.nodes[node]
    var valid = bit_get(Span(a.validity), i)
    if doc.is_null(expect):
        if valid:
            raise Error(
                String(
                    where,
                    ": expected null, got ",
                    render_scalar(a, i) if not a.type.is_nested() else String("a value"),
                )
            )
        return
    if not valid:
        raise Error(String(where, ": expected a value, got null"))
    var id = a.type.id
    if id == AT_LIST or id == AT_LARGE_LIST:
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        assert_equal(doc.len_of(expect), hi - lo, String(where, ": list length"))
        for k in range(hi - lo):
            match_value(
                doc,
                doc.child(expect, k),
                arena,
                a.children[0],
                lo + k,
                String(where, "[", k, "]"),
            )
        return
    if id == AT_MAP:
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        assert_equal(doc.len_of(expect), hi - lo, String(where, ": map size"))
        var entries = a.children[0]
        var kf = arena.nodes[entries].children[0]
        var vf = arena.nodes[entries].children[1]
        for k in range(hi - lo):
            var pair = doc.child(expect, k)
            assert_equal(doc.len_of(pair), 2, String(where, ": map entry"))
            match_value(
                doc, doc.child(pair, 0), arena, kf, lo + k, String(where, ".key")
            )
            match_value(
                doc, doc.child(pair, 1), arena, vf, lo + k, String(where, ".value")
            )
        return
    if id == AT_STRUCT:
        for c in a.children:
            ref ca = arena.nodes[c]
            var kid = doc.get(expect, ca.name)
            if kid < 0:
                raise Error(String(where, ": oracle has no field '", ca.name, "'"))
            match_value(doc, kid, arena, c, i, String(where, ".", ca.name))
        return
    assert_equal(render_scalar(a, i), doc.as_string(expect), where)
