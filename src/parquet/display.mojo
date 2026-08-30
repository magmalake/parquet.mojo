"""Human- and JSON-readable rendering of decoded Arrow arrays, for the CLI."""

from parquet.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_DATE32,
    AT_DECIMAL128,
    AT_FIXED_SIZE_BINARY,
    AT_FLOAT16,
    AT_FLOAT32,
    AT_FLOAT64,
    AT_LARGE_BINARY,
    AT_LARGE_LIST,
    AT_LARGE_UTF8,
    AT_LIST,
    AT_MAP,
    AT_STRUCT,
    AT_UINT16,
    AT_UINT32,
    AT_UINT64,
    AT_UINT8,
    AT_UTF8,
    ArrayArena,
    ArrayData,
    bit_get,
)
from std.memory import bitcast

comptime HEXDIGITS = "0123456789abcdef"


def hex_string(data: Span[UInt8, _]) -> String:
    var out = String()
    for b in data:
        out += HEXDIGITS[byte=Int(b >> 4)]
        out += HEXDIGITS[byte=Int(b & 15)]
    return out^


def json_escape(s: StringSlice) -> String:
    var out = String('"')
    for i in range(s.byte_length()):
        var c = s.as_bytes()[i]
        if c == 0x22:
            out += '\\"'
        elif c == 0x5C:
            out += "\\\\"
        elif c == 0x0A:
            out += "\\n"
        elif c == 0x0D:
            out += "\\r"
        elif c == 0x09:
            out += "\\t"
        elif c < 0x20:
            out += "\\u00"
            out += HEXDIGITS[byte=Int(c >> 4)]
            out += HEXDIGITS[byte=Int(c & 15)]
        else:
            out += String(StringSlice(unsafe_from_utf8=s.as_bytes()[i : i + 1]))
    out += '"'
    return out^


def _u128_decimal(lo_in: UInt64, hi_in: UInt64) -> String:
    if lo_in == 0 and hi_in == 0:
        return String("0")
    var lo = lo_in
    var hi = hi_in
    var digits = String()
    while hi != 0 or lo != 0:
        var q_hi = hi // 10
        var r_hi = hi % 10
        var top = (r_hi << 32) | (lo >> 32)
        var q_top = top // 10
        var bot = ((top % 10) << 32) | (lo & 0xFFFFFFFF)
        var q_bot = bot // 10
        var r = bot % 10
        hi = q_hi
        lo = (q_top << 32) | q_bot
        digits += HEXDIGITS[byte=Int(r)]
    var out = String()
    var n = digits.byte_length()
    for i in range(n):
        out += digits[byte=n - 1 - i]
    return out^


def decimal_text(bytes: Span[UInt8, _], scale: Int) raises -> String:
    """A 16-byte little-endian two's-complement decimal, with its point."""
    if len(bytes) != 16:
        raise Error("parquet.display: decimal value is not 16 bytes")
    var lo: UInt64 = 0
    var hi: UInt64 = 0
    for k in range(8):
        lo |= UInt64(bytes[k]) << UInt64(8 * k)
    for k in range(8):
        hi |= UInt64(bytes[8 + k]) << UInt64(8 * k)
    var sign = String()
    if (hi >> 63) == 1:
        lo = ~lo
        hi = ~hi
        if lo == 0xFFFFFFFFFFFFFFFF:
            lo = 0
            hi += 1
        else:
            lo += 1
        sign = String("-")
    var digits = _u128_decimal(lo, hi)
    if scale <= 0:
        return String(sign, digits)
    while digits.byte_length() <= scale:
        digits = String("0", digits)
    var n = digits.byte_length()
    var head = String()
    for i in range(n - scale):
        head += digits[byte=i]
    var tail = String()
    for i in range(n - scale, n):
        tail += digits[byte=i]
    return String(sign, head, ".", tail)


def _load_le(a: ArrayData, i: Int, w: Int) -> UInt64:
    var u: UInt64 = 0
    for k in range(w):
        u |= UInt64(a.values[i * w + k]) << UInt64(8 * k)
    return u


def scalar_json(a: ArrayData, i: Int) raises -> String:
    var id = a.type.id
    if id == AT_BOOL:
        return String("true") if bit_get(Span(a.values), i) else String("false")
    if id == AT_UTF8 or id == AT_LARGE_UTF8:
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        return json_escape(StringSlice(unsafe_from_utf8=Span(a.values)[lo:hi]))
    if id == AT_BINARY or id == AT_LARGE_BINARY:
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        return json_escape(hex_string(Span(a.values)[lo:hi]))
    if id == AT_FIXED_SIZE_BINARY:
        var w = a.type.byte_width
        return json_escape(hex_string(Span(a.values)[i * w : (i + 1) * w]))
    if id == AT_DECIMAL128:
        return json_escape(
            decimal_text(Span(a.values)[i * 16 : (i + 1) * 16], a.type.scale)
        )
    if id == AT_FLOAT64:
        return String(bitcast[DType.float64](_load_le(a, i, 8)))
    if id == AT_FLOAT32:
        return String(bitcast[DType.float32](UInt32(_load_le(a, i, 4))))
    if id == AT_FLOAT16:
        return String(bitcast[DType.float16](UInt16(_load_le(a, i, 2))))
    if id == AT_UINT8 or id == AT_UINT16 or id == AT_UINT32 or id == AT_UINT64:
        return String(_load_le(a, i, a.type.fixed_width()))
    var w = a.type.fixed_width()
    if w == 0:
        raise Error(String("parquet.display: cannot render ", String(a.type)))
    var u = _load_le(a, i, w)
    if w < 8:
        var sign_bit = UInt64(1) << UInt64(8 * w - 1)
        if (u & sign_bit) != 0:
            u |= ~((UInt64(1) << UInt64(8 * w)) - 1)
    return String(bitcast[DType.int64](u))


def value_json(arena: ArrayArena, node: Int, i: Int) raises -> String:
    """One value of one array, as a JSON fragment."""
    ref a = arena.nodes[node]
    if not bit_get(Span(a.validity), i):
        return String("null")
    var id = a.type.id
    if id == AT_LIST or id == AT_LARGE_LIST:
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        var out = String("[")
        for k in range(lo, hi):
            if k > lo:
                out += ","
            out += value_json(arena, a.children[0], k)
        out += "]"
        return out^
    if id == AT_MAP:
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        var entries = a.children[0]
        var kf = arena.nodes[entries].children[0]
        var vf = arena.nodes[entries].children[1]
        var out = String("{")
        for k in range(lo, hi):
            if k > lo:
                out += ","
            out += json_escape(value_text(arena, kf, k))
            out += ":"
            out += value_json(arena, vf, k)
        out += "}"
        return out^
    if id == AT_STRUCT:
        var out = String("{")
        for k in range(len(a.children)):
            if k:
                out += ","
            var c = a.children[k]
            out += json_escape(arena.nodes[c].name)
            out += ":"
            out += value_json(arena, c, i)
        out += "}"
        return out^
    return scalar_json(a, i)


def value_text(arena: ArrayArena, node: Int, i: Int) raises -> String:
    """One value, for the plain-text `cat` output."""
    ref a = arena.nodes[node]
    if not bit_get(Span(a.validity), i):
        return String("null")
    if a.type.is_nested():
        return value_json(arena, node, i)
    if a.type.id == AT_UTF8 or a.type.id == AT_LARGE_UTF8:
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        return String(StringSlice(unsafe_from_utf8=Span(a.values)[lo:hi]))
    return scalar_json(a, i)
