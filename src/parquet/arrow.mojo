"""Arrow types and Arrow array data, in the layout the C Data Interface wants.

`ArrayData` is one Arrow array: a type, a length, an optional validity bitmap,
up to one offsets buffer, a values buffer, and children. Nesting is stored in
an *arena* (`ArrayArena`) with children referenced by index, because Mojo does
not allow a struct to contain a `List` of itself.

Buffer layouts follow the Arrow columnar spec exactly, so `parquet.carrow` can
hand the buffers to another runtime without touching them:

| type | buffers |
|---|---|
| null | — |
| boolean | validity, values (bit-packed, LSB first) |
| fixed width | validity, values |
| utf8 / binary | validity, offsets (i32), data |
| large utf8 / large binary | validity, offsets (i64), data |
| fixed-size binary, decimal128 | validity, values |
| list / map | validity, offsets (i32) + one child |
| large list | validity, offsets (i64) + one child |
| struct | validity + N children |
"""

from std.memory import bitcast

# ── Arrow type ids ─────────────────────────────────────────────────────────

comptime AT_NULL = 0
comptime AT_BOOL = 1
comptime AT_INT8 = 2
comptime AT_INT16 = 3
comptime AT_INT32 = 4
comptime AT_INT64 = 5
comptime AT_UINT8 = 6
comptime AT_UINT16 = 7
comptime AT_UINT32 = 8
comptime AT_UINT64 = 9
comptime AT_FLOAT16 = 10
comptime AT_FLOAT32 = 11
comptime AT_FLOAT64 = 12
comptime AT_UTF8 = 13
comptime AT_LARGE_UTF8 = 14
comptime AT_BINARY = 15
comptime AT_LARGE_BINARY = 16
comptime AT_FIXED_SIZE_BINARY = 17
comptime AT_DECIMAL128 = 18
comptime AT_DATE32 = 19
comptime AT_TIME32 = 20
comptime AT_TIME64 = 21
comptime AT_TIMESTAMP = 22
comptime AT_LIST = 23
comptime AT_LARGE_LIST = 24
comptime AT_STRUCT = 25
comptime AT_MAP = 26

# Time units.
comptime TU_SECOND = 0
comptime TU_MILLI = 1
comptime TU_MICRO = 2
comptime TU_NANO = 3


def unit_suffix(unit: Int) -> String:
    if unit == TU_SECOND:
        return String("s")
    if unit == TU_MILLI:
        return String("m")
    if unit == TU_MICRO:
        return String("u")
    return String("n")


def unit_name(unit: Int) -> String:
    if unit == TU_SECOND:
        return String("s")
    if unit == TU_MILLI:
        return String("ms")
    if unit == TU_MICRO:
        return String("us")
    return String("ns")


@fieldwise_init
struct ArrowType(Copyable, Movable, Writable):
    """One Arrow logical type. `id` is one of the `AT_*` constants."""

    var id: Int
    var byte_width: Int
    """Element width for `AT_FIXED_SIZE_BINARY` and `AT_DECIMAL128` (16)."""
    var precision: Int
    var scale: Int
    var unit: Int
    """Time unit for time/timestamp types, one of the `TU_*` constants."""
    var tz: String
    """`"UTC"` for a UTC-adjusted timestamp, empty for a local one."""
    var extension: String
    """An Arrow extension name such as `arrow.uuid`, or empty."""

    def __init__(out self, id: Int):
        self.id = id
        self.byte_width = 0
        self.precision = 0
        self.scale = 0
        self.unit = TU_MICRO
        self.tz = String()
        self.extension = String()

    def __init__(out self):
        self = ArrowType(AT_NULL)

    def is_nested(self) -> Bool:
        return (
            self.id == AT_LIST
            or self.id == AT_LARGE_LIST
            or self.id == AT_STRUCT
            or self.id == AT_MAP
        )

    def fixed_width(self) -> Int:
        """Bytes per element for a fixed-width type, or 0 if not fixed width."""
        if self.id == AT_INT8 or self.id == AT_UINT8:
            return 1
        if self.id == AT_INT16 or self.id == AT_UINT16 or self.id == AT_FLOAT16:
            return 2
        if (
            self.id == AT_INT32
            or self.id == AT_UINT32
            or self.id == AT_FLOAT32
            or self.id == AT_DATE32
            or self.id == AT_TIME32
        ):
            return 4
        if (
            self.id == AT_INT64
            or self.id == AT_UINT64
            or self.id == AT_FLOAT64
            or self.id == AT_TIME64
            or self.id == AT_TIMESTAMP
        ):
            return 8
        if self.id == AT_DECIMAL128:
            return 16
        if self.id == AT_FIXED_SIZE_BINARY:
            return self.byte_width
        return 0

    def format(self) -> String:
        """The Arrow C Data Interface format string for this type."""
        var i = self.id
        if i == AT_NULL:
            return String("n")
        if i == AT_BOOL:
            return String("b")
        if i == AT_INT8:
            return String("c")
        if i == AT_UINT8:
            return String("C")
        if i == AT_INT16:
            return String("s")
        if i == AT_UINT16:
            return String("S")
        if i == AT_INT32:
            return String("i")
        if i == AT_UINT32:
            return String("I")
        if i == AT_INT64:
            return String("l")
        if i == AT_UINT64:
            return String("L")
        if i == AT_FLOAT16:
            return String("e")
        if i == AT_FLOAT32:
            return String("f")
        if i == AT_FLOAT64:
            return String("g")
        if i == AT_UTF8:
            return String("u")
        if i == AT_LARGE_UTF8:
            return String("U")
        if i == AT_BINARY:
            return String("z")
        if i == AT_LARGE_BINARY:
            return String("Z")
        if i == AT_FIXED_SIZE_BINARY:
            return String("w:", self.byte_width)
        if i == AT_DECIMAL128:
            return String("d:", self.precision, ",", self.scale)
        if i == AT_DATE32:
            return String("tdD")
        if i == AT_TIME32:
            return String("tt", unit_suffix(self.unit))
        if i == AT_TIME64:
            return String("tt", unit_suffix(self.unit))
        if i == AT_TIMESTAMP:
            return String("ts", unit_suffix(self.unit), ":", self.tz)
        if i == AT_LIST:
            return String("+l")
        if i == AT_LARGE_LIST:
            return String("+L")
        if i == AT_STRUCT:
            return String("+s")
        if i == AT_MAP:
            return String("+m")
        return String("n")

    def write_to(self, mut writer: Some[Writer]):
        var i = self.id
        if i == AT_DECIMAL128:
            writer.write("decimal128(", self.precision, ", ", self.scale, ")")
        elif i == AT_FIXED_SIZE_BINARY:
            writer.write("fixed_size_binary[", self.byte_width, "]")
        elif i == AT_TIMESTAMP:
            writer.write("timestamp[", unit_name(self.unit))
            if self.tz:
                writer.write(", tz=", self.tz)
            writer.write("]")
        elif i == AT_TIME32 or i == AT_TIME64:
            writer.write("time", 32 if i == AT_TIME32 else 64)
            writer.write("[", unit_name(self.unit), "]")
        else:
            writer.write(_type_name(i))
        if self.extension:
            writer.write(" <", self.extension, ">")


def _type_name(i: Int) -> String:
    var names: List[String] = [
        String("null"),
        String("bool"),
        String("int8"),
        String("int16"),
        String("int32"),
        String("int64"),
        String("uint8"),
        String("uint16"),
        String("uint32"),
        String("uint64"),
        String("halffloat"),
        String("float"),
        String("double"),
        String("string"),
        String("large_string"),
        String("binary"),
        String("large_binary"),
        String("fixed_size_binary"),
        String("decimal128"),
        String("date32[day]"),
        String("time32"),
        String("time64"),
        String("timestamp"),
        String("list"),
        String("large_list"),
        String("struct"),
        String("map"),
    ]
    if i < 0 or i >= len(names):
        return String("?")
    return names[i].copy()


# ── constructors for the common types ──────────────────────────────────────


def at(id: Int) -> ArrowType:
    return ArrowType(id)


def at_fixed(width: Int) -> ArrowType:
    var t = ArrowType(AT_FIXED_SIZE_BINARY)
    t.byte_width = width
    return t^


def at_decimal(precision: Int, scale: Int) -> ArrowType:
    var t = ArrowType(AT_DECIMAL128)
    t.byte_width = 16
    t.precision = precision
    t.scale = scale
    return t^


def at_timestamp(unit: Int, var tz: String) -> ArrowType:
    var t = ArrowType(AT_TIMESTAMP)
    t.unit = unit
    t.tz = tz^
    return t^


def at_time(unit: Int) -> ArrowType:
    var t = ArrowType(AT_TIME32 if unit <= TU_MILLI else AT_TIME64)
    t.unit = unit
    return t^


# ── validity bitmaps ───────────────────────────────────────────────────────


def bitmap_bytes(n: Int) -> Int:
    return (n + 7) // 8


def bit_get(bitmap: Span[UInt8, _], i: Int) -> Bool:
    if len(bitmap) == 0:
        return True
    var byte = i // 8
    if byte >= len(bitmap):
        return False
    return ((bitmap[byte] >> UInt8(i % 8)) & 1) == 1


def bit_fill_valid(mut bitmap: List[UInt8], count: Int):
    """Materialise a bitmap in which the first `count` entries are all valid."""
    var whole = count // 8
    for _ in range(whole):
        bitmap.append(0xFF)
    if count % 8:
        bitmap.append((UInt8(1) << UInt8(count % 8)) - 1)


def bit_set(mut bitmap: List[UInt8], i: Int, value: Bool):
    var byte = i // 8
    while len(bitmap) <= byte:
        bitmap.append(0)
    if value:
        bitmap[byte] |= UInt8(1) << UInt8(i % 8)
    else:
        bitmap[byte] &= ~(UInt8(1) << UInt8(i % 8))


# ── array data ─────────────────────────────────────────────────────────────


struct ArrayData(Copyable, Movable):
    """One Arrow array. Children are indices into the owning `ArrayArena`."""

    var type: ArrowType
    var name: String
    var nullable: Bool
    var field_id: Int32
    var length: Int
    var null_count: Int
    var validity: List[UInt8]
    var offsets: List[Int32]
    var large_offsets: List[Int64]
    var values: List[UInt8]
    var children: List[Int]

    def __init__(out self, var type: ArrowType, var name: String):
        self.type = type^
        self.name = name^
        self.nullable = True
        self.field_id = -1
        self.length = 0
        self.null_count = 0
        self.validity = List[UInt8]()
        self.offsets = List[Int32]()
        self.large_offsets = List[Int64]()
        self.values = List[UInt8]()
        self.children = List[Int]()

    def __init__(out self):
        self = ArrayData(ArrowType(AT_NULL), String())

    def __init__(out self, *, copy: Self):
        self.type = copy.type.copy()
        self.name = copy.name.copy()
        self.nullable = copy.nullable
        self.field_id = copy.field_id
        self.length = copy.length
        self.null_count = copy.null_count
        self.validity = copy.validity.copy()
        self.offsets = copy.offsets.copy()
        self.large_offsets = copy.large_offsets.copy()
        self.values = copy.values.copy()
        self.children = copy.children.copy()

    def __init__(out self, *, deinit move: Self):
        self.type = move.type^
        self.name = move.name^
        self.nullable = move.nullable
        self.field_id = move.field_id
        self.length = move.length
        self.null_count = move.null_count
        self.validity = move.validity^
        self.offsets = move.offsets^
        self.large_offsets = move.large_offsets^
        self.values = move.values^
        self.children = move.children^

    def is_valid(self, i: Int) -> Bool:
        return bit_get(Span(self.validity), i)


struct ArrayArena(Copyable, Movable, Defaultable):
    """Owns every `ArrayData` of a batch; children are referenced by index."""

    var nodes: List[ArrayData]

    def __init__(out self):
        self.nodes = List[ArrayData]()

    def __init__(out self, *, copy: Self):
        self.nodes = copy.nodes.copy()

    def __init__(out self, *, deinit move: Self):
        self.nodes = move.nodes^

    def add(mut self, var node: ArrayData) -> Int:
        self.nodes.append(node^)
        return len(self.nodes) - 1

    def __len__(self) -> Int:
        return len(self.nodes)


# ── typed views over a values buffer ───────────────────────────────────────


def load_i32(buf: Span[UInt8, _], i: Int) -> Int32:
    var v: UInt32 = 0
    for k in range(4):
        v |= UInt32(buf[i * 4 + k]) << UInt32(8 * k)
    return bitcast[DType.int32](v)


def load_i64(buf: Span[UInt8, _], i: Int) -> Int64:
    var v: UInt64 = 0
    for k in range(8):
        v |= UInt64(buf[i * 8 + k]) << UInt64(8 * k)
    return bitcast[DType.int64](v)


def load_u64(buf: Span[UInt8, _], i: Int) -> UInt64:
    var v: UInt64 = 0
    for k in range(8):
        v |= UInt64(buf[i * 8 + k]) << UInt64(8 * k)
    return v


def load_f32(buf: Span[UInt8, _], i: Int) -> Float32:
    var v: UInt32 = 0
    for k in range(4):
        v |= UInt32(buf[i * 4 + k]) << UInt32(8 * k)
    return bitcast[DType.float32](v)


def load_f64(buf: Span[UInt8, _], i: Int) -> Float64:
    return bitcast[DType.float64](load_u64(buf, i))


def store_u32(mut buf: List[UInt8], v: UInt32):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))
    buf.append(UInt8((v >> 16) & 0xFF))
    buf.append(UInt8((v >> 24) & 0xFF))


def store_u64(mut buf: List[UInt8], v: UInt64):
    for k in range(8):
        buf.append(UInt8((v >> UInt64(8 * k)) & 0xFF))
