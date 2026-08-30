"""Column statistics, decoded from their raw bytes to typed values.

`Statistics.min_value` / `max_value` are the *physical* bytes of a value — four
little-endian bytes for an `INT32` column, the raw string for a `BYTE_ARRAY`
one — so every consumer has to decode them against the column's physical type.
`TypedStats` does that once and hands back `Int64`, `Float64`, `String` or raw
bytes, which is what `iceberg.mojo` wants for `lower_bounds` / `upper_bounds`.

The deprecated `min` / `max` fields are used only when `min_value` /
`max_value` are absent, and only for types whose old sort order was already
correct (everything except signed comparisons of `BYTE_ARRAY`, where
parquet-mr's historical order was wrong — such a column reports no bounds
rather than a wrong one).
"""

from parquet.arrow import AT_DECIMAL128, AT_UINT32, AT_UINT64, ArrowType
from parquet.convert import int96_to_nanos
from parquet.schema import LeafColumn
from std.memory import bitcast
from thrift import Statistics, Type

comptime SV_NONE = 0
comptime SV_INT = 1
comptime SV_UINT = 2
comptime SV_FLOAT = 3
comptime SV_BYTES = 4
comptime SV_BOOL = 5


struct ScalarValue(Copyable, Movable, Defaultable, Writable):
    """One decoded statistic, or one predicate literal."""

    var kind: Int
    var i: Int64
    var u: UInt64
    var f: Float64
    var b: List[UInt8]

    def __init__(out self):
        self.kind = SV_NONE
        self.i = 0
        self.u = 0
        self.f = 0.0
        self.b = List[UInt8]()

    def __init__(out self, *, copy: Self):
        self.kind = copy.kind
        self.i = copy.i
        self.u = copy.u
        self.f = copy.f
        self.b = copy.b.copy()

    def __init__(out self, *, deinit move: Self):
        self.kind = move.kind
        self.i = move.i
        self.u = move.u
        self.f = move.f
        self.b = move.b^

    @staticmethod
    def of_int(v: Int64) -> Self:
        var s = ScalarValue()
        s.kind = SV_INT
        s.i = v
        return s^

    @staticmethod
    def of_uint(v: UInt64) -> Self:
        var s = ScalarValue()
        s.kind = SV_UINT
        s.u = v
        return s^

    @staticmethod
    def of_float(v: Float64) -> Self:
        var s = ScalarValue()
        s.kind = SV_FLOAT
        s.f = v
        return s^

    @staticmethod
    def of_bool(v: Bool) -> Self:
        var s = ScalarValue()
        s.kind = SV_BOOL
        s.i = 1 if v else 0
        return s^

    @staticmethod
    def of_bytes(data: Span[UInt8, _]) -> Self:
        var s = ScalarValue()
        s.kind = SV_BYTES
        s.b.extend(data)
        return s^

    @staticmethod
    def of_string(text: StringSlice) -> Self:
        return ScalarValue.of_bytes(text.as_bytes())

    def as_string(self) -> String:
        return String(StringSlice(unsafe_from_utf8=Span(self.b)))

    def write_to(self, mut writer: Some[Writer]):
        if self.kind == SV_NONE:
            writer.write("none")
        elif self.kind == SV_INT:
            writer.write(self.i)
        elif self.kind == SV_UINT:
            writer.write(self.u)
        elif self.kind == SV_FLOAT:
            writer.write(self.f)
        elif self.kind == SV_BOOL:
            writer.write("true" if self.i else "false")
        else:
            writer.write(self.as_string())


def compare_bytes(a: Span[UInt8, _], b: Span[UInt8, _]) -> Int:
    var n = len(a) if len(a) < len(b) else len(b)
    for i in range(n):
        if a[i] < b[i]:
            return -1
        if a[i] > b[i]:
            return 1
    if len(a) < len(b):
        return -1
    if len(a) > len(b):
        return 1
    return 0


def compare_scalars(a: ScalarValue, b: ScalarValue) raises -> Int:
    """Order two scalars of the same kind. Raises on a kind mismatch."""
    if a.kind != b.kind:
        raise Error("parquet.stats: cannot compare values of different kinds")
    if a.kind == SV_INT or a.kind == SV_BOOL:
        return -1 if a.i < b.i else (1 if a.i > b.i else 0)
    if a.kind == SV_UINT:
        return -1 if a.u < b.u else (1 if a.u > b.u else 0)
    if a.kind == SV_FLOAT:
        return -1 if a.f < b.f else (1 if a.f > b.f else 0)
    return compare_bytes(Span(a.b), Span(b.b))


def decode_statistic(leaf: LeafColumn, raw: Span[UInt8, _]) raises -> ScalarValue:
    """Decode one raw statistic against the leaf's physical and Arrow types."""
    var phys = leaf.physical
    var id = leaf.arrow.id
    if phys == Type.BOOLEAN.value:
        if len(raw) < 1:
            raise Error("parquet.stats: empty BOOLEAN statistic")
        return ScalarValue.of_bool(raw[0] != 0)
    if phys == Type.INT32.value:
        if len(raw) < 4:
            raise Error("parquet.stats: short INT32 statistic")
        var u: UInt32 = 0
        for k in range(4):
            u |= UInt32(raw[k]) << UInt32(8 * k)
        if id == AT_DECIMAL128:
            return ScalarValue.of_int(Int64(bitcast[DType.int32](u)))
        if id == AT_UINT32:
            return ScalarValue.of_uint(UInt64(u))
        return ScalarValue.of_int(Int64(bitcast[DType.int32](u)))
    if phys == Type.INT64.value:
        if len(raw) < 8:
            raise Error("parquet.stats: short INT64 statistic")
        var u: UInt64 = 0
        for k in range(8):
            u |= UInt64(raw[k]) << UInt64(8 * k)
        if id == AT_UINT64:
            return ScalarValue.of_uint(u)
        return ScalarValue.of_int(bitcast[DType.int64](u))
    if phys == Type.INT96.value:
        if len(raw) < 12:
            raise Error("parquet.stats: short INT96 statistic")
        return ScalarValue.of_int(int96_to_nanos(raw))
    if phys == Type.FLOAT.value:
        if len(raw) < 4:
            raise Error("parquet.stats: short FLOAT statistic")
        var u: UInt32 = 0
        for k in range(4):
            u |= UInt32(raw[k]) << UInt32(8 * k)
        return ScalarValue.of_float(Float64(bitcast[DType.float32](u)))
    if phys == Type.DOUBLE.value:
        if len(raw) < 8:
            raise Error("parquet.stats: short DOUBLE statistic")
        var u: UInt64 = 0
        for k in range(8):
            u |= UInt64(raw[k]) << UInt64(8 * k)
        return ScalarValue.of_float(bitcast[DType.float64](u))
    return ScalarValue.of_bytes(raw)


struct TypedStats(Copyable, Movable, Defaultable):
    """One column chunk's statistics, decoded."""

    var has_min_max: Bool
    var min: ScalarValue
    var max: ScalarValue
    var has_null_count: Bool
    var null_count: Int64
    var has_distinct_count: Bool
    var distinct_count: Int64

    def __init__(out self):
        self.has_min_max = False
        self.min = ScalarValue()
        self.max = ScalarValue()
        self.has_null_count = False
        self.null_count = 0
        self.has_distinct_count = False
        self.distinct_count = 0

    def __init__(out self, *, copy: Self):
        self.has_min_max = copy.has_min_max
        self.min = copy.min.copy()
        self.max = copy.max.copy()
        self.has_null_count = copy.has_null_count
        self.null_count = copy.null_count
        self.has_distinct_count = copy.has_distinct_count
        self.distinct_count = copy.distinct_count

    def __init__(out self, *, deinit move: Self):
        self.has_min_max = move.has_min_max
        self.min = move.min^
        self.max = move.max^
        self.has_null_count = move.has_null_count
        self.null_count = move.null_count
        self.has_distinct_count = move.has_distinct_count
        self.distinct_count = move.distinct_count


def decode_stats(leaf: LeafColumn, st: Statistics) raises -> TypedStats:
    """Decode a `Statistics` struct against one leaf column."""
    var out = TypedStats()
    if st.null_count:
        out.has_null_count = True
        out.null_count = st.null_count.value()
    if st.distinct_count:
        out.has_distinct_count = True
        out.distinct_count = st.distinct_count.value()
    var lo = List[UInt8]()
    var hi = List[UInt8]()
    var have = False
    if st.min_value and st.max_value:
        lo = st.min_value.value().copy()
        hi = st.max_value.value().copy()
        have = True
    elif st.min and st.max:
        # The deprecated fields. parquet-mr wrote signed comparisons for
        # BYTE_ARRAY here, which is the wrong order for UTF8, so those are
        # ignored rather than trusted.
        if leaf.physical != Type.BYTE_ARRAY.value and (
            leaf.physical != Type.FIXED_LEN_BYTE_ARRAY.value
        ):
            lo = st.min.value().copy()
            hi = st.max.value().copy()
            have = True
    if have:
        out.min = decode_statistic(leaf, Span(lo))
        out.max = decode_statistic(leaf, Span(hi))
        out.has_min_max = True
    return out^
