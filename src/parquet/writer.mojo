"""`ParquetWriter` — write Arrow arrays back out as a Parquet file.

The writer takes exactly what the reader produces: an `ArrayArena` and the
indices of the top-level arrays in it. That makes a round trip one line, and it
means anything the reader can decode, the writer can write back.

```mojo
from parquet import ParquetReader, ParquetWriter, WriterOptions

var r = ParquetReader.open("in.parquet")
var t = r.read_table()

var opts = WriterOptions()
opts.codec = CompressionCodec.SNAPPY.value
var w = ParquetWriter(opts^)
for b in t.batches:
    w.write_batch(b.arena, b.roots)
var bytes = w^.finish()
```

What it writes: Parquet 2.6 metadata with v1 data pages, `PLAIN` and
`RLE_DICTIONARY` values, `RLE` levels, three-level lists, `key_value` maps,
structs, field ids, per-chunk statistics with the sort order each logical type
requires, an `OffsetIndex` and a `ColumnIndex` per chunk, and key/value
metadata. Decimals go out as `FIXED_LEN_BYTE_ARRAY(16)`, which is legal for
every precision Arrow can hold.
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
    TU_MICRO,
    TU_MILLI,
    TU_NANO,
    ArrayArena,
    ArrayData,
    ArrowType,
    bit_get,
)
from parquet.codec import CodecSet, DefaultCodecs
from parquet.encoding import (
    PK_BOOL,
    PK_FIXED,
    PK_VAR,
    PhysBuffer,
    physical_kind,
    physical_width,
)
from parquet.rle_encode import encode_hybrid, encode_levels, write_uleb128
from parquet.stats import compare_bytes
from std.memory import bitcast
from thrift import (
    BoundaryOrder,
    ColumnChunk,
    ColumnIndex,
    ColumnMetaData,
    CompressionCodec,
    ConvertedType,
    DataPageHeader,
    DateType,
    DecimalType,
    DictionaryPageHeader,
    Encoding,
    FieldRepetitionType,
    FileMetaData,
    Float16Type,
    IntType,
    JsonType,
    KeyValue,
    ListType,
    LogicalType,
    MapType,
    MicroSeconds,
    MilliSeconds,
    NanoSeconds,
    OffsetIndex,
    PageHeader,
    TCompactProtocolWriter,
    PageLocation,
    PageType,
    RowGroup,
    SchemaElement,
    Statistics,
    StringType,
    TimeType,
    TimeUnit,
    TimestampType,
    Type,
    UUIDType,
    write_footer,
    write_footer_trailer,
)

comptime WN_PRIM = 0
comptime WN_LIST = 1
comptime WN_MAP = 2
comptime WN_STRUCT = 3

comptime CREATED_BY = "parquet.mojo version 0.1.0"


struct WriterOptions(Copyable, Movable, Defaultable):
    """How to write. Every field has a sensible default."""

    var codec: Int32
    var row_group_size: Int
    var data_page_size: Int
    """Target *uncompressed* bytes of values per data page."""
    var use_dictionary: Bool
    var write_statistics: Bool
    var write_page_index: Bool
    var created_by: String

    def __init__(out self):
        self.codec = CompressionCodec.SNAPPY.value
        self.row_group_size = 1 << 20
        self.data_page_size = 1 << 20
        self.use_dictionary = True
        self.write_statistics = True
        self.write_page_index = True
        self.created_by = String(CREATED_BY)

    def __init__(out self, *, copy: Self):
        self.codec = copy.codec
        self.row_group_size = copy.row_group_size
        self.data_page_size = copy.data_page_size
        self.use_dictionary = copy.use_dictionary
        self.write_statistics = copy.write_statistics
        self.write_page_index = copy.write_page_index
        self.created_by = copy.created_by.copy()

    def __init__(out self, *, deinit move: Self):
        self.codec = move.codec
        self.row_group_size = move.row_group_size
        self.data_page_size = move.data_page_size
        self.use_dictionary = move.use_dictionary
        self.write_statistics = move.write_statistics
        self.write_page_index = move.write_page_index
        self.created_by = move.created_by^


struct WNode(Copyable, Movable, Defaultable):
    """One node of the tree the writer walks, mirroring the Arrow field tree."""

    var name: String
    var kind: Int
    var arrow: ArrowType
    var nullable: Bool
    var field_id: Int32
    var children: List[Int]
    var def_level: Int
    var rep_level: Int
    var elem_def_level: Int
    var elem_rep_level: Int
    var leaf: Int

    def __init__(out self):
        self.name = String()
        self.kind = WN_PRIM
        self.arrow = ArrowType()
        self.nullable = True
        self.field_id = -1
        self.children = List[Int]()
        self.def_level = 0
        self.rep_level = 0
        self.elem_def_level = 0
        self.elem_rep_level = 0
        self.leaf = -1

    def __init__(out self, *, copy: Self):
        self.name = copy.name.copy()
        self.kind = copy.kind
        self.arrow = copy.arrow.copy()
        self.nullable = copy.nullable
        self.field_id = copy.field_id
        self.children = copy.children.copy()
        self.def_level = copy.def_level
        self.rep_level = copy.rep_level
        self.elem_def_level = copy.elem_def_level
        self.elem_rep_level = copy.elem_rep_level
        self.leaf = copy.leaf

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.kind = move.kind
        self.arrow = move.arrow^
        self.nullable = move.nullable
        self.field_id = move.field_id
        self.children = move.children^
        self.def_level = move.def_level
        self.rep_level = move.rep_level
        self.elem_def_level = move.elem_def_level
        self.elem_rep_level = move.elem_rep_level
        self.leaf = move.leaf


struct WLeaf(Copyable, Movable, Defaultable):
    """A column the writer will produce chunks for."""

    var path: List[String]
    var arrow: ArrowType
    var physical: Int32
    var type_length: Int
    var max_def: Int
    var max_rep: Int
    var field_id: Int32

    def __init__(out self):
        self.path = List[String]()
        self.arrow = ArrowType()
        self.physical = 0
        self.type_length = 0
        self.max_def = 0
        self.max_rep = 0
        self.field_id = -1

    def __init__(out self, *, copy: Self):
        self.path = copy.path.copy()
        self.arrow = copy.arrow.copy()
        self.physical = copy.physical
        self.type_length = copy.type_length
        self.max_def = copy.max_def
        self.max_rep = copy.max_rep
        self.field_id = copy.field_id

    def __init__(out self, *, deinit move: Self):
        self.path = move.path^
        self.arrow = move.arrow^
        self.physical = move.physical
        self.type_length = move.type_length
        self.max_def = move.max_def
        self.max_rep = move.max_rep
        self.field_id = move.field_id


def physical_for(t: ArrowType) raises -> Tuple[Int32, Int]:
    """The Parquet physical type and `type_length` for an Arrow type."""
    var i = t.id
    if i == AT_BOOL:
        return (Type.BOOLEAN.value, 0)
    if (
        i == AT_INT8
        or i == AT_INT16
        or i == AT_INT32
        or i == AT_UINT8
        or i == AT_UINT16
        or i == AT_UINT32
        or i == AT_DATE32
        or i == AT_TIME32
    ):
        return (Type.INT32.value, 0)
    if i == AT_INT64 or i == AT_UINT64 or i == AT_TIME64 or i == AT_TIMESTAMP:
        return (Type.INT64.value, 0)
    if i == AT_FLOAT32:
        return (Type.FLOAT.value, 0)
    if i == AT_FLOAT64:
        return (Type.DOUBLE.value, 0)
    if i == AT_FLOAT16:
        return (Type.FIXED_LEN_BYTE_ARRAY.value, 2)
    if i == AT_DECIMAL128:
        return (Type.FIXED_LEN_BYTE_ARRAY.value, 16)
    if i == AT_FIXED_SIZE_BINARY:
        return (Type.FIXED_LEN_BYTE_ARRAY.value, t.byte_width)
    if (
        i == AT_UTF8
        or i == AT_BINARY
        or i == AT_LARGE_UTF8
        or i == AT_LARGE_BINARY
    ):
        return (Type.BYTE_ARRAY.value, 0)
    raise Error(String("parquet.writer: cannot write Arrow type ", String(t)))


def _time_unit(unit: Int) -> TimeUnit:
    var u = TimeUnit()
    if unit == TU_MILLI:
        u.MILLIS = MilliSeconds()
    elif unit == TU_NANO:
        u.NANOS = NanoSeconds()
    else:
        u.MICROS = MicroSeconds()
    return u^


def annotate(mut el: SchemaElement, t: ArrowType) raises:
    """Give a `SchemaElement` the LogicalType and ConvertedType of `t`."""
    var i = t.id
    var lt = LogicalType()
    if i == AT_INT8 or i == AT_INT16 or i == AT_INT32 or i == AT_INT64:
        var bits = 8 * t.fixed_width()
        if bits == 32 and i == AT_INT32:
            return
        if bits == 64 and i == AT_INT64:
            return
        var it = IntType()
        it.bitWidth = Int8(bits)
        it.isSigned = True
        lt.INTEGER = it^
        el.converted_type = ConvertedType(
            ConvertedType.INT_8.value + Int32(0 if bits == 8 else (1 if bits == 16 else 2))
        )
    elif i == AT_UINT8 or i == AT_UINT16 or i == AT_UINT32 or i == AT_UINT64:
        var bits = 8 * t.fixed_width()
        var it = IntType()
        it.bitWidth = Int8(bits)
        it.isSigned = False
        lt.INTEGER = it^
        var step = 0 if bits == 8 else (1 if bits == 16 else (2 if bits == 32 else 3))
        el.converted_type = ConvertedType(ConvertedType.UINT_8.value + Int32(step))
    elif i == AT_UTF8 or i == AT_LARGE_UTF8:
        if t.extension == "arrow.json":
            lt.JSON = JsonType()
            el.converted_type = ConvertedType.JSON
        else:
            lt.STRING = StringType()
            el.converted_type = ConvertedType.UTF8
    elif i == AT_DECIMAL128:
        var dt = DecimalType()
        dt.scale = Int32(t.scale)
        dt.precision = Int32(t.precision)
        lt.DECIMAL = dt^
        el.converted_type = ConvertedType.DECIMAL
        el.precision = Int32(t.precision)
        el.scale = Int32(t.scale)
    elif i == AT_DATE32:
        lt.DATE = DateType()
        el.converted_type = ConvertedType.DATE
    elif i == AT_TIME32 or i == AT_TIME64:
        var tt = TimeType()
        tt.isAdjustedToUTC = False
        tt.unit = _time_unit(t.unit)
        lt.TIME = tt^
        if t.unit == TU_MILLI:
            el.converted_type = ConvertedType.TIME_MILLIS
        elif t.unit == TU_MICRO:
            el.converted_type = ConvertedType.TIME_MICROS
    elif i == AT_TIMESTAMP:
        var utc = t.tz == "UTC"
        var ts = TimestampType()
        ts.isAdjustedToUTC = utc
        ts.unit = _time_unit(t.unit)
        lt.TIMESTAMP = ts^
        if utc and t.unit == TU_MILLI:
            el.converted_type = ConvertedType.TIMESTAMP_MILLIS
        elif utc and t.unit == TU_MICRO:
            el.converted_type = ConvertedType.TIMESTAMP_MICROS
    elif i == AT_FLOAT16:
        lt.FLOAT16 = Float16Type()
    elif i == AT_FIXED_SIZE_BINARY and t.extension == "arrow.uuid":
        lt.UUID = UUIDType()
    else:
        return
    el.logicalType = lt^


struct WriteSchema(Copyable, Movable, Defaultable):
    """The writer's mirror of the Arrow field tree, plus the Parquet schema."""

    var nodes: List[WNode]
    var roots: List[Int]
    var leaves: List[WLeaf]
    var elements: List[SchemaElement]

    def __init__(out self):
        self.nodes = List[WNode]()
        self.roots = List[Int]()
        self.leaves = List[WLeaf]()
        self.elements = List[SchemaElement]()

    def __init__(out self, *, copy: Self):
        self.nodes = copy.nodes.copy()
        self.roots = copy.roots.copy()
        self.leaves = copy.leaves.copy()
        self.elements = copy.elements.copy()

    def __init__(out self, *, deinit move: Self):
        self.nodes = move.nodes^
        self.roots = move.roots^
        self.leaves = move.leaves^
        self.elements = move.elements^


def _elem(
    var name: String, rep: Int32, n_children: Int, field_id: Int32
) -> SchemaElement:
    var el = SchemaElement()
    el.name = name^
    el.repetition_type = FieldRepetitionType(rep)
    if n_children >= 0:
        el.num_children = Int32(n_children)
    if field_id >= 0:
        el.field_id = field_id
    return el^


def _build_node(
    mut s: WriteSchema,
    arena: ArrayArena,
    node: Int,
    var name: String,
    parent_def: Int,
    parent_rep: Int,
    mut path: List[String],
) raises -> Int:
    """Build the writer node for one Arrow array, appending its schema elements."""
    ref a = arena.nodes[node]
    var w = WNode()
    w.name = name.copy()
    w.arrow = a.type.copy()
    w.nullable = a.nullable
    w.field_id = a.field_id
    w.def_level = parent_def + (1 if a.nullable else 0)
    w.rep_level = parent_rep
    var rep = FieldRepetitionType.OPTIONAL.value if a.nullable else FieldRepetitionType.REQUIRED.value
    var id = a.type.id

    if id == AT_STRUCT:
        w.kind = WN_STRUCT
        s.elements.append(_elem(name^, rep, len(a.children), a.field_id))
        var me = len(s.nodes)
        s.nodes.append(w^)
        path.append(s.elements[len(s.elements) - 1].name.copy())
        for c in a.children:
            var kid = _build_node(
                s, arena, c, arena.nodes[c].name.copy(), w_def(s, me), parent_rep, path
            )
            s.nodes[me].children.append(kid)
        _ = path.pop()
        return me

    if id == AT_LIST or id == AT_LARGE_LIST or id == AT_MAP:
        var is_map = id == AT_MAP
        w.kind = WN_MAP if is_map else WN_LIST
        var group = _elem(name.copy(), rep, 1, a.field_id)
        var lt = LogicalType()
        if is_map:
            lt.MAP = MapType()
            group.converted_type = ConvertedType.MAP
        else:
            lt.LIST = ListType()
            group.converted_type = ConvertedType.LIST
        group.logicalType = lt^
        s.elements.append(group^)
        var me = len(s.nodes)
        w.elem_def_level = w.def_level + 1
        w.elem_rep_level = parent_rep + 1
        s.nodes.append(w^)
        path.append(name.copy())
        var inner_name = String("key_value") if is_map else String("list")
        var child = arena.nodes[node].children[0]
        var n_inner = 1
        if is_map:
            n_inner = len(arena.nodes[child].children)
        s.elements.append(
            _elem(inner_name.copy(), FieldRepetitionType.REPEATED.value, n_inner, -1)
        )
        path.append(inner_name^)
        var edef = s.nodes[me].elem_def_level
        var erep = s.nodes[me].elem_rep_level
        if is_map:
            # The repeated `key_value` group *is* the entries struct: its own
            # children go straight under it, so no extra level.
            var entries = child
            var kids = List[Int]()
            for c in arena.nodes[entries].children:
                kids.append(
                    _build_node(
                        s, arena, c, arena.nodes[c].name.copy(), edef, erep, path
                    )
                )
            var entry_node = WNode()
            entry_node.name = String("key_value")
            entry_node.kind = WN_STRUCT
            entry_node.arrow = arena.nodes[entries].type.copy()
            entry_node.nullable = False
            entry_node.def_level = edef
            entry_node.rep_level = erep
            entry_node.children = kids^
            s.nodes.append(entry_node^)
            s.nodes[me].children.append(len(s.nodes) - 1)
        else:
            var kid = _build_node(
                s, arena, child, String("element"), edef, erep, path
            )
            s.nodes[me].children.append(kid)
        _ = path.pop()
        _ = path.pop()
        return me

    # A primitive leaf.
    var phys = physical_for(a.type)
    w.kind = WN_PRIM
    w.leaf = len(s.leaves)
    var el = _elem(name.copy(), rep, -1, a.field_id)
    el.type_ = Type(phys[0])
    if phys[1] > 0:
        el.type_length = Int32(phys[1])
    annotate(el, a.type)
    s.elements.append(el^)
    var lc = WLeaf()
    path.append(name^)
    lc.path = path.copy()
    _ = path.pop()
    lc.arrow = a.type.copy()
    lc.physical = phys[0]
    lc.type_length = phys[1]
    lc.max_def = w.def_level
    lc.max_rep = w.rep_level
    lc.field_id = a.field_id
    s.leaves.append(lc^)
    var me = len(s.nodes)
    s.nodes.append(w^)
    return me


def w_def(s: WriteSchema, node: Int) -> Int:
    return s.nodes[node].def_level


def build_write_schema(arena: ArrayArena, roots: List[Int]) raises -> WriteSchema:
    """Derive the Parquet schema from the Arrow arrays that will be written."""
    var s = WriteSchema()
    s.elements.append(_elem(String("schema"), FieldRepetitionType.REQUIRED.value, len(roots), -1))
    var path = List[String]()
    for r in roots:
        s.roots.append(
            _build_node(s, arena, r, arena.nodes[r].name.copy(), 0, 0, path)
        )
    return s^


# ── shredding: Arrow arrays back to levels and values ──────────────────────


struct LeafBuffer(Copyable, Movable, Defaultable):
    var defs: List[UInt16]
    var reps: List[UInt16]
    var values: PhysBuffer

    def __init__(out self):
        self.defs = List[UInt16]()
        self.reps = List[UInt16]()
        self.values = PhysBuffer()

    def __init__(out self, *, copy: Self):
        self.defs = copy.defs.copy()
        self.reps = copy.reps.copy()
        self.values = copy.values.copy()

    def __init__(out self, *, deinit move: Self):
        self.defs = move.defs^
        self.reps = move.reps^
        self.values = move.values^


def _put_le(mut out: List[UInt8], v: UInt64, n: Int):
    for k in range(n):
        out.append(UInt8((v >> UInt64(8 * k)) & 0xFF))


def _read_le(buf: List[UInt8], at: Int, n: Int) -> UInt64:
    var v: UInt64 = 0
    for k in range(n):
        v |= UInt64(buf[at + k]) << UInt64(8 * k)
    return v


def append_physical(
    mut out: PhysBuffer, leaf: WLeaf, a: ArrayData, i: Int
) raises:
    """Append value `i` of `a` in the leaf's physical Parquet representation."""
    var id = a.type.id
    if id == AT_BOOL:
        out.append_bool(bit_get(Span(a.values), i))
        return
    if (
        id == AT_UTF8
        or id == AT_BINARY
        or id == AT_LARGE_UTF8
        or id == AT_LARGE_BINARY
    ):
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        out.append_bytes(Span(a.values)[lo:hi])
        return
    if id == AT_DECIMAL128:
        # Arrow stores decimals little-endian; Parquet's FLBA wants big-endian.
        for k in range(16):
            out.bytes.append(a.values[i * 16 + 15 - k])
        out.count += 1
        return
    var w = a.type.fixed_width()
    if leaf.physical == Type.FIXED_LEN_BYTE_ARRAY.value or leaf.physical == Type.FLOAT.value or leaf.physical == Type.DOUBLE.value:
        out.bytes.extend(Span(a.values)[i * w : (i + 1) * w])
        out.count += 1
        return
    var target = 4 if leaf.physical == Type.INT32.value else 8
    var u = _read_le(a.values, i * w, w)
    var signed = not (
        id == AT_UINT8 or id == AT_UINT16 or id == AT_UINT32 or id == AT_UINT64
    )
    if signed and w < target:
        var sign_bit = UInt64(1) << UInt64(8 * w - 1)
        if (u & sign_bit) != 0:
            u |= ~((UInt64(1) << UInt64(8 * w)) - 1)
    _put_le(out.bytes, u, target)
    out.count += 1


def _emit_missing(
    s: WriteSchema, w: Int, def_level: Int, rep: Int, mut cols: List[LeafBuffer]
):
    """Record one absent value for every leaf under writer node `w`."""
    ref node = s.nodes[w]
    if node.kind == WN_PRIM:
        cols[node.leaf].defs.append(UInt16(def_level))
        cols[node.leaf].reps.append(UInt16(rep))
        return
    for c in node.children:
        _emit_missing(s, c, def_level, rep, cols)


def shred(
    s: WriteSchema,
    w: Int,
    arena: ArrayArena,
    node: Int,
    i: Int,
    parent_def: Int,
    rep: Int,
    mut cols: List[LeafBuffer],
) raises:
    """Write one value of one Arrow array into the leaf buffers under it."""
    ref a = arena.nodes[node]
    if a.nullable and not bit_get(Span(a.validity), i):
        _emit_missing(s, w, parent_def, rep, cols)
        return
    var d = s.nodes[w].def_level
    var kind = s.nodes[w].kind
    if kind == WN_PRIM:
        var leaf = s.nodes[w].leaf
        cols[leaf].defs.append(UInt16(d))
        cols[leaf].reps.append(UInt16(rep))
        append_physical(cols[leaf].values, s.leaves[leaf], a, i)
        return
    if kind == WN_STRUCT:
        for k in range(len(a.children)):
            shred(s, s.nodes[w].children[k], arena, a.children[k], i, d, rep, cols)
        return
    # A list or a map.
    var lo = Int(a.offsets[i])
    var hi = Int(a.offsets[i + 1])
    if lo == hi:
        _emit_missing(s, w, d, rep, cols)
        return
    var child_w = s.nodes[w].children[0]
    var child_node = a.children[0]
    var edef = s.nodes[w].elem_def_level
    var erep = s.nodes[w].elem_rep_level
    for k in range(lo, hi):
        shred(
            s,
            child_w,
            arena,
            child_node,
            k,
            edef,
            rep if k == lo else erep,
            cols,
        )


# ── statistics ─────────────────────────────────────────────────────────────


def _stat_less(leaf: WLeaf, a: Span[UInt8, _], b: Span[UInt8, _]) raises -> Bool:
    """Is `a` below `b` in the sort order this column's type requires?"""
    var phys = leaf.physical
    var id = leaf.arrow.id
    if phys == Type.BOOLEAN.value:
        return a[0] < b[0]
    if phys == Type.INT32.value:
        var ua = UInt32(_read_le_span(a, 0, 4))
        var ub = UInt32(_read_le_span(b, 0, 4))
        if id == AT_UINT8 or id == AT_UINT16 or id == AT_UINT32:
            return ua < ub
        return bitcast[DType.int32](ua) < bitcast[DType.int32](ub)
    if phys == Type.INT64.value:
        var ua = _read_le_span(a, 0, 8)
        var ub = _read_le_span(b, 0, 8)
        if id == AT_UINT64:
            return ua < ub
        return bitcast[DType.int64](ua) < bitcast[DType.int64](ub)
    if phys == Type.FLOAT.value:
        return bitcast[DType.float32](UInt32(_read_le_span(a, 0, 4))) < bitcast[
            DType.float32
        ](UInt32(_read_le_span(b, 0, 4)))
    if phys == Type.DOUBLE.value:
        return bitcast[DType.float64](_read_le_span(a, 0, 8)) < bitcast[
            DType.float64
        ](_read_le_span(b, 0, 8))
    if id == AT_DECIMAL128:
        # Big-endian two's complement: the sign bit decides first.
        var na = (a[0] & 0x80) != 0
        var nb = (b[0] & 0x80) != 0
        if na != nb:
            return na
        return compare_bytes(a, b) < 0
    return compare_bytes(a, b) < 0


def _read_le_span(s: Span[UInt8, _], at: Int, n: Int) -> UInt64:
    var v: UInt64 = 0
    for k in range(n):
        v |= UInt64(s[at + k]) << UInt64(8 * k)
    return v


def _is_nan(leaf: WLeaf, v: Span[UInt8, _]) -> Bool:
    if leaf.physical == Type.FLOAT.value:
        var f = bitcast[DType.float32](UInt32(_read_le_span(v, 0, 4)))
        return f != f
    if leaf.physical == Type.DOUBLE.value:
        var d = bitcast[DType.float64](_read_le_span(v, 0, 8))
        return d != d
    return False


def min_max(
    leaf: WLeaf, values: PhysBuffer, start: Int, count: Int
) raises -> Tuple[List[UInt8], List[UInt8], Bool]:
    """The bounds of `count` values from `start`, in this column's sort order."""
    var mn = List[UInt8]()
    var mx = List[UInt8]()
    if count == 0:
        return (mn^, mx^, False)
    if values.kind == PK_BOOL:
        var seen_false = False
        var seen_true = False
        for k in range(count):
            if values.bool_at(start + k):
                seen_true = True
            else:
                seen_false = True
        mn.append(UInt8(0) if seen_false else UInt8(1))
        mx.append(UInt8(1) if seen_true else UInt8(0))
        return (mn^, mx^, True)
    var lo = -1
    var hi = -1
    for k in range(count):
        var i = start + k
        var v = values.value_span(i)
        if _is_nan(leaf, v):
            continue
        if lo < 0:
            lo = i
            hi = i
            continue
        if _stat_less(leaf, v, values.value_span(lo)):
            lo = i
        if _stat_less(leaf, values.value_span(hi), v):
            hi = i
    if lo < 0:
        return (mn^, mx^, False)
    mn.extend(values.value_span(lo))
    mx.extend(values.value_span(hi))
    return (mn^, mx^, True)


def column_statistics(
    leaf: WLeaf, buf: LeafBuffer, null_count: Int
) raises -> Statistics:
    var st = Statistics()
    st.null_count = Int64(null_count)
    var got = min_max(leaf, buf.values, 0, buf.values.count)
    if got[2]:
        st.min_value = got[0].copy()
        st.max_value = got[1].copy()
        st.is_min_value_exact = True
        st.is_max_value_exact = True
    return st^


# ── dictionaries ───────────────────────────────────────────────────────────


def _hash_bytes(v: Span[UInt8, _]) -> UInt64:
    var h: UInt64 = 0xCBF29CE484222325
    for b in v:
        h = (h ^ UInt64(b)) * 0x100000001B3
    return h


struct DictBuilder(Movable, Defaultable):
    var values: PhysBuffer
    var buckets: Dict[UInt64, List[Int]]

    def __init__(out self):
        self.values = PhysBuffer()
        self.buckets = Dict[UInt64, List[Int]]()

    def __init__(out self, kind: Int, width: Int):
        self.values = PhysBuffer(kind, width)
        self.buckets = Dict[UInt64, List[Int]]()

    def __init__(out self, *, deinit move: Self):
        self.values = move.values^
        self.buckets = move.buckets^

    def index_of(mut self, v: Span[UInt8, _]) raises -> Int:
        var h = _hash_bytes(v)
        if h in self.buckets:
            ref bucket = self.buckets[h]
            for k in bucket:
                if compare_bytes(self.values.value_span(k), v) == 0:
                    return k
        var idx = self.values.count
        if self.values.kind == PK_VAR:
            self.values.append_bytes(v)
        else:
            self.values.bytes.extend(v)
            self.values.count += 1
        if h in self.buckets:
            self.buckets[h].append(idx)
        else:
            var fresh: List[Int] = [idx]
            self.buckets[h] = fresh^
        return idx


# ── the writer ─────────────────────────────────────────────────────────────


struct ParquetWriter[Codecs: CodecSet = DefaultCodecs](Movable):
    """Writes Arrow arrays out as Parquet. One `write_batch` per record batch."""

    var options: WriterOptions
    var schema: WriteSchema
    var out: List[UInt8]
    var row_groups: List[RowGroup]
    var offset_indexes: List[List[OffsetIndex]]
    var column_indexes: List[List[ColumnIndex]]
    var num_rows: Int64
    var kv: List[KeyValue]
    var _started: Bool

    def __init__(out self, var options: WriterOptions):
        self.options = options^
        self.schema = WriteSchema()
        self.out = List[UInt8]()
        self.row_groups = List[RowGroup]()
        self.offset_indexes = List[List[OffsetIndex]]()
        self.column_indexes = List[List[ColumnIndex]]()
        self.num_rows = 0
        self.kv = List[KeyValue]()
        self._started = False

    def __init__(out self, *, deinit move: Self):
        self.options = move.options^
        self.schema = move.schema^
        self.out = move.out^
        self.row_groups = move.row_groups^
        self.offset_indexes = move.offset_indexes^
        self.column_indexes = move.column_indexes^
        self.num_rows = move.num_rows
        self.kv = move.kv^
        self._started = move._started

    def add_metadata(mut self, key: StringSlice, value: StringSlice):
        var e = KeyValue()
        e.key = String(key)
        e.value = String(value)
        self.kv.append(e^)

    def write_batch(mut self, arena: ArrayArena, roots: List[Int]) raises:
        """Write one record batch, split into row groups of `row_group_size`."""
        if len(roots) == 0:
            raise Error("parquet.writer: a batch needs at least one column")
        if not self._started:
            self.schema = build_write_schema(arena, roots)
            self.out.extend(StringSlice("PAR1").as_bytes())
            self._started = True
        var rows = arena.nodes[roots[0]].length
        for r in roots:
            if arena.nodes[r].length != rows:
                raise Error(
                    String(
                        "parquet.writer: column '",
                        arena.nodes[r].name,
                        "' has ",
                        arena.nodes[r].length,
                        " rows but the batch has ",
                        rows,
                    )
                )
        if rows == 0:
            # Still emit one (empty) row group, so the file keeps its columns.
            self._write_row_group(arena, roots, 0, 0)
            return
        var start = 0
        while start < rows:
            var end = start + self.options.row_group_size
            if end > rows:
                end = rows
            self._write_row_group(arena, roots, start, end)
            start = end

    def _write_row_group(
        mut self, arena: ArrayArena, roots: List[Int], r0: Int, r1: Int
    ) raises:
        var n_leaves = len(self.schema.leaves)
        var cols = List[LeafBuffer]()
        for i in range(n_leaves):
            var b = LeafBuffer()
            b.values = PhysBuffer(
                physical_kind(self.schema.leaves[i].physical),
                physical_width(
                    self.schema.leaves[i].physical,
                    self.schema.leaves[i].type_length,
                ),
            )
            cols.append(b^)
        for k in range(len(roots)):
            for i in range(r0, r1):
                shred(
                    self.schema,
                    self.schema.roots[k],
                    arena,
                    roots[k],
                    i,
                    0,
                    0,
                    cols,
                )
        var rg = RowGroup()
        rg.num_rows = Int64(r1 - r0)
        rg.file_offset = Int64(len(self.out))
        var total_size: Int64 = 0
        var offs = List[OffsetIndex]()
        var cidx = List[ColumnIndex]()
        for i in range(n_leaves):
            var got = self._write_chunk(i, cols[i])
            var chunk = got[0].copy()
            total_size += chunk.meta_data.value().total_uncompressed_size
            rg.columns.append(chunk^)
            offs.append(got[1].copy())
            cidx.append(got[2].copy())
        rg.total_byte_size = total_size
        rg.ordinal = Int16(len(self.row_groups))
        self.row_groups.append(rg^)
        self.offset_indexes.append(offs^)
        self.column_indexes.append(cidx^)
        self.num_rows += Int64(r1 - r0)

    def _write_chunk(
        mut self, leaf_index: Int, buf: LeafBuffer
    ) raises -> Tuple[ColumnChunk, OffsetIndex, ColumnIndex]:
        ref leaf = self.schema.leaves[leaf_index]
        var codec = self.options.codec
        var chunk_start = len(self.out)
        var n_slots = len(buf.defs)
        var n_values = buf.values.count
        var null_count = n_slots - n_values

        # Dictionary, if it pays for itself.
        var use_dict = (
            self.options.use_dictionary
            and buf.values.kind != PK_BOOL
            and n_values > 0
        )
        var indices = List[UInt16]()
        var dict = PhysBuffer()
        var dict_offset: Int64 = 0
        var dict_uncompressed: Int64 = 0
        var dict_compressed: Int64 = 0
        if use_dict:
            var builder = DictBuilder(buf.values.kind, buf.values.width)
            var raw = List[Int]()
            for i in range(n_values):
                raw.append(builder.index_of(buf.values.value_span(i)))
            dict = builder.values.copy()
            if dict.count > 65535 or (dict.count * 2 > n_values and dict.count > 64):
                use_dict = False
            else:
                for v in raw:
                    indices.append(UInt16(v))
        var encoding = (
            Encoding.RLE_DICTIONARY.value if use_dict else Encoding.PLAIN.value
        )

        var encodings = List[Encoding]()
        encodings.append(Encoding.RLE)
        encodings.append(Encoding(encoding))
        if use_dict:
            encodings.append(Encoding.PLAIN)
            var body = _plain_bytes(leaf, dict)
            dict_offset = Int64(len(self.out))
            var compressed = Self.Codecs.compress(codec, Span(body))
            var ph = PageHeader()
            ph.type_ = PageType.DICTIONARY_PAGE
            ph.uncompressed_page_size = Int32(len(body))
            ph.compressed_page_size = Int32(len(compressed))
            var dh = DictionaryPageHeader()
            dh.num_values = Int32(dict.count)
            dh.encoding = Encoding.PLAIN
            ph.dictionary_page_header = dh^
            var hlen = _write_page(self.out, ph^, Span(compressed))
            dict_uncompressed = Int64(len(body))
            dict_compressed = Int64(hlen + len(compressed))

        # Data pages, split at record boundaries by target page size.
        var data_offset = Int64(len(self.out))
        var starts = List[Int]()
        for i in range(n_slots):
            if leaf.max_rep == 0 or buf.reps[i] == 0:
                starts.append(i)
        starts.append(n_slots)
        var oi = OffsetIndex()
        var ci = ColumnIndex()
        var page_nulls = List[Int64]()
        var total_uncompressed: Int64 = 0
        var total_compressed: Int64 = 0
        var value_width = leaf.type_length if leaf.type_length > 0 else 8
        var rec = 0
        var first_row: Int64 = 0
        var value_cursor = 0
        while rec < len(starts) - 1:
            var rec_end = rec
            var bytes_so_far = 0
            while rec_end < len(starts) - 1:
                var slots = starts[rec_end + 1] - starts[rec_end]
                bytes_so_far += slots * value_width
                rec_end += 1
                if bytes_so_far >= self.options.data_page_size:
                    break
            var s0 = starts[rec]
            var s1 = starts[rec_end]
            var page_values = 0
            for i in range(s0, s1):
                if Int(buf.defs[i]) == leaf.max_def:
                    page_values += 1
            var body = List[UInt8]()
            if leaf.max_rep > 0:
                var reps = List[UInt16]()
                for i in range(s0, s1):
                    reps.append(buf.reps[i])
                body.extend(Span(encode_levels(reps, _bit_width(leaf.max_rep))))
            if leaf.max_def > 0:
                var defs = List[UInt16]()
                for i in range(s0, s1):
                    defs.append(buf.defs[i])
                body.extend(Span(encode_levels(defs, _bit_width(leaf.max_def))))
            if use_dict:
                var page_idx = List[UInt16]()
                for k in range(value_cursor, value_cursor + page_values):
                    page_idx.append(indices[k])
                var width = _bit_width(dict.count - 1)
                body.append(UInt8(width))
                body.extend(Span(encode_hybrid(page_idx, width)))
            else:
                body.extend(Span(_plain_slice(leaf, buf.values, value_cursor, page_values)))
            value_cursor += page_values
            var compressed = Self.Codecs.compress(codec, Span(body))
            var ph = PageHeader()
            ph.type_ = PageType.DATA_PAGE
            ph.uncompressed_page_size = Int32(len(body))
            ph.compressed_page_size = Int32(len(compressed))
            var dph = DataPageHeader()
            dph.num_values = Int32(s1 - s0)
            dph.encoding = Encoding(encoding)
            dph.definition_level_encoding = Encoding.RLE
            dph.repetition_level_encoding = Encoding.RLE
            ph.data_page_header = dph^
            var page_at = Int64(len(self.out))
            var header_len = _write_page(self.out, ph^, Span(compressed))
            var loc = PageLocation()
            loc.offset = page_at
            loc.compressed_page_size = Int32(header_len + len(compressed))
            loc.first_row_index = first_row
            oi.page_locations.append(loc^)
            var bounds = min_max(leaf, buf.values, value_cursor - page_values, page_values)
            ci.null_pages.append(not bounds[2])
            ci.min_values.append(bounds[0].copy())
            ci.max_values.append(bounds[1].copy())
            page_nulls.append(Int64(s1 - s0 - page_values))
            total_uncompressed += Int64(len(body))
            total_compressed += Int64(header_len + len(compressed))
            first_row += Int64(rec_end - rec)
            rec = rec_end
        ci.boundary_order = BoundaryOrder.UNORDERED
        ci.null_counts = page_nulls^

        var cm = ColumnMetaData()
        cm.type_ = Type(leaf.physical)
        cm.encodings = encodings^
        cm.path_in_schema = leaf.path.copy()
        cm.codec = CompressionCodec(codec)
        cm.num_values = Int64(n_slots)
        cm.data_page_offset = data_offset
        if use_dict:
            cm.dictionary_page_offset = dict_offset
            total_compressed += dict_compressed
            total_uncompressed += dict_uncompressed
        cm.total_uncompressed_size = total_uncompressed
        cm.total_compressed_size = total_compressed
        if self.options.write_statistics:
            cm.statistics = column_statistics(leaf, buf, null_count)
        var chunk = ColumnChunk()
        chunk.file_offset = Int64(chunk_start)
        chunk.meta_data = cm^
        return (chunk^, oi^, ci^)

    def finish(deinit self) raises -> List[UInt8]:
        """Close the file and return its bytes."""
        if not self._started:
            raise Error("parquet.writer: nothing was written")
        var out = self.out^
        var groups = self.row_groups^
        if self.options.write_page_index:
            for g in range(len(groups)):
                for c in range(len(groups[g].columns)):
                    var w = TCompactProtocolWriter()
                    self.offset_indexes[g][c].write(w)
                    var body = w^.take()
                    groups[g].columns[c].offset_index_offset = Int64(len(out))
                    groups[g].columns[c].offset_index_length = Int32(len(body))
                    out.extend(Span(body))
            for g in range(len(groups)):
                for c in range(len(groups[g].columns)):
                    var w = TCompactProtocolWriter()
                    self.column_indexes[g][c].write(w)
                    var body = w^.take()
                    groups[g].columns[c].column_index_offset = Int64(len(out))
                    groups[g].columns[c].column_index_length = Int32(len(body))
                    out.extend(Span(body))
        var meta = FileMetaData()
        meta.version = 2
        meta.schema = self.schema.elements.copy()
        meta.num_rows = self.num_rows
        meta.row_groups = groups^
        meta.created_by = self.options.created_by.copy()
        if len(self.kv):
            meta.key_value_metadata = self.kv.copy()
        var body = write_footer(meta)
        out.extend(Span(body))
        write_footer_trailer(out, len(body))
        return out^


def _bit_width(max_value: Int) -> Int:
    var w = 0
    var v = max_value
    while v > 0:
        w += 1
        v >>= 1
    return w


def _write_page(
    mut out: List[UInt8], var header: PageHeader, body: Span[UInt8, _]
) raises -> Int:
    """Append a page header and its body; return the header's byte length."""
    var w = TCompactProtocolWriter()
    header.write(w)
    var hdr = w^.take()
    out.extend(Span(hdr))
    out.extend(body)
    return len(hdr)


def _plain_bytes(leaf: WLeaf, values: PhysBuffer) raises -> List[UInt8]:
    return _plain_slice(leaf, values, 0, values.count)


def _plain_slice(
    leaf: WLeaf, values: PhysBuffer, start: Int, count: Int
) raises -> List[UInt8]:
    """PLAIN-encode `count` values starting at `start`."""
    var out = List[UInt8]()
    if values.kind == PK_BOOL:
        for k in range(count):
            var i = start + k
            if k % 8 == 0:
                out.append(0)
            if values.bool_at(i):
                out[len(out) - 1] |= UInt8(1) << UInt8(k % 8)
        return out^
    if values.kind == PK_VAR:
        for k in range(count):
            var v = values.value_span(start + k)
            var n = UInt32(len(v))
            for b in range(4):
                out.append(UInt8((n >> UInt32(8 * b)) & 0xFF))
            out.extend(v)
        return out^
    var w = values.width
    out.extend(Span(values.bytes)[start * w : (start + count) * w])
    return out^
