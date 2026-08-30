"""The Parquet schema tree, its leaf columns, and the Arrow schema it means.

A Parquet file's schema is a flat list of `SchemaElement`s in depth-first
order, each group carrying its child count. `ParquetSchema` turns that back
into a tree, computes the **maximum definition and repetition level** of every
leaf, and then derives the *Arrow* field tree by applying the LogicalTypes
rules — including the backward-compatibility rules for two-level lists that
old writers produced.

Three views come out of it:

* `nodes` — the Parquet tree itself, one node per `SchemaElement`.
* `leaves` — the columns that actually hold data, in column-chunk order.
* `fields` / `roots` — the Arrow field tree, which is what a `RecordBatch`
  looks like. A `LIST` field collapses the annotated group *and* its repeated
  child into one node; a `MAP` field collapses the annotated group and keeps
  the `key_value` group as its single `STRUCT` child, exactly as Arrow wants.
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
    ArrowType,
    TU_MICRO,
    TU_MILLI,
    TU_NANO,
    at,
    at_decimal,
    at_fixed,
    at_timestamp,
)
from thrift import ConvertedType, FieldRepetitionType, SchemaElement, Type

comptime REP_REQUIRED = 0
comptime REP_OPTIONAL = 1
comptime REP_REPEATED = 2


struct SchemaNode(Copyable, Defaultable, Movable):
    """One node of the Parquet schema tree."""

    var name: String
    var elem: Int
    """Index into `FileMetaData.schema`."""
    var parent: Int
    var children: List[Int]
    var repetition: Int
    var max_def: Int
    var max_rep: Int
    var is_leaf: Bool
    var leaf: Int
    """Index into `ParquetSchema.leaves`, or -1."""
    var path: List[String]
    var field_id: Int32
    """The Parquet field id, or -1 when the writer did not set one."""

    def __init__(out self):
        self.name = String()
        self.elem = 0
        self.parent = -1
        self.children = List[Int]()
        self.repetition = REP_REQUIRED
        self.max_def = 0
        self.max_rep = 0
        self.is_leaf = False
        self.leaf = -1
        self.path = List[String]()
        self.field_id = -1

    def __init__(out self, *, copy: Self):
        self.name = copy.name.copy()
        self.elem = copy.elem
        self.parent = copy.parent
        self.children = copy.children.copy()
        self.repetition = copy.repetition
        self.max_def = copy.max_def
        self.max_rep = copy.max_rep
        self.is_leaf = copy.is_leaf
        self.leaf = copy.leaf
        self.path = copy.path.copy()
        self.field_id = copy.field_id

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.elem = move.elem
        self.parent = move.parent
        self.children = move.children^
        self.repetition = move.repetition
        self.max_def = move.max_def
        self.max_rep = move.max_rep
        self.is_leaf = move.is_leaf
        self.leaf = move.leaf
        self.path = move.path^
        self.field_id = move.field_id


struct LeafColumn(Copyable, Defaultable, Movable):
    """A column that holds data — one column chunk per row group."""

    var node: Int
    var path: List[String]
    var physical: Int32
    var type_length: Int
    var max_def: Int
    var max_rep: Int
    var arrow: ArrowType
    var field_id: Int32
    var converted: Int32
    """The `ConvertedType`, or -1."""
    var scale: Int
    var precision: Int

    def __init__(out self):
        self.node = 0
        self.path = List[String]()
        self.physical = 0
        self.type_length = 0
        self.max_def = 0
        self.max_rep = 0
        self.arrow = ArrowType()
        self.field_id = -1
        self.converted = -1
        self.scale = 0
        self.precision = 0

    def __init__(out self, *, copy: Self):
        self.node = copy.node
        self.path = copy.path.copy()
        self.physical = copy.physical
        self.type_length = copy.type_length
        self.max_def = copy.max_def
        self.max_rep = copy.max_rep
        self.arrow = copy.arrow.copy()
        self.field_id = copy.field_id
        self.converted = copy.converted
        self.scale = copy.scale
        self.precision = copy.precision

    def __init__(out self, *, deinit move: Self):
        self.node = move.node
        self.path = move.path^
        self.physical = move.physical
        self.type_length = move.type_length
        self.max_def = move.max_def
        self.max_rep = move.max_rep
        self.arrow = move.arrow^
        self.field_id = move.field_id
        self.converted = move.converted
        self.scale = move.scale
        self.precision = move.precision

    def dotted(self) -> String:
        var out = String()
        for i in range(len(self.path)):
            if i:
                out += "."
            out += self.path[i]
        return out^


struct ArrowField(Copyable, Defaultable, Movable):
    """One node of the Arrow field tree derived from the Parquet schema."""

    var name: String
    var type: ArrowType
    var nullable: Bool
    var children: List[Int]
    var leaf: Int
    """Index into `ParquetSchema.leaves` for a primitive field, else -1."""
    var field_id: Int32
    var node: Int
    """The Parquet node this field came from."""
    var def_level: Int
    """This field is null in a record whose definition level is below this."""
    var rep_level: Int
    var elem_def_level: Int
    """`LIST`/`MAP` only: at or above this level the container has an element."""
    var elem_rep_level: Int
    """`LIST`/`MAP` only: a repetition level of exactly this continues a list."""

    def __init__(out self):
        self.name = String()
        self.type = ArrowType()
        self.nullable = True
        self.children = List[Int]()
        self.leaf = -1
        self.field_id = -1
        self.node = 0
        self.def_level = 0
        self.rep_level = 0
        self.elem_def_level = 0
        self.elem_rep_level = 0

    def __init__(out self, *, copy: Self):
        self.name = copy.name.copy()
        self.type = copy.type.copy()
        self.nullable = copy.nullable
        self.children = copy.children.copy()
        self.leaf = copy.leaf
        self.field_id = copy.field_id
        self.node = copy.node
        self.def_level = copy.def_level
        self.rep_level = copy.rep_level
        self.elem_def_level = copy.elem_def_level
        self.elem_rep_level = copy.elem_rep_level

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.type = move.type^
        self.nullable = move.nullable
        self.children = move.children^
        self.leaf = move.leaf
        self.field_id = move.field_id
        self.node = move.node
        self.def_level = move.def_level
        self.rep_level = move.rep_level
        self.elem_def_level = move.elem_def_level
        self.elem_rep_level = move.elem_rep_level


def _rep_of(el: SchemaElement) -> Int:
    if el.repetition_type:
        return Int(el.repetition_type.value().value)
    return REP_REQUIRED


def _converted_of(el: SchemaElement) -> Int32:
    if el.converted_type:
        return el.converted_type.value().value
    return -1


def _time_unit(el: SchemaElement) -> Int:
    if el.logicalType:
        ref lt = el.logicalType.value()
        if lt.TIME:
            ref u = lt.TIME.value().unit
            if u.MILLIS:
                return TU_MILLI
            if u.NANOS:
                return TU_NANO
            return TU_MICRO
        if lt.TIMESTAMP:
            ref u2 = lt.TIMESTAMP.value().unit
            if u2.MILLIS:
                return TU_MILLI
            if u2.NANOS:
                return TU_NANO
            return TU_MICRO
    return TU_MICRO


def arrow_type_of(el: SchemaElement) raises -> ArrowType:
    """The Arrow type of one primitive `SchemaElement`.

    `LogicalType` wins when present; `ConvertedType` is the fallback for files
    written before Parquet 2.4. Anything unrecognised falls back to the plain
    physical type, which is what every other reader does.
    """
    if not el.type_:
        raise Error(String("parquet.schema: leaf '", el.name, "' has no type"))
    var phys = el.type_.value().value
    var conv = _converted_of(el)
    var width = Int(el.type_length.or_else(0))
    var precision = Int(el.precision.or_else(0))
    var scale = Int(el.scale.or_else(0))

    var has_logical = Bool(el.logicalType)
    var is_string = False
    var is_json = False
    var is_bson = False
    var is_uuid = False
    var is_float16 = False
    var is_decimal = False
    var is_date = False
    var is_time = False
    var is_timestamp = False
    var is_enum = False
    var int_bits = 0
    var int_signed = True
    var utc = False
    if has_logical:
        ref lt = el.logicalType.value()
        is_string = Bool(lt.STRING)
        is_json = Bool(lt.JSON)
        is_bson = Bool(lt.BSON)
        is_uuid = Bool(lt.UUID)
        is_float16 = Bool(lt.FLOAT16)
        is_enum = Bool(lt.ENUM)
        if lt.DECIMAL:
            is_decimal = True
            precision = Int(lt.DECIMAL.value().precision)
            scale = Int(lt.DECIMAL.value().scale)
        is_date = Bool(lt.DATE)
        if lt.TIME:
            is_time = True
            utc = lt.TIME.value().isAdjustedToUTC
        if lt.TIMESTAMP:
            is_timestamp = True
            utc = lt.TIMESTAMP.value().isAdjustedToUTC
        if lt.INTEGER:
            int_bits = Int(lt.INTEGER.value().bitWidth)
            int_signed = lt.INTEGER.value().isSigned
    # ConvertedType fallbacks.
    if conv == ConvertedType.UTF8.value:
        is_string = True
    elif conv == ConvertedType.JSON.value:
        is_json = True
    elif conv == ConvertedType.BSON.value:
        is_bson = True
    elif conv == ConvertedType.ENUM.value:
        is_enum = True
    elif conv == ConvertedType.DECIMAL.value:
        is_decimal = True
    elif conv == ConvertedType.DATE.value:
        is_date = True
    elif (
        conv == ConvertedType.TIME_MILLIS.value
        or conv == ConvertedType.TIME_MICROS.value
    ):
        is_time = True
    elif (
        conv == ConvertedType.TIMESTAMP_MILLIS.value
        or conv == ConvertedType.TIMESTAMP_MICROS.value
    ):
        is_timestamp = True
    elif (
        conv >= ConvertedType.UINT_8.value
        and conv <= ConvertedType.UINT_64.value
    ):
        int_signed = False
        int_bits = 8 << Int(conv - ConvertedType.UINT_8.value)
    elif (
        conv >= ConvertedType.INT_8.value and conv <= ConvertedType.INT_64.value
    ):
        int_signed = True
        int_bits = 8 << Int(conv - ConvertedType.INT_8.value)

    var unit = _time_unit(el)
    if not has_logical:
        if conv == ConvertedType.TIME_MILLIS.value:
            unit = TU_MILLI
        elif conv == ConvertedType.TIME_MICROS.value:
            unit = TU_MICRO
        elif conv == ConvertedType.TIMESTAMP_MILLIS.value:
            unit = TU_MILLI
        elif conv == ConvertedType.TIMESTAMP_MICROS.value:
            unit = TU_MICRO
        # A converted-type-only timestamp is always UTC-adjusted.
        if is_timestamp:
            utc = True

    if phys == Type.BOOLEAN.value:
        return at(AT_BOOL)
    if phys == Type.INT32.value:
        if is_decimal:
            return at_decimal(precision, scale)
        if is_date:
            return at(AT_DATE32)
        if is_time:
            var t = ArrowType(AT_TIME32)
            t.unit = unit if unit == TU_MILLI else TU_MILLI
            return t^
        if int_bits == 8:
            return at(AT_INT8 if int_signed else AT_UINT8)
        if int_bits == 16:
            return at(AT_INT16 if int_signed else AT_UINT16)
        if int_bits == 32 and not int_signed:
            return at(AT_UINT32)
        return at(AT_INT32)
    if phys == Type.INT64.value:
        if is_decimal:
            return at_decimal(precision, scale)
        if is_timestamp:
            return at_timestamp(unit, String("UTC") if utc else String())
        if is_time:
            var t = ArrowType(AT_TIME64)
            t.unit = unit if unit != TU_MILLI else TU_MICRO
            return t^
        if int_bits == 64 and not int_signed:
            return at(AT_UINT64)
        return at(AT_INT64)
    if phys == Type.INT96.value:
        # Deprecated, and always a nanosecond timestamp with no time zone —
        # this is what pyarrow, arrow-rs and parquet-mr all do with it.
        return at_timestamp(TU_NANO, String())
    if phys == Type.FLOAT.value:
        return at(AT_FLOAT32)
    if phys == Type.DOUBLE.value:
        return at(AT_FLOAT64)
    if phys == Type.BYTE_ARRAY.value:
        if is_decimal:
            return at_decimal(precision, scale)
        if is_json:
            var t = at(AT_UTF8)
            t.extension = String("arrow.json")
            return t^
        if is_bson:
            return at(AT_BINARY)
        if is_string or is_enum:
            return at(AT_UTF8)
        return at(AT_BINARY)
    if phys == Type.FIXED_LEN_BYTE_ARRAY.value:
        if is_decimal:
            return at_decimal(precision, scale)
        if is_float16:
            return at(AT_FLOAT16)
        if is_uuid:
            var t = at_fixed(16)
            t.extension = String("arrow.uuid")
            return t^
        return at_fixed(width)
    raise Error(String("parquet.schema: unknown physical type ", phys))


struct ParquetSchema(Copyable, Defaultable, Movable):
    """The schema tree, its leaves, and the Arrow fields they add up to."""

    var nodes: List[SchemaNode]
    var leaves: List[LeafColumn]
    var fields: List[ArrowField]
    var roots: List[Int]

    def __init__(out self):
        self.nodes = List[SchemaNode]()
        self.leaves = List[LeafColumn]()
        self.fields = List[ArrowField]()
        self.roots = List[Int]()

    def __init__(out self, *, copy: Self):
        self.nodes = copy.nodes.copy()
        self.leaves = copy.leaves.copy()
        self.fields = copy.fields.copy()
        self.roots = copy.roots.copy()

    def __init__(out self, *, deinit move: Self):
        self.nodes = move.nodes^
        self.leaves = move.leaves^
        self.fields = move.fields^
        self.roots = move.roots^

    def num_leaves(self) -> Int:
        return len(self.leaves)

    def leaf_by_path(self, path: StringSlice) -> Int:
        """The leaf whose dotted path is `path`, or -1."""
        for i in range(len(self.leaves)):
            if self.leaves[i].dotted() == path:
                return i
        return -1

    def field_by_name(self, name: StringSlice) -> Int:
        for i in range(len(self.roots)):
            if self.fields[self.roots[i]].name == name:
                return self.roots[i]
        return -1

    def field_by_id(self, id: Int32) -> Int:
        """Depth-first search of the whole tree for a Parquet field id."""
        for i in range(len(self.fields)):
            if self.fields[i].field_id == id:
                return i
        return -1


def _elem_field_id(el: SchemaElement) -> Int32:
    if el.field_id:
        return el.field_id.value()
    return -1


def _build_node(
    mut s: ParquetSchema,
    elements: List[SchemaElement],
    mut pos: Int,
    parent: Int,
) raises -> Int:
    """Consume one `SchemaElement` and its whole subtree; return its node index.
    """
    if pos >= len(elements):
        raise Error("parquet.schema: schema list ends inside a group")
    var idx = pos
    pos += 1
    ref el = elements[idx]
    var node = SchemaNode()
    node.name = el.name.copy()
    node.elem = idx
    node.parent = parent
    node.repetition = _rep_of(el)
    node.field_id = _elem_field_id(el)
    if parent >= 0:
        node.max_def = s.nodes[parent].max_def
        node.max_rep = s.nodes[parent].max_rep
        node.path = s.nodes[parent].path.copy()
        node.path.append(el.name.copy())
        if node.repetition == REP_OPTIONAL:
            node.max_def += 1
        elif node.repetition == REP_REPEATED:
            node.max_def += 1
            node.max_rep += 1
    var n_children = Int(el.num_children.or_else(0))
    node.is_leaf = n_children == 0 and parent >= 0
    var me = len(s.nodes)
    s.nodes.append(node^)
    for _ in range(n_children):
        var child = _build_node(s, elements, pos, me)
        s.nodes[me].children.append(child)
    return me


def _collect_leaves(
    mut s: ParquetSchema, elements: List[SchemaElement], n: Int
) raises:
    if s.nodes[n].is_leaf:
        ref el = elements[s.nodes[n].elem]
        var lc = LeafColumn()
        lc.node = n
        lc.path = s.nodes[n].path.copy()
        lc.physical = el.type_.value().value if el.type_ else -1
        lc.type_length = Int(el.type_length.or_else(0))
        lc.max_def = s.nodes[n].max_def
        lc.max_rep = s.nodes[n].max_rep
        lc.arrow = arrow_type_of(el)
        lc.field_id = s.nodes[n].field_id
        lc.converted = _converted_of(el)
        lc.scale = Int(el.scale.or_else(0))
        lc.precision = Int(el.precision.or_else(0))
        s.nodes[n].leaf = len(s.leaves)
        s.leaves.append(lc^)
        return
    var kids = s.nodes[n].children.copy()
    for c in kids:
        _collect_leaves(s, elements, c)


def _is_list_group(
    s: ParquetSchema, elements: List[SchemaElement], n: Int
) -> Bool:
    ref el = elements[s.nodes[n].elem]
    if el.logicalType and Bool(el.logicalType.value().LIST):
        return True
    return _converted_of(el) == ConvertedType.LIST.value


def _is_map_group(
    s: ParquetSchema, elements: List[SchemaElement], n: Int
) -> Bool:
    ref el = elements[s.nodes[n].elem]
    if el.logicalType and Bool(el.logicalType.value().MAP):
        return True
    var c = _converted_of(el)
    return (
        c == ConvertedType.MAP.value or c == ConvertedType.MAP_KEY_VALUE.value
    )


def _make_field(
    mut s: ParquetSchema,
    elements: List[SchemaElement],
    n: Int,
    var name: String,
    force_required: Bool,
) raises -> Int:
    """Build the Arrow field for Parquet node `n`; return its index."""
    var rep = REP_REQUIRED if force_required else s.nodes[n].repetition
    var is_leaf = s.nodes[n].is_leaf
    var leaf = s.nodes[n].leaf
    var max_def = s.nodes[n].max_def
    var max_rep = s.nodes[n].max_rep
    var field_id = s.nodes[n].field_id
    var f = ArrowField()
    f.name = name^
    f.node = n
    f.field_id = field_id
    f.def_level = max_def
    f.rep_level = max_rep

    if is_leaf and rep != REP_REPEATED:
        f.type = s.leaves[leaf].arrow.copy()
        f.leaf = leaf
        f.nullable = rep == REP_OPTIONAL
        s.fields.append(f^)
        return len(s.fields) - 1

    if is_leaf:
        # A repeated primitive is a two-level list of that primitive.
        var elem = ArrowField()
        elem.name = String("element")
        elem.node = n
        elem.field_id = field_id
        elem.type = s.leaves[leaf].arrow.copy()
        elem.leaf = leaf
        elem.nullable = False
        elem.def_level = max_def
        elem.rep_level = max_rep
        s.fields.append(elem^)
        var ei = len(s.fields) - 1
        f.type = at(AT_LIST)
        f.nullable = False
        f.def_level = max_def - 1
        f.rep_level = max_rep - 1
        f.elem_def_level = max_def
        f.elem_rep_level = max_rep
        f.children.append(ei)
        s.fields.append(f^)
        return len(s.fields) - 1

    var children = s.nodes[n].children.copy()
    var is_list = _is_list_group(s, elements, n)
    var is_map = _is_map_group(s, elements, n)
    if len(children) == 1 and (is_list or is_map):
        var rn = children[0]
        if s.nodes[rn].repetition != REP_REPEATED:
            raise Error(
                String(
                    "parquet.schema: '",
                    s.nodes[n].name,
                    "' is annotated LIST/MAP but its child '",
                    s.nodes[rn].name,
                    "' is not repeated",
                )
            )
        var rep_is_leaf = s.nodes[rn].is_leaf
        var rep_kids = len(s.nodes[rn].children)
        var rep_name = s.nodes[rn].name.copy()
        var rep_def = s.nodes[rn].max_def
        var rep_rep = s.nodes[rn].max_rep
        # Two-level backward compatibility, for LIST only.
        var two_level = False
        if not is_map:
            if rep_is_leaf:
                two_level = True
            elif rep_kids != 1:
                two_level = True
            elif rep_name == "array" or rep_name == String(
                s.nodes[n].name, "_tuple"
            ):
                two_level = True
        var elem_index: Int
        if is_map:
            elem_index = _make_field(s, elements, rn, rep_name.copy(), True)
        elif two_level:
            elem_index = _make_field(s, elements, rn, String("element"), True)
        else:
            var gn = s.nodes[rn].children[0]
            elem_index = _make_field(
                s, elements, gn, s.nodes[gn].name.copy(), False
            )
        f.type = at(AT_MAP if is_map else AT_LIST)
        f.nullable = rep == REP_OPTIONAL
        f.elem_def_level = rep_def
        f.elem_rep_level = rep_rep
        f.children.append(elem_index)
        s.fields.append(f^)
        return len(s.fields) - 1

    if rep == REP_REPEATED:
        # A repeated group with no annotation: a list of struct.
        var inner = ArrowField()
        inner.name = String("element")
        inner.node = n
        inner.type = at(AT_STRUCT)
        inner.nullable = False
        inner.def_level = max_def
        inner.rep_level = max_rep
        for c in children:
            inner.children.append(
                _make_field(s, elements, c, s.nodes[c].name.copy(), False)
            )
        s.fields.append(inner^)
        var ii = len(s.fields) - 1
        f.type = at(AT_LIST)
        f.nullable = False
        f.def_level = max_def - 1
        f.rep_level = max_rep - 1
        f.elem_def_level = max_def
        f.elem_rep_level = max_rep
        f.children.append(ii)
        s.fields.append(f^)
        return len(s.fields) - 1

    f.type = at(AT_STRUCT)
    f.nullable = rep == REP_OPTIONAL
    var kids = List[Int]()
    for c in children:
        kids.append(_make_field(s, elements, c, s.nodes[c].name.copy(), False))
    f.children = kids^
    s.fields.append(f^)
    return len(s.fields) - 1


def build_schema(elements: List[SchemaElement]) raises -> ParquetSchema:
    """Turn the flat depth-first `SchemaElement` list into a `ParquetSchema`."""
    if len(elements) == 0:
        raise Error("parquet.schema: empty schema")
    var s = ParquetSchema()
    var pos = 0
    _ = _build_node(s, elements, pos, -1)
    if pos != len(elements):
        raise Error(
            String(
                "parquet.schema: ",
                len(elements) - pos,
                " trailing schema element(s) — num_children does not add up",
            )
        )
    var top = s.nodes[0].children.copy()
    for c in top:
        _collect_leaves(s, elements, c)
    for c in top:
        s.roots.append(
            _make_field(s, elements, c, s.nodes[c].name.copy(), False)
        )
    return s^
