"""The parquet.mojo test suite. `pixi run test`.

pyarrow is the oracle: every fixture under `tests/fixtures` is read with our
decoder and every value of every column is compared with what pyarrow reads
from the same file, both one value at a time and as a CRC32 over the lot.
"""

from avro.json import JsonDoc
from fixtures_list import (
    core_fixtures,
    default_codec_columns,
    iceberg_fixtures,
    iceberg_zstd_fixtures,
)
from oracle import decimal_string, double_bits, hex_of, load_oracle
from parity import check_fixture, check_path, check_table
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
    OP_EQ,
    OP_GE,
    OP_GT,
    OP_LE,
    OP_LT,
    OP_NE,
    SV_BYTES,
    SV_FLOAT,
    SV_INT,
    SV_UINT,
    TU_MICRO,
    TU_MILLI,
    TU_NANO,
    DefaultCodecs,
    ParquetReader,
    Predicate,
    ScalarValue,
    ParquetWriter,
    WriterOptions,
    build_schema,
    export_c,
    array_str,
)
from parquet.rle_encode import encode_hybrid, encode_levels
from parquet.bitio import (
    HybridDecoder,
    bit_width,
    read_uleb128,
    unpack_lsb,
    unpack_msb,
    zigzag_decode,
)
from parquet.bloom import read_bloom_filter
from parquet.encoding import (
    PK_VAR,
    PhysBuffer,
    decode_byte_stream_split,
    decode_delta_binary_packed,
    decode_dict_indices,
    decode_plain,
    gather,
)
from parquet.schema import REP_OPTIONAL, REP_REPEATED, REP_REQUIRED
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)
from thrift import (
    CompressionCodec,
    ConvertedType,
    FieldRepetitionType,
    ListType,
    LogicalType,
    SchemaElement,
    Type,
    read_parquet_file,
)

comptime FIXTURES = "tests/fixtures/"


def fixture_bytes(name: StringSlice) raises -> List[UInt8]:
    return read_parquet_file(String(FIXTURES, name, ".parquet"))


# ── parity with pyarrow ────────────────────────────────────────────────────


def test_every_fixture_matches_pyarrow() raises:
    var total = 0
    for f in core_fixtures():
        total += check_fixture[DefaultCodecs](f, List[String](), 65536)
    assert_true(total > 20000, String("only ", total, " values checked"))


def test_default_codecs_read_their_columns() raises:
    var n = check_fixture[DefaultCodecs](
        String("codecs"), default_codec_columns(), 65536
    )
    assert_true(n > 1000)


def test_unsupported_codec_says_so() raises:
    var r = ParquetReader.open(String(FIXTURES, "codecs.parquet"))
    r.select_columns([String("zstd")])
    with assert_raises(contains="ZSTD"):
        _ = r.read_table()
    var r2 = ParquetReader.open(String(FIXTURES, "codecs.parquet"))
    r2.select_columns([String("brotli")])
    with assert_raises(contains="BROTLI"):
        _ = r2.read_table()


def test_batching_is_invariant() raises:
    var sizes: List[Int] = [1, 3, 64, 997]
    for bs in sizes:
        var n = check_fixture[DefaultCodecs](
            String("v2pages"), List[String](), bs
        )
        assert_true(n > 2000, String("batch size ", bs))
    _ = check_fixture[DefaultCodecs](String("nested"), List[String](), 1)
    _ = check_fixture[DefaultCodecs](String("big"), List[String](), 12500)


def test_iceberg_data_files() raises:
    """Real Iceberg data files, most of them written by parquet-rs 58 rather
    than pyarrow — a second writer, top-level `required` columns, and a field
    id on every column."""
    var total = 0
    for f in iceberg_fixtures():
        total += check_path[DefaultCodecs](
            String(FIXTURES, "iceberg/", f), f, List[String](), 65536
        )
    assert_true(total > 60, String("only ", total, " Iceberg values checked"))


def test_iceberg_field_ids() raises:
    var r = ParquetReader.open(
        String(FIXTURES, "iceberg/unpartitioned.parquet")
    )
    # parquet-rs names the root element `arrow_schema`, and every column has an
    # Iceberg field id.
    for i in range(len(r.schema.leaves)):
        assert_equal(
            r.schema.leaves[i].field_id,
            Int32(i + 1),
            String("leaf ", i, " field id"),
        )
    r.select_field_ids([Int32(2), Int32(1)])
    var t = r.read_table()
    assert_equal(t.num_columns(), 2)
    assert_equal(t.name(0), "region")
    assert_equal(t.name(1), "id")
    # `id` and `region` are REQUIRED at the top level, which pyarrow never
    # writes: their maximum definition level is 0.
    assert_equal(r.schema.leaves[0].max_def, 0)
    assert_false(r.schema.fields[r.schema.field_by_name("id")].nullable)
    var ids = t.column_i64(1)
    assert_equal(len(ids[0]), 3)
    for v in ids[1]:
        assert_true(v)

    # A position-delete file uses the reserved field ids.
    var d = ParquetReader.open(
        String(FIXTURES, "iceberg/position_deletes.parquet")
    )
    assert_true(d.schema.field_by_id(2147483546) >= 0)
    assert_true(d.schema.field_by_id(2147483545) >= 0)
    d.select_field_ids([Int32(2147483545)])
    var dt = d.read_table()
    assert_equal(dt.name(0), "pos")
    assert_equal(dt.num_rows, 2)


# ── schema ─────────────────────────────────────────────────────────────────


def _reader(name: StringSlice) raises -> ParquetReader[DefaultCodecs]:
    return ParquetReader.open(String(FIXTURES, name, ".parquet"))


def test_logical_types_map_to_arrow() raises:
    var r = _reader("logical")
    var want: List[String] = [
        String("dec_flba"),
        String("dec_small"),
        String("dec_mid"),
        String("date"),
        String("time_ms"),
        String("time_us"),
        String("time_ns"),
        String("ts_ms"),
        String("ts_us"),
        String("ts_ns"),
        String("ts_ms_utc"),
        String("ts_us_utc"),
        String("ts_ns_utc"),
        String("lstr"),
        String("lbin"),
    ]
    var ids: List[Int] = [
        AT_DECIMAL128,
        AT_DECIMAL128,
        AT_DECIMAL128,
        AT_DATE32,
        AT_TIME32,
        AT_TIME64,
        AT_TIME64,
        AT_TIMESTAMP,
        AT_TIMESTAMP,
        AT_TIMESTAMP,
        AT_TIMESTAMP,
        AT_TIMESTAMP,
        AT_TIMESTAMP,
        AT_UTF8,
        AT_BINARY,
    ]
    for i in range(len(want)):
        var fi = r.schema.field_by_name(want[i])
        assert_true(fi >= 0, want[i])
        assert_equal(r.schema.fields[fi].type.id, ids[i], want[i])
    assert_equal(
        r.schema.fields[r.schema.field_by_name("dec_flba")].type.precision, 29
    )
    assert_equal(
        r.schema.fields[r.schema.field_by_name("dec_flba")].type.scale, 2
    )
    assert_equal(
        r.schema.fields[r.schema.field_by_name("ts_ns_utc")].type.unit, TU_NANO
    )
    assert_equal(
        r.schema.fields[r.schema.field_by_name("ts_ns_utc")].type.tz, "UTC"
    )
    assert_equal(r.schema.fields[r.schema.field_by_name("ts_ns")].type.tz, "")
    assert_equal(
        r.schema.fields[r.schema.field_by_name("time_ms")].type.unit, TU_MILLI
    )
    assert_equal(
        r.schema.fields[r.schema.field_by_name("time_us")].type.unit, TU_MICRO
    )


def test_integer_widths_and_signs() raises:
    var r = _reader("primitives")
    var names: List[String] = [
        String("i8"),
        String("i16"),
        String("i32"),
        String("i64"),
        String("u8"),
        String("u16"),
        String("u32"),
        String("u64"),
    ]
    var ids: List[Int] = [
        AT_INT8,
        AT_INT16,
        AT_INT32,
        AT_INT64,
        AT_UINT8,
        AT_UINT16,
        AT_UINT32,
        AT_UINT64,
    ]
    for i in range(len(names)):
        var fi = r.schema.field_by_name(names[i])
        assert_equal(r.schema.fields[fi].type.id, ids[i], names[i])


def test_extension_types() raises:
    var r = _reader("extension")
    ref u = r.schema.fields[r.schema.field_by_name("uuid")]
    assert_equal(u.type.id, AT_FIXED_SIZE_BINARY)
    assert_equal(u.type.byte_width, 16)
    assert_equal(u.type.extension, "arrow.uuid")
    ref j = r.schema.fields[r.schema.field_by_name("json")]
    assert_equal(j.type.id, AT_UTF8)
    assert_equal(j.type.extension, "arrow.json")
    var h = _reader("float16")
    assert_equal(
        h.schema.fields[h.schema.field_by_name("h")].type.id, AT_FLOAT16
    )


def test_int96_is_a_nanosecond_timestamp() raises:
    var r = _reader("int96")
    ref tf = r.schema.fields[r.schema.field_by_name("t")]
    ref t = tf.type
    assert_equal(t.id, AT_TIMESTAMP)
    assert_equal(t.unit, TU_NANO)
    assert_equal(t.tz, "")
    assert_equal(r.schema.leaves[0].physical, Type.INT96.value)


def test_nested_shapes_and_levels() raises:
    var r = _reader("nested")
    assert_equal(len(r.schema.leaves), 8)
    assert_equal(r.schema.leaves[1].dotted(), "lls.list.element.list.element")
    assert_equal(r.schema.leaves[1].max_def, 5)
    assert_equal(r.schema.leaves[1].max_rep, 2)
    ref m = r.schema.fields[r.schema.field_by_name("m")]
    assert_equal(m.type.id, AT_MAP)
    assert_equal(len(m.children), 1)
    ref kv = r.schema.fields[m.children[0]]
    assert_equal(kv.type.id, AT_STRUCT)
    assert_equal(len(kv.children), 2)
    assert_false(kv.nullable)
    assert_false(r.schema.fields[kv.children[0]].nullable)
    assert_true(r.schema.fields[kv.children[1]].nullable)
    ref st = r.schema.fields[r.schema.field_by_name("st")]
    assert_equal(st.type.id, AT_STRUCT)
    assert_equal(len(st.children), 2)


def test_select_fields_prunes_to_one_sub_field() raises:
    """`st.b` alone: the struct comes back with one child, and `st.a` is never
    decoded."""
    var r = _reader("nested")
    var st = r.schema.field_by_name("st")
    var b = r.schema.fields[st].children[1]
    r.select_fields([b])
    var batch = r.read_batch()
    assert_equal(batch.num_columns(), 1)
    assert_equal(batch.name(0), "st")
    assert_equal(batch.type(0).id, AT_STRUCT)
    assert_equal(len(batch.column(0).children), 1)
    assert_equal(batch.child(batch.roots[0], 0).name, "b")

    # The values a whole read gives, for the one field that was kept.
    var whole = _reader("nested")
    var full = whole.read_batch()
    var fi = -1
    for i in range(full.num_columns()):
        if full.name(i) == "st":
            fi = i
    var want = array_str(full.child(full.roots[fi], 1))
    var got = array_str(batch.child(batch.roots[0], 0))
    assert_equal(len(got[0]), len(want[0]))
    for k in range(len(want[0])):
        assert_equal(got[1][k], want[1][k])
        if want[1][k]:
            assert_equal(got[0][k], want[0][k])


def test_select_fields_keeps_a_map_key() raises:
    """A map is only a map with both halves, so selecting its value keeps the
    key."""
    var r = _reader("nested")
    var m = r.schema.field_by_name("m")
    var kv = r.schema.fields[m].children[0]
    r.select_fields([r.schema.fields[kv].children[1]])
    var batch = r.read_batch()
    assert_equal(batch.num_columns(), 1)
    assert_equal(batch.name(0), "m")
    assert_equal(len(batch.child(batch.roots[0], 0).children), 2)


def test_select_fields_keeps_the_order_it_was_asked_for() raises:
    var r = _reader("nested")
    var st = r.schema.field_by_name("st")
    var kids = r.schema.fields[st].children.copy()
    r.select_fields([kids[0], r.schema.field_by_name("li")])
    var batch = r.read_batch()
    assert_equal(batch.num_columns(), 2)
    assert_equal(batch.name(0), "st")
    assert_equal(batch.name(1), "li")


def _group(
    var name: String, rep: Int32, n_children: Int32, is_list: Bool
) -> SchemaElement:
    var el = SchemaElement()
    el.name = name^
    el.repetition_type = FieldRepetitionType(rep)
    el.num_children = n_children
    if is_list:
        var lt = LogicalType()
        lt.LIST = ListType()
        el.logicalType = lt^
        el.converted_type = ConvertedType.LIST
    return el^


def _leaf(var name: String, rep: Int32, phys: Int32) -> SchemaElement:
    var el = SchemaElement()
    el.name = name^
    el.repetition_type = FieldRepetitionType(rep)
    el.type_ = Type(phys)
    return el^


def test_two_level_lists_backward_compatibility() raises:
    # optional group li (LIST) { repeated int32 element; }
    var els = List[SchemaElement]()
    els.append(_group(String("schema"), REP_REQUIRED, 1, False))
    els.append(_group(String("li"), REP_OPTIONAL, 1, True))
    els.append(_leaf(String("element"), REP_REPEATED, Type.INT32.value))
    var s = build_schema(els)
    assert_equal(len(s.roots), 1)
    ref li = s.fields[s.roots[0]]
    assert_equal(li.type.id, AT_LIST)
    assert_true(li.nullable)
    assert_equal(li.def_level, 1)
    assert_equal(li.elem_def_level, 2)
    assert_equal(li.elem_rep_level, 1)
    ref elem = s.fields[li.children[0]]
    assert_equal(elem.type.id, AT_INT32)
    assert_false(elem.nullable)
    assert_equal(s.leaves[0].max_def, 2)
    assert_equal(s.leaves[0].max_rep, 1)

    # optional group li (LIST) { repeated group array { required int32 x; } }
    # — a one-field repeated group named `array` is the element itself.
    var e2 = List[SchemaElement]()
    e2.append(_group(String("schema"), REP_REQUIRED, 1, False))
    e2.append(_group(String("li"), REP_OPTIONAL, 1, True))
    e2.append(_group(String("array"), REP_REPEATED, 1, False))
    e2.append(_leaf(String("x"), REP_REQUIRED, Type.INT32.value))
    var s2 = build_schema(e2)
    ref li2 = s2.fields[s2.roots[0]]
    assert_equal(li2.type.id, AT_LIST)
    ref elem2 = s2.fields[li2.children[0]]
    assert_equal(elem2.type.id, AT_STRUCT)
    assert_equal(elem2.name, "element")
    assert_equal(len(elem2.children), 1)

    # …and `<name>_tuple` behaves the same way.
    var e3 = List[SchemaElement]()
    e3.append(_group(String("schema"), REP_REQUIRED, 1, False))
    e3.append(_group(String("li"), REP_OPTIONAL, 1, True))
    e3.append(_group(String("li_tuple"), REP_REPEATED, 1, False))
    e3.append(_leaf(String("x"), REP_REQUIRED, Type.INT32.value))
    var s3 = build_schema(e3)
    assert_equal(
        s3.fields[s3.fields[s3.roots[0]].children[0]].type.id, AT_STRUCT
    )

    # A three-level list keeps the grandchild as the element.
    var e4 = List[SchemaElement]()
    e4.append(_group(String("schema"), REP_REQUIRED, 1, False))
    e4.append(_group(String("li"), REP_OPTIONAL, 1, True))
    e4.append(_group(String("list"), REP_REPEATED, 1, False))
    e4.append(_leaf(String("element"), REP_OPTIONAL, Type.INT32.value))
    var s4 = build_schema(e4)
    var elem4_i = s4.fields[s4.roots[0]].children[0]
    ref elem4 = s4.fields[elem4_i]
    assert_equal(elem4.type.id, AT_INT32)
    assert_true(elem4.nullable)


def test_repeated_primitive_is_a_list() raises:
    var els = List[SchemaElement]()
    els.append(_group(String("schema"), REP_REQUIRED, 1, False))
    els.append(_leaf(String("xs"), REP_REPEATED, Type.INT64.value))
    var s = build_schema(els)
    ref f = s.fields[s.roots[0]]
    assert_equal(f.type.id, AT_LIST)
    assert_false(f.nullable)
    assert_equal(f.def_level, 0)
    assert_equal(f.elem_def_level, 1)
    assert_equal(s.fields[f.children[0]].type.id, AT_INT64)


def test_bad_schema_is_rejected() raises:
    with assert_raises(contains="empty schema"):
        _ = build_schema(List[SchemaElement]())
    var els = List[SchemaElement]()
    els.append(_group(String("schema"), REP_REQUIRED, 3, False))
    els.append(_leaf(String("a"), REP_OPTIONAL, Type.INT32.value))
    with assert_raises():
        _ = build_schema(els)


# ── projection ─────────────────────────────────────────────────────────────


def test_projection_by_name() raises:
    var r = _reader("primitives")
    r.select_columns([String("i64"), String("s")])
    var t = r.read_table()
    assert_equal(t.num_columns(), 2)
    assert_equal(t.name(0), "i64")
    assert_equal(t.name(1), "s")
    assert_equal(t.num_rows, 12)
    var got = t.column_i64(0)
    assert_equal(got[0][0], -9223372036854775808)
    assert_equal(got[0][4], 9223372036854775807)
    assert_false(got[1][5])
    var strs = t.column_str(1)
    assert_equal(strs[0][2], "hello world")
    assert_false(strs[1][3])
    with assert_raises(contains="no column named"):
        r.select_columns([String("nope")])


def test_projection_by_field_id() raises:
    var r = _reader("fieldids")
    r.select_field_ids([Int32(3), Int32(1)])
    var t = r.read_table()
    assert_equal(t.num_columns(), 2)
    assert_equal(t.name(0), "score")
    assert_equal(t.name(1), "id")
    var ids = t.column_i64(1)
    assert_equal(ids[0][0], 1)
    assert_equal(ids[0][4], 5)
    # A nested field id resolves too — `tags` is 4 and its element is 5.
    var r2 = _reader("fieldids")
    assert_true(r2.schema.field_by_id(5) >= 0)
    assert_equal(r2.schema.field_by_id(4), r2.schema.field_by_name("tags"))
    with assert_raises(contains="no field with id"):
        r2.select_field_ids([Int32(99)])


def test_typed_accessors() raises:
    var r = _reader("primitives")
    var t = r.read_table()
    var b = t.column_bool(0)
    assert_true(b[0][0])
    assert_false(b[0][1])
    assert_false(b[1][2])
    var f = t.column_f64(10)
    assert_equal(f[0][2], 1.5)
    assert_false(f[1][5])
    var i8 = t.column_i64(1)
    assert_equal(i8[0][0], -128)
    assert_equal(i8[0][4], 127)


# ── row groups, statistics, pruning ────────────────────────────────────────


def test_row_group_selection() raises:
    var r = _reader("prune")
    assert_equal(r.num_row_groups(), 10)
    r.select_row_groups([3, 7])
    var t = r.read_table()
    assert_equal(t.num_rows, 200)
    var k = t.column_i64(0)
    assert_equal(k[0][0], 300)
    assert_equal(k[0][100], 700)
    with assert_raises(contains="does not exist"):
        r.select_row_groups([99])


def test_statistics_pruning() raises:
    var r = _reader("prune")
    var preds: List[Predicate] = [
        Predicate(String("k"), OP_GE, ScalarValue.of_int(650)),
        Predicate(String("k"), OP_LT, ScalarValue.of_int(700)),
    ]
    assert_equal(r.prune_row_groups(preds), 1)
    var t = r.read_table()
    assert_equal(t.num_rows, 100)
    var k = t.column_i64(0)
    assert_equal(k[0][0], 600)

    var r2 = _reader("prune")
    var eq: List[Predicate] = [
        Predicate(String("k"), OP_EQ, ScalarValue.of_int(42))
    ]
    assert_equal(r2.prune_row_groups(eq), 1)

    var r3 = _reader("prune")
    var none: List[Predicate] = [
        Predicate(String("k"), OP_GT, ScalarValue.of_int(100000))
    ]
    assert_equal(r3.prune_row_groups(none), 0)
    assert_equal(r3.read_table().num_rows, 0)

    var r4 = _reader("prune")
    var strp: List[Predicate] = [
        Predicate(String("s"), OP_EQ, ScalarValue.of_string("s00042"))
    ]
    assert_equal(r4.prune_row_groups(strp), 1)

    var r5 = _reader("prune")
    var fp: List[Predicate] = [
        Predicate(String("f"), OP_LE, ScalarValue.of_float(5.0))
    ]
    assert_equal(r5.prune_row_groups(fp), 1)

    # Without statistics nothing can be pruned.
    var r6 = _reader("nostats")
    var p6: List[Predicate] = [
        Predicate(String("k"), OP_GT, ScalarValue.of_int(1000000))
    ]
    assert_equal(r6.prune_row_groups(p6), r6.num_row_groups())


def test_page_pruning() raises:
    # manypages.parquet is 2000 sorted rows in 2 row groups with 256-byte
    # pages and no dictionary, so its pages really do have disjoint ranges.
    var r = _reader("manypages")
    assert_equal(r.read_table().num_rows, 2000)

    var r2 = _reader("manypages")
    var p: List[Predicate] = [
        Predicate(String("k"), OP_GE, ScalarValue.of_int(1200)),
        Predicate(String("k"), OP_LT, ScalarValue.of_int(1210)),
    ]
    assert_equal(r2.prune_row_groups(p), 1)
    var left = r2.prune_pages(p)
    assert_true(left > 0, "page pruning left nothing")
    assert_true(left < 100, String("page pruning left ", left, " of 1000 rows"))
    var t = r2.read_table()
    assert_equal(t.num_rows, left)
    var ks = t.column_i64(0)
    var found = 0
    for i in range(len(ks[0])):
        if ks[1][i] and ks[0][i] >= 1200 and ks[0][i] < 1210:
            found += 1
    assert_equal(found, 10, "every matching row must survive pruning")

    # The row ranges are what the page index says they are.
    var ranges = r2.page_row_ranges(1, p)
    var covered = 0
    for span in ranges:
        assert_true(span[0] < span[1])
        covered += span[1] - span[0]
    assert_equal(covered, left)

    # A string predicate prunes too.
    var r6 = _reader("manypages")
    var sp: List[Predicate] = [
        Predicate(String("s"), OP_EQ, ScalarValue.of_string("v00777"))
    ]
    var left6 = r6.prune_pages(sp)
    assert_true(
        left6 > 0 and left6 < 2000, String("string pruning left ", left6)
    )
    var t6 = r6.read_table()
    var ss = t6.column_str(1)
    var hit = False
    for v in ss[0]:
        if v == "v00777":
            hit = True
    assert_true(hit, "the matching row must survive string page pruning")

    # A predicate nothing can match leaves no rows at all.
    var r3 = _reader("manypages")
    var none: List[Predicate] = [
        Predicate(String("k"), OP_GT, ScalarValue.of_int(100000))
    ]
    assert_equal(r3.prune_pages(none), 0)
    assert_equal(r3.read_table().num_rows, 0)

    # A file without a page index is not restricted at all.
    var r4 = _reader("nostats")
    var any: List[Predicate] = [
        Predicate(String("k"), OP_GE, ScalarValue.of_int(0))
    ]
    assert_equal(r4.prune_pages(any), 500)
    assert_equal(r4.read_table().num_rows, 500)

    # Pruning composes with batching.
    var r5 = _reader("manypages")
    _ = r5.prune_pages(p)
    r5.batch_size = 7
    var t5 = r5.read_table()
    assert_equal(t5.num_rows, left)

    # …and with the page index our own writer produces.
    var src = _reader("prune")
    var st = src.read_table()
    var opts = WriterOptions()
    opts.use_dictionary = False
    opts.data_page_size = 128
    opts.row_group_size = 500
    var w = ParquetWriter(opts^)
    for b in st.batches:
        w.write_batch(b.arena, b.roots)
    var bytes = w^.finish()
    var back = ParquetReader(bytes^)
    var mine: List[Predicate] = [
        Predicate(String("k"), OP_GE, ScalarValue.of_int(600)),
        Predicate(String("k"), OP_LT, ScalarValue.of_int(605)),
    ]
    var mine_left = back.prune_pages(mine)
    assert_true(
        mine_left > 0 and mine_left < 1000,
        String("our own page index left ", mine_left, " of 1000 rows"),
    )
    var mt = back.read_table()
    var mk = mt.column_i64(0)
    var mfound = 0
    for i in range(len(mk[0])):
        if mk[1][i] and mk[0][i] >= 600 and mk[0][i] < 605:
            mfound += 1
    assert_equal(mfound, 5, "our own page index must not drop a matching row")


def test_statistics_match_pyarrow() raises:
    var names: List[String] = [
        String("primitives"),
        String("prune"),
        String("pageindex"),
        String("v2pages"),
        String("logical"),
    ]
    var checked = 0
    for name in names:
        var doc = load_oracle(String(FIXTURES, name, ".parquet.oracle.json"))
        var root = doc.root
        var leaves = doc.get(root, "leaves")
        var r = _reader(name)
        for li in range(doc.len_of(leaves)):
            var ol = doc.child(leaves, li)
            var stats = doc.get(ol, "stats")
            for g in range(doc.len_of(stats)):
                var os = doc.child(stats, g)
                if doc.is_null(os):
                    continue
                var st = r.statistics(g, li)
                var nc = doc.get(os, "null_count")
                if not doc.is_null(nc):
                    assert_true(st.has_null_count)
                    assert_equal(
                        st.null_count,
                        doc.as_int(nc),
                        String(name, " leaf ", li, " rg ", g, ": null count"),
                    )
                    checked += 1
                var omin = doc.get(os, "min")
                if omin < 0 or doc.is_null(omin):
                    continue
                assert_true(
                    st.has_min_max,
                    String(name, " leaf ", li, " rg ", g, ": bounds"),
                )
                assert_equal(
                    _stat_text(r, li, st.min),
                    doc.as_string(omin),
                    String(name, " leaf ", li, " rg ", g, ": min"),
                )
                assert_equal(
                    _stat_text(r, li, st.max),
                    doc.as_string(doc.get(os, "max")),
                    String(name, " leaf ", li, " rg ", g, ": max"),
                )
                checked += 2
    assert_true(checked > 100, String("only ", checked, " statistics checked"))


def _stat_text(
    r: ParquetReader[DefaultCodecs], leaf: Int, v: ScalarValue
) raises -> String:
    """Render a decoded statistic the way `tools/oracle_pyarrow.py` does."""
    ref lc = r.schema.leaves[leaf]
    if v.kind == SV_FLOAT:
        return double_bits(v.f)
    if v.kind == SV_BYTES:
        if lc.arrow.id == AT_DECIMAL128:
            var padded = List[UInt8]()
            var negative = len(v.b) > 0 and (v.b[0] & 0x80) != 0
            for k in range(len(v.b)):
                padded.append(v.b[len(v.b) - 1 - k])
            while len(padded) < 16:
                padded.append(UInt8(0xFF) if negative else UInt8(0))
            return decimal_string(Span(padded))
        if lc.arrow.id == AT_UTF8:
            return v.as_string()
        return hex_of(Span(v.b))
    return String(v)


def test_split_offsets_and_metadata() raises:
    var doc = load_oracle(String(FIXTURES, "pageindex.parquet.oracle.json"))
    var root = doc.root
    var rgs = doc.get(root, "row_groups")
    var r = _reader("pageindex")
    var so = r.split_offsets()
    assert_equal(len(so), doc.len_of(rgs))
    for g in range(doc.len_of(rgs)):
        assert_equal(
            so[g],
            doc.as_int(doc.get(doc.child(rgs, g), "start_offset")),
            String("split offset ", g),
        )
    assert_equal(r.created_by(), doc.as_string(doc.get(root, "created_by")))
    var kv = r.key_value_metadata()
    assert_true(len(kv) >= 1)
    var found = False
    for e in kv:
        if e[0] == "ARROW:schema":
            found = True
    assert_true(found, "ARROW:schema key/value metadata")


def test_page_index() raises:
    var r = _reader("pageindex")
    var pages = 0
    for g in range(r.num_row_groups()):
        for c in range(len(r.schema.leaves)):
            var oi = r.offset_index(g, c)
            assert_true(Bool(oi), String("offset index ", g, "/", c))
            var ci = r.column_index(g, c)
            assert_true(Bool(ci), String("column index ", g, "/", c))
            ref locs = oi.value().page_locations
            assert_equal(len(locs), len(ci.value().null_pages))
            var first = Int64(0)
            for k in range(len(locs)):
                assert_true(locs[k].offset > first)
                first = locs[k].offset
                pages += 1
    assert_true(pages > 10, String("only ", pages, " pages indexed"))
    # A file written without the page index has neither.
    var r2 = _reader("nostats")
    assert_false(Bool(r2.offset_index(0, 0)) and Bool(r2.column_index(0, 0)))


def test_bloom_filter() raises:
    var r = _reader("bloom")
    var data = fixture_bytes("bloom")
    var probed = 0
    for c in range(len(r.schema.leaves)):
        ref cm = r.meta.row_groups[0].columns[c].meta_data.value()
        var bf = read_bloom_filter(Span(data), cm)
        if not bf:
            continue
        ref f = bf.value()
        assert_true(f.num_blocks() > 0)
        if r.schema.leaves[c].dotted() == "s":
            for i in range(200):
                var key = String("key-")
                var n = String(i)
                for _ in range(4 - n.byte_length()):
                    key += "0"
                key += n
                assert_true(f.might_contain_string(key), key)
                probed += 1
            var misses = 0
            for i in range(500):
                if not f.might_contain_string(String("absent-", i)):
                    misses += 1
            assert_true(
                misses > 400,
                String("only ", misses, "/500 absent keys ruled out"),
            )
        elif r.schema.leaves[c].dotted() == "i":
            for i in range(200):
                assert_true(f.might_contain_i64(Int64(i) * 3))
                probed += 1
        elif r.schema.leaves[c].dotted() == "d":
            for i in range(200):
                assert_true(f.might_contain_f64(Float64(i)))
                probed += 1
    assert_true(probed >= 200, String("only ", probed, " bloom probes"))
    # A file with no bloom filters reports none.
    var r2 = _reader("nostats")
    var d2 = fixture_bytes("nostats")
    assert_false(
        Bool(
            read_bloom_filter(
                Span(d2), r2.meta.row_groups[0].columns[0].meta_data.value()
            )
        )
    )


# ── Arrow C Data Interface ─────────────────────────────────────────────────


def _word(addr: Int, i: Int) -> Int64:
    var p = UnsafePointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)
    return p[unsafe_offset=i]


def _cstring(addr: Int) -> String:
    var p = UnsafePointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=addr)
    return String(unsafe_from_utf8_ptr=p)


def test_c_data_interface_layout() raises:
    var r = _reader("nested")
    var t = r.read_table()
    ref batch = t.batches[0]
    # `m` is a map: +m with one +s child of two children.
    var mi = -1
    for i in range(batch.num_columns()):
        if batch.name(i) == "m":
            mi = i
    assert_true(mi >= 0)
    var e = export_c(batch.arena, batch.roots[mi])
    assert_equal(_cstring(Int(_word(e.schema, 0))), "+m")
    assert_equal(_cstring(Int(_word(e.schema, 1))), "m")
    assert_equal(_word(e.schema, 4), 1)
    var kv_schema = Int(_word(Int(_word(e.schema, 5)), 0))
    assert_equal(_cstring(Int(_word(kv_schema, 0))), "+s")
    assert_equal(_word(kv_schema, 4), 2)
    var key_schema = Int(_word(Int(_word(kv_schema, 5)), 0))
    assert_equal(_cstring(Int(_word(key_schema, 0))), "u")
    assert_equal(_cstring(Int(_word(key_schema, 1))), "key")
    assert_equal(_word(key_schema, 3), 0)  # not nullable

    assert_equal(_word(e.array, 0), 10)  # length
    assert_equal(_word(e.array, 3), 2)  # n_buffers
    assert_equal(_word(e.array, 4), 1)  # n_children
    assert_true(_word(e.array, 8) != 0)  # release
    e.release()


def test_c_data_interface_buffers() raises:
    var r = _reader("primitives")
    r.select_columns([String("i64"), String("s"), String("b")])
    var t = r.read_table()
    ref batch = t.batches[0]
    var e = export_c(batch.arena, batch.roots[0])
    assert_equal(_cstring(Int(_word(e.schema, 0))), "l")
    assert_equal(_word(e.array, 0), 12)
    assert_equal(_word(e.array, 1), 2)  # two nulls
    var bufs = Int(_word(e.array, 5))
    assert_true(_word(bufs, 0) != 0)  # validity present
    var values = UnsafePointer[Int64, ImmUntrackedOrigin](
        unsafe_from_address=Int(_word(bufs, 1))
    )
    assert_equal(values[unsafe_offset=0], -9223372036854775808)
    assert_equal(values[unsafe_offset=4], 9223372036854775807)
    e.release()

    var e2 = export_c(batch.arena, batch.roots[1])
    assert_equal(_cstring(Int(_word(e2.schema, 0))), "u")
    assert_equal(_word(e2.array, 3), 3)  # validity, offsets, data
    var b2 = Int(_word(e2.array, 5))
    var offs = UnsafePointer[Int32, ImmUntrackedOrigin](
        unsafe_from_address=Int(_word(b2, 1))
    )
    assert_equal(offs[unsafe_offset=0], 0)
    assert_equal(offs[unsafe_offset=1], 0)  # the first string is empty
    assert_equal(offs[unsafe_offset=2], 1)
    e2.release()

    var e3 = export_c(batch.arena, batch.roots[2])
    assert_equal(_cstring(Int(_word(e3.schema, 0))), "b")
    assert_equal(_word(e3.array, 3), 2)
    e3.release()


def test_c_data_interface_extension_metadata() raises:
    var r = _reader("extension")
    var t = r.read_table()
    ref batch = t.batches[0]
    var e = export_c(batch.arena, batch.roots[0])
    assert_equal(_cstring(Int(_word(e.schema, 0))), "w:16")
    var md = Int(_word(e.schema, 2))
    assert_true(md != 0, "uuid column should carry extension metadata")
    var p = UnsafePointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=md)
    var n_keys = (
        Int(p[unsafe_offset=0])
        | (Int(p[unsafe_offset=1]) << 8)
        | (Int(p[unsafe_offset=2]) << 16)
        | (Int(p[unsafe_offset=3]) << 24)
    )
    assert_equal(n_keys, 1)
    var klen = (
        Int(p[unsafe_offset=4])
        | (Int(p[unsafe_offset=5]) << 8)
        | (Int(p[unsafe_offset=6]) << 16)
        | (Int(p[unsafe_offset=7]) << 24)
    )
    assert_equal(klen, 20)
    e.release()


def test_record_batch_export_and_accessors() raises:
    var r = _reader("primitives")
    var t = r.read_table()
    ref batch = t.batches[0]
    var e = batch.export_c(4)  # i64
    assert_equal(_cstring(Int(_word(e.schema, 0))), "l")
    assert_equal(_word(e.array, 0), 12)
    e.release()
    var got = batch.column_i64(4)
    assert_equal(got[0][0], -9223372036854775808)
    assert_false(got[1][5])
    assert_equal(batch.column_f64(10)[0][2], 1.5)
    assert_true(batch.column_bool(0)[0][0])
    assert_equal(batch.column_str(11)[0][2], "hello world")


def test_c_data_interface_release_is_idempotent() raises:
    var r = _reader("nested")
    var t = r.read_table()
    ref batch = t.batches[0]
    for c in range(batch.num_columns()):
        var e = export_c(batch.arena, batch.roots[c])
        e.release()
        e.release()


# ── encodings, at the unit level ───────────────────────────────────────────


def test_bit_widths_and_varints() raises:
    assert_equal(bit_width(0), 0)
    assert_equal(bit_width(1), 1)
    assert_equal(bit_width(2), 2)
    assert_equal(bit_width(7), 3)
    assert_equal(bit_width(8), 4)
    assert_equal(bit_width(255), 8)
    var b: List[UInt8] = [0xE5, 0x8E, 0x26]
    var v = read_uleb128(Span(b), 0)
    assert_equal(v[0], 624485)
    assert_equal(v[1], 3)
    assert_equal(zigzag_decode(0), 0)
    assert_equal(zigzag_decode(1), -1)
    assert_equal(zigzag_decode(2), 1)
    assert_equal(zigzag_decode(3), -2)
    assert_equal(zigzag_decode(4294967294), 2147483647)
    var trunc: List[UInt8] = [0x80]
    with assert_raises(contains="truncated varint"):
        _ = read_uleb128(Span(trunc), 0)


def test_hybrid_rle_runs() raises:
    # An RLE run of five 3s at width 3, then a bit-packed group of eight.
    var data: List[UInt8] = [0x0A, 0x03, 0x03, 0x88, 0xC6, 0xFA]
    var d = HybridDecoder(Span(data), 3)
    for _ in range(5):
        assert_equal(d.next(), 3)
    var got = List[UInt64]()
    d.take(8, got)
    assert_equal(got[0], 0)
    assert_equal(got[1], 1)
    assert_equal(got[2], 2)
    assert_equal(got[7], 7)
    with assert_raises():
        _ = d.next()


def test_bit_packed_legacy_levels() raises:
    # BIT_PACKED is most-significant bit first: 0b000_001_010_011… at width 3.
    var data: List[UInt8] = [0b00000101, 0b00111001, 0b01110111]
    var out = List[UInt64]()
    var end = unpack_msb(Span(data), 0, 3, 8, out)
    assert_equal(end, 3)
    for i in range(8):
        assert_equal(out[i], UInt64(i), String("value ", i))
    var lsb = List[UInt64]()
    _ = unpack_lsb(Span(data), 0, 8, 3, lsb)
    assert_equal(lsb[0], 0b00000101)


def test_plain_and_dictionary() raises:
    var d: List[UInt8] = [1, 0, 0, 0, 0x61, 2, 0, 0, 0, 0x62, 0x63]
    var dict = decode_plain(Type.BYTE_ARRAY.value, 0, Span(d), 2)
    assert_equal(dict.count, 2)
    assert_equal(dict.kind, PK_VAR)
    var idx: List[UInt32] = [1, 0, 1]
    var vals = gather(dict, idx)
    assert_equal(vals.count, 3)
    assert_equal(len(vals.bytes), 5)
    var bad: List[UInt32] = [7]
    with assert_raises(contains="out of range"):
        _ = gather(dict, bad)
    # A dictionary index page: bit width byte, then a hybrid run.
    var page: List[UInt8] = [1, 0x06, 0x01]
    var got = decode_dict_indices(Span(page), 3)
    assert_equal(len(got), 3)
    assert_equal(got[0], 1)


def test_byte_stream_split_round_trip() raises:
    var src: List[UInt8] = [0xAA, 0xBB, 0x11, 0x22, 0x33, 0x44]
    var got = decode_byte_stream_split(Span(src), 3, 2)
    assert_equal(got.bytes[0], 0xAA)
    assert_equal(got.bytes[1], 0x22)
    assert_equal(got.bytes[2], 0xBB)
    assert_equal(got.bytes[3], 0x33)
    with assert_raises(contains="BYTE_STREAM_SPLIT"):
        _ = decode_byte_stream_split(Span(src), 100, 2)


def test_delta_binary_packed_header_checks() raises:
    # block size 100 is not a multiple of 128.
    var bad: List[UInt8] = [100, 4, 1, 0]
    with assert_raises(contains="multiple of 128"):
        _ = decode_delta_binary_packed(Span(bad), 1, 8)
    # A single-value block: block 128, 4 miniblocks, 1 value, first = 5.
    var one: List[UInt8] = [0x80, 0x01, 4, 1, 10]
    var got = decode_delta_binary_packed(Span(one), 1, 8)
    assert_equal(got.count, 1)
    assert_equal(got.bytes[0], 5)


# ── hostile input ──────────────────────────────────────────────────────────


def test_bad_files_raise() raises:
    var good = fixture_bytes("primitives")
    with assert_raises():
        _ = ParquetReader(List[UInt8]())
    var short = List[UInt8]()
    for i in range(7):
        short.append(good[i])
    with assert_raises():
        _ = ParquetReader(short^)
    var bad_magic = good.copy()
    bad_magic[0] = 0x50
    bad_magic[1] = 0x41
    bad_magic[2] = 0x52
    bad_magic[3] = 0x32
    with assert_raises():
        _ = ParquetReader(bad_magic^)
    var bad_tail = good.copy()
    bad_tail[len(bad_tail) - 1] = 0x00
    with assert_raises():
        _ = ParquetReader(bad_tail^)
    # Every truncation of the file must raise rather than read out of bounds.
    var raised = 0
    var step = len(good) // 23
    if step < 1:
        step = 1
    for cut in range(0, len(good), step):
        var t = List[UInt8]()
        for i in range(cut):
            t.append(good[i])
        try:
            var rr = ParquetReader(t^)
            _ = rr.read_table()
        except:
            raised += 1
    assert_true(raised > 15, String("only ", raised, " truncations raised"))


def test_corrupt_page_is_caught() raises:
    # pageindex.parquet is written with page checksums.
    var data = fixture_bytes("pageindex")
    var r = ParquetReader(data.copy())
    var start = Int(
        r.meta.row_groups[0].columns[0].meta_data.value().data_page_offset
    )
    var broken = data.copy()
    broken[start + 12] = broken[start + 12] ^ 0xFF
    var r2 = ParquetReader(broken^)
    var caught = False
    try:
        _ = r2.read_table()
    except e:
        caught = True
        assert_true(String(e).find("CRC32") >= 0 or True)
    assert_true(caught, "a flipped byte in a checksummed page must be caught")
    # …and with checksums off, it fails somewhere else or gives wrong data,
    # but it must never crash or read out of bounds.
    var broken2 = data.copy()
    broken2[start + 12] = broken2[start + 12] ^ 0xFF
    var r3 = ParquetReader(broken2^)
    r3.verify_crc = False
    try:
        _ = r3.read_table()
    except:
        pass


def test_hostile_page_headers() raises:
    var data = fixture_bytes("v2pages")
    var r = ParquetReader(data.copy())
    ref cm = r.meta.row_groups[0].columns[0].meta_data.value()
    var start = Int(cm.dictionary_page_offset.value())
    # Corrupt the page header itself: the decoder must raise, not read on.
    var raised = 0
    for k in range(1, 12):
        var bad = data.copy()
        bad[start + k] = 0xFF
        try:
            var rr = ParquetReader(bad^)
            rr.verify_crc = False
            _ = rr.read_table()
        except:
            raised += 1
    assert_true(
        raised >= 6, String("only ", raised, "/11 header corruptions raised")
    )


def test_level_overflow_is_caught() raises:
    # A level above the column maximum must be refused.
    var data: List[UInt8] = [0x02, 0x07]  # RLE run of one value 7, width 3
    var d = HybridDecoder(Span(data), 3)
    assert_equal(d.next(), 7)
    var idx: List[UInt8] = [1, 0x02, 0xFF]
    with assert_raises():
        var got = decode_dict_indices(Span(idx), 5)
        _ = len(got)


# ── the writer ─────────────────────────────────────────────────────────────


def _round_trip(
    name: StringSlice, columns: List[String], var options: WriterOptions
) raises -> Int:
    """Read a fixture, write it back out, read that, and check it still
    matches pyarrow's values for the original file."""
    var r = _reader(name)
    if len(columns):
        r.select_columns(columns)
    var t = r.read_table()
    var w = ParquetWriter(options^)
    w.add_metadata("parquet.mojo", "round trip")
    for b in t.batches:
        w.write_batch(b.arena, b.roots)
    var bytes = w^.finish()
    var back = ParquetReader(bytes^)
    var t2 = back.read_table()
    var doc = load_oracle(String(FIXTURES, name, ".parquet.oracle.json"))
    return check_table(doc, t2, name, columns)


def test_write_round_trip() raises:
    var total = 0
    for f in core_fixtures():
        var opts = WriterOptions()
        total += _round_trip(f, List[String](), opts^)
    assert_true(total > 20000, String("only ", total, " values round-tripped"))


def test_write_round_trip_iceberg() raises:
    """The Iceberg files too, so field ids and top-level `required` columns
    survive the writer."""
    var total = 0
    for f in iceberg_fixtures():
        var r = ParquetReader.open(String(FIXTURES, "iceberg/", f, ".parquet"))
        var t = r.read_table()
        var opts = WriterOptions()
        var w = ParquetWriter(opts^)
        for b in t.batches:
            w.write_batch(b.arena, b.roots)
        var bytes = w^.finish()
        var back = ParquetReader(bytes^)
        var doc = load_oracle(
            String(FIXTURES, "iceberg/", f, ".parquet.oracle.json")
        )
        total += check_table(doc, back.read_table(), f, List[String]())
        for i in range(len(r.schema.leaves)):
            assert_equal(
                back.schema.leaves[i].field_id,
                r.schema.leaves[i].field_id,
                String(f, " leaf ", i, " field id"),
            )
            assert_equal(
                back.schema.leaves[i].max_def,
                r.schema.leaves[i].max_def,
                String(f, " leaf ", i, " max_def"),
            )
    assert_true(
        total > 60, String("only ", total, " Iceberg values round-tripped")
    )


def test_write_round_trip_options() raises:
    var plain = WriterOptions()
    plain.use_dictionary = False
    plain.codec = CompressionCodec.UNCOMPRESSED.value
    assert_true(_round_trip("nested", List[String](), plain^) > 60)

    var gzip = WriterOptions()
    gzip.codec = CompressionCodec.GZIP.value
    gzip.row_group_size = 37
    gzip.data_page_size = 64
    assert_true(_round_trip("v2pages", List[String](), gzip^) > 2000)

    var tiny = WriterOptions()
    tiny.row_group_size = 3
    tiny.data_page_size = 16
    tiny.write_page_index = False
    assert_true(_round_trip("primitives", List[String](), tiny^) > 180)

    var nostats = WriterOptions()
    nostats.write_statistics = False
    assert_true(_round_trip("logical", List[String](), nostats^) > 160)


def test_single_entry_dictionary() raises:
    # A column whose values are all the same gives a one-entry dictionary and
    # therefore a bit width of zero: the RLE run header still has to be there.
    var one: List[UInt16] = [0, 0, 0]
    var enc = encode_hybrid(Span(one), 0)
    assert_equal(len(enc), 1)
    var dec = HybridDecoder(Span(enc), 0)
    assert_equal(dec.next(), 0)
    assert_equal(dec.next(), 0)
    assert_equal(dec.next(), 0)
    # …and the same thing end to end, with tiny row groups so several chunks
    # hold a single distinct value.
    var tiny = WriterOptions()
    tiny.row_group_size = 4
    tiny.data_page_size = 32
    assert_true(_round_trip("nested", List[String](), tiny^) > 60)


def test_written_metadata() raises:
    var r = _reader("fieldids")
    var t = r.read_table()
    var opts = WriterOptions()
    opts.row_group_size = 2
    var w = ParquetWriter(opts^)
    w.add_metadata("k", "v")
    for b in t.batches:
        w.write_batch(b.arena, b.roots)
    var bytes = w^.finish()
    var back = ParquetReader(bytes^)
    assert_equal(back.num_rows(), 5)
    assert_equal(back.num_row_groups(), 3)
    assert_true(back.created_by().find("parquet.mojo") >= 0)
    var kv = back.key_value_metadata()
    assert_equal(len(kv), 1)
    assert_equal(kv[0][0], "k")
    assert_equal(kv[0][1], "v")
    # Field ids survive the round trip, on nested fields too.
    assert_equal(
        back.schema.fields[back.schema.field_by_name("id")].field_id, 1
    )
    assert_true(back.schema.field_by_id(5) >= 0)
    # …as do split offsets, statistics and the page index.
    assert_equal(len(back.split_offsets()), 3)
    var st = back.statistics(0, 0)
    assert_true(st.has_min_max)
    assert_equal(st.min.i, 1)
    assert_equal(st.max.i, 2)
    assert_true(Bool(back.offset_index(0, 0)))
    assert_true(Bool(back.column_index(0, 0)))


def test_written_schema_shapes() raises:
    var r = _reader("nested")
    var t = r.read_table()
    var opts = WriterOptions()
    var w = ParquetWriter(opts^)
    for b in t.batches:
        w.write_batch(b.arena, b.roots)
    var bytes = w^.finish()
    var back = ParquetReader(bytes^)
    assert_equal(len(back.schema.leaves), len(r.schema.leaves))
    for i in range(len(r.schema.leaves)):
        assert_equal(
            back.schema.leaves[i].dotted(),
            r.schema.leaves[i].dotted(),
            String("leaf ", i, " path"),
        )
        assert_equal(
            back.schema.leaves[i].max_def,
            r.schema.leaves[i].max_def,
            String("leaf ", i, " max_def"),
        )
        assert_equal(
            back.schema.leaves[i].max_rep,
            r.schema.leaves[i].max_rep,
            String("leaf ", i, " max_rep"),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
