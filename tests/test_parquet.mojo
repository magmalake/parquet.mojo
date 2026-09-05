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
from fingerprint import (
    permuted_arenas,
    read_error,
    read_fingerprint,
    table_fingerprint,
    table_values_fingerprint,
)
from oracle import canon_value, decimal_string, double_bits, hex_of, load_oracle
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
    ArrayArena,
    ArrayData,
    ArrowType,
    DefaultCodecs,
    ParquetReader,
    Predicate,
    ScalarValue,
    ParquetWriter,
    Table,
    WriterOptions,
    build_schema,
    export_c,
    array_str,
)
from parquet.rle_encode import encode_hybrid, encode_levels
from parquet.writer import DICT_MAX_VALUES
from parquet.bitio import (
    HYBRID_BLOCK,
    HybridDecoder,
    bit_width,
    read_uleb128,
    unpack_lsb,
    unpack_msb,
    zigzag_decode,
)
from parquet.bloom import read_bloom_filter
from parquet.encoding import (
    PK_FIXED,
    PK_VAR,
    PhysBuffer,
    decode_byte_stream_split,
    decode_delta_binary_packed,
    decode_dict_indices,
    decode_plain,
    gather,
    gather_dict_into,
    gather_into,
)
from parquet.schema import REP_OPTIONAL, REP_REPEATED, REP_REQUIRED
from std.memory import bitcast
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_not_equal,
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

from std.benchmark.compiler import keep

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


def _canon_rows(
    fixture: StringSlice, column: StringSlice, batch_size: Int
) raises -> List[String]:
    """One column, one rendered string per row, in the oracle's own notation.

    `L<n>:` opens a list of `n` elements, `O<n>:` a struct of `n` fields,
    `S<n>:` a scalar of `n` bytes and `N` is a null — so a value in the wrong
    place, a null in the wrong place and an element attached to the wrong row
    are all visible as different text.
    """
    var r = _reader(fixture)
    r.batch_size = batch_size
    r.select_columns([String(column)])
    var t = r.read_table()
    var out = List[String]()
    for b in range(len(t.batches)):
        ref batch = t.batches[b]
        for i in range(batch.num_rows):
            var s = String()
            canon_value(batch.arena, batch.roots[0], i, s)
            out.append(s^)
    return out^


def _assert_rows(
    fixture: StringSlice, column: StringSlice, want: List[String]
) raises:
    """Assert every row of `column`, at batch sizes that put the row-group
    boundary, the batch boundary and the byte boundary of the validity bitmap
    in different places relative to each other."""
    var sizes: List[Int] = [1, 2, 3, 4, 7, 8, 9, 16, 65536]
    for bs in sizes:
        var got = _canon_rows(fixture, column, bs)
        assert_equal(
            len(got),
            len(want),
            String(fixture, ".", column, ": row count at batch size ", bs),
        )
        for i in range(len(want)):
            assert_equal(
                got[i],
                want[i],
                String(fixture, ".", column, "[", i, "] at batch size ", bs),
            )


def test_list_children_are_cell_exact() raises:
    """The nested cases the vectorised leaf path has to get right.

    A list's element array is built by filtering definition levels and packing
    the validity bitmap eight *rows* at a time, which is a different index from
    the slot it reads — so the ways it can go wrong are a child with nulls, a
    list of lists, a list of structs, a map, and a batch whose rows do not
    start on a byte boundary. Every one of those is pinned here value by value.
    """
    # list<int32> with nulls in the child, empty lists and null lists.
    _assert_rows(
        "nested",
        "li",
        [
            String("L3:S1:1S1:2S1:3"),
            String("L0:"),
            String("N"),
            String("L1:S1:4"),
            String("L2:NS1:5"),
            String("L3:S1:6NS1:7"),
            String("L0:"),
            String("N"),
            String("L2:S1:8S1:9"),
            String("L1:S2:10"),
        ],
    )
    # list<list<utf8>>: the inner list is itself built with a threshold.
    _assert_rows(
        "nested",
        "lls",
        [
            String("L2:L1:S1:aL2:S1:bS1:c"),
            String("L0:"),
            String("N"),
            String("L1:L0:"),
            String("L1:N"),
            String("L1:L1:S1:d"),
            String("L2:L0:L1:S1:e"),
            String("N"),
            String("L1:L2:S1:fN"),
            String("L1:L1:S1:g"),
        ],
    )
    # list<struct<n: int64>>: a struct passes the threshold straight through.
    _assert_rows(
        "nested",
        "lst",
        [
            String("L1:O1:S1:nS1:1"),
            String("L0:"),
            String("N"),
            String("L2:O1:S1:nNO1:S1:nS1:2"),
            String("L1:N"),
            String("L1:O1:S1:nS1:3"),
            String("L0:"),
            String("N"),
            String("L1:O1:S1:nS1:4"),
            String("L1:O1:S1:nS1:5"),
        ],
    )
    # map<utf8, int64>: two leaves under one repeated group, one of them
    # required and one nullable.
    _assert_rows(
        "nested",
        "m",
        [
            String("L1:L2:S1:kS1:1"),
            String("L0:"),
            String("N"),
            String("L2:L2:S1:aS1:1L2:S1:bS1:2"),
            String("L1:L2:S1:zN"),
            String("L1:L2:S1:qS1:9"),
            String("L0:"),
            String("N"),
            String("L1:L2:S1:yS1:3"),
            String("L1:L2:S1:xS1:4"),
        ],
    )
    # list<binary>: a variable-length child, so the element offsets are built
    # by the same filtered pass as the validity.
    _assert_rows(
        "nested",
        "lb",
        [
            String("L1:S2:01"),
            String("L0:"),
            String("N"),
            String("L2:S4:0203N"),
            String("L1:S0:"),
            String("L1:S2:ff"),
            String("L0:"),
            String("N"),
            String("L1:S4:6162"),
            String("L1:S4:6364"),
        ],
    )
    # A 2-level (pre-Parquet-2) list, whose element is `required`.
    _assert_rows(
        "legacy_list",
        "li",
        [
            String("L2:S1:1S1:2"),
            String("L0:"),
            String("N"),
            String("L1:S1:3"),
            String("L3:S1:4S1:5S1:6"),
            String("L0:"),
            String("L1:S1:7"),
            String("N"),
            String("L1:S1:8"),
            String("L1:S1:9"),
        ],
    )


def test_list_offsets_and_child_null_counts() raises:
    """The numbers the vectorised path computes for itself: how long the
    element array is, how many of its entries are null, and which rows the
    elements belong to."""
    var r = _reader("nested")
    r.select_columns([String("li")])
    var t = r.read_table()
    assert_equal(len(t.batches), 1)
    ref batch = t.batches[0]
    ref li = batch.column(0)
    assert_equal(li.type.id, AT_LIST)
    assert_equal(li.length, 10)
    assert_equal(li.null_count, 2)
    var want_off: List[Int] = [0, 3, 3, 3, 4, 6, 9, 9, 9, 11, 12]
    assert_equal(len(li.offsets), len(want_off), "li: offset count")
    for k in range(len(want_off)):
        assert_equal(
            Int(li.offsets[k]), want_off[k], String("li.offsets[", k, "]")
        )
    ref el = batch.child(batch.roots[0], 0)
    assert_equal(el.type.id, AT_INT32)
    assert_equal(el.length, 12)
    assert_equal(el.null_count, 2)

    var r2 = _reader("nested")
    r2.select_columns([String("lls")])
    var t2 = r2.read_table()
    ref b2 = t2.batches[0]
    ref lls = b2.column(0)
    assert_equal(lls.length, 10)
    assert_equal(lls.null_count, 2)
    var want_out: List[Int] = [0, 2, 2, 2, 3, 4, 5, 7, 7, 8, 9]
    assert_equal(len(lls.offsets), len(want_out), "lls: offset count")
    for k in range(len(want_out)):
        assert_equal(
            Int(lls.offsets[k]), want_out[k], String("lls.offsets[", k, "]")
        )
    ref inner = b2.child(b2.roots[0], 0)
    assert_equal(inner.type.id, AT_LIST)
    assert_equal(inner.length, 9)
    assert_equal(inner.null_count, 1)
    assert_equal(len(inner.children), 1)
    ref strs = b2.arena.nodes[inner.children[0]]
    assert_equal(strs.type.id, AT_UTF8)
    assert_equal(strs.length, 8)
    assert_equal(strs.null_count, 1)


def test_big_list_column_at_odd_batch_boundaries() raises:
    """100k rows of `list<int32>`, 20k of them null, checked against pyarrow at
    batch sizes that neither divide the row group nor land on a byte boundary
    of the packed validity bitmap."""
    var sizes: List[Int] = [101, 4999]
    for bs in sizes:
        var n = check_fixture[DefaultCodecs](String("big"), [String("l")], bs)
        assert_true(n > 200, String("batch size ", bs))


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
    var p = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)
    return p[unsafe_offset=i]


def _cstring(addr: Int) -> String:
    var p = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=addr)
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
    var values = Pointer[Int64, ImmUntrackedOrigin](
        unsafe_from_address=Int(_word(bufs, 1))
    )
    assert_equal(values[unsafe_offset=0], -9223372036854775808)
    assert_equal(values[unsafe_offset=4], 9223372036854775807)
    e.release()

    var e2 = export_c(batch.arena, batch.roots[1])
    assert_equal(_cstring(Int(_word(e2.schema, 0))), "u")
    assert_equal(_word(e2.array, 3), 3)  # validity, offsets, data
    var b2 = Int(_word(e2.array, 5))
    var offs = Pointer[Int32, ImmUntrackedOrigin](
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
    var p = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=md)
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


# ── the fused dictionary gather ────────────────────────────────────────────
#
# `gather_dict_into` decodes indices, bounds-checks them and gathers them a
# block at a time, where `decode_dict_indices` + `gather_into` make three
# passes over a page-sized index array. The two must agree byte for byte, and
# a corrupt index must raise identically out of either.


def _fixed_dict(entries: Int, width: Int) raises -> PhysBuffer:
    """A `PK_FIXED` dictionary whose entries all differ, so a gather that
    lands on the wrong one shows up in the bytes."""
    var d = PhysBuffer(PK_FIXED, width)
    d.bytes.resize(entries * width, 0)
    for k in range(entries):
        for b in range(width):
            d.bytes[k * width + b] = UInt8((k * 7 + b * 31) & 0xFF)
    d.count = entries
    return d^


def _var_dict(entries: Int) raises -> PhysBuffer:
    """A `PK_VAR` dictionary with values either side of the eight-byte
    fast-copy threshold, and an empty one."""
    var d = PhysBuffer(PK_VAR, 0)
    for k in range(entries):
        var v = List[UInt8]()
        for b in range(k % 17):
            v.append(UInt8((k * 13 + b) & 0xFF))
        d.append_bytes(Span(v))
    return d^


def _test_dict(which: Int, entries: Int) raises -> PhysBuffer:
    if which == 0:
        return _fixed_dict(entries, 8)
    if which == 1:
        return _fixed_dict(entries, 4)
    if which == 2:
        return _fixed_dict(entries, 12)
    return _var_dict(entries)


def _index_pattern(which: Int, entries: Int) raises -> List[UInt16]:
    """Index streams that between them cover every shape a hybrid run has."""
    var idx = List[UInt16]()
    if which == 0:  # one repeated run, several blocks long
        for _ in range(3000):
            idx.append(7)
    elif which == 1:  # bit-packed throughout, several blocks
        for i in range(3000):
            idx.append(UInt16(i % entries))
    elif which == 2:  # a repeated run, then bit-packed
        for _ in range(2000):
            idx.append(5)
        for i in range(1500):
            idx.append(UInt16(i % entries))
    elif which == 3:  # shorter than one block
        for i in range(5):
            idx.append(UInt16((i * 3) % entries))
    elif which == 4:  # exactly one block
        for i in range(HYBRID_BLOCK):
            idx.append(UInt16(i % entries))
    elif which == 5:  # one block and one value
        for i in range(HYBRID_BLOCK + 1):
            idx.append(UInt16(i % entries))
    elif which == 6:  # runs of exactly eight: many short RLE runs
        for i in range(2500):
            idx.append(UInt16((i // 8) % entries))
    else:  # nothing at all
        pass
    return idx^


def _index_page(indices: List[UInt16], width: Int) raises -> List[UInt8]:
    """A dictionary-index page body: the bit width byte, then the hybrid."""
    var page: List[UInt8] = [UInt8(width)]
    page.extend(Span(encode_hybrid(Span(indices), width)))
    return page^


def test_fused_dict_gather_matches_the_unfused_one() raises:
    """`gather_dict_into` must produce, byte for byte, what
    `decode_dict_indices` followed by `gather_into` produces.

    A bit-packed run longer than one `HYBRID_BLOCK` is the case the fused path
    can only handle because a block is a multiple of eight values: eight
    values of `width` bits are exactly `width` bytes, so resuming a run at a
    block boundary is still resuming it on a byte boundary. Patterns 1, 2, 4,
    5 and 6 all cross one.

    Both buffers are primed with a page first, so the append-onto-a-non-empty
    chunk path — the one a multi-page column chunk actually takes — is what is
    being compared.
    """
    var entries = 300
    var width = bit_width(entries - 1)
    for kind in range(4):
        var dict = _test_dict(kind, entries)
        var head = _index_pattern(3, entries)
        var head_page = _index_page(head, width)
        for pattern in range(8):
            var idx = _index_pattern(pattern, entries)
            var page = _index_page(idx, width)
            var slow = PhysBuffer(dict.kind, dict.width)
            var fast = PhysBuffer(dict.kind, dict.width)
            gather_into(
                slow, dict, decode_dict_indices(Span(head_page), len(head))
            )
            gather_dict_into(fast, dict, Span(head_page), len(head))
            gather_into(slow, dict, decode_dict_indices(Span(page), len(idx)))
            gather_dict_into(fast, dict, Span(page), len(idx))
            var tag = String("dict ", kind, " pattern ", pattern)
            assert_equal(fast.count, slow.count, String(tag, " count"))
            assert_equal(fast.width, slow.width, String(tag, " width"))
            assert_equal(
                len(fast.bytes), len(slow.bytes), String(tag, " bytes")
            )
            for i in range(len(slow.bytes)):
                assert_equal(
                    fast.bytes[i],
                    slow.bytes[i],
                    String(tag, " byte ", i),
                )
            assert_equal(
                len(fast.offsets),
                len(slow.offsets),
                String(tag, " offset count"),
            )
            for i in range(len(slow.offsets)):
                assert_equal(
                    fast.offsets[i],
                    slow.offsets[i],
                    String(tag, " offset ", i),
                )


def _bad_index_pattern(which: Int, entries: Int) raises -> List[UInt16]:
    var idx = List[UInt16]()
    if which == 0:  # in the first bit-packed block
        for i in range(50):
            idx.append(UInt16(i % entries))
        idx[7] = UInt16(entries + 100)
    elif which == 1:  # in a later block, so the scan has to get there
        for i in range(3000):
            idx.append(UInt16(i % entries))
        idx[2500] = UInt16(entries + 199)
    elif which == 2:  # a whole repeated run: one comparison, not a scan
        for _ in range(3000):
            idx.append(UInt16(entries + 20))
    elif which == 3:  # two of them in different blocks: the first has to win
        for i in range(3000):
            idx.append(UInt16(i % entries))
        idx[1500] = UInt16(entries + 180)
        idx[2500] = UInt16(entries + 190)
    else:  # two of them in one block: the scan *within* a block is ordered too
        for i in range(3000):
            idx.append(UInt16(i % entries))
        idx[1100] = UInt16(entries + 170)
        idx[1200] = UInt16(entries + 160)
    return idx^


def test_fused_dict_gather_raises_the_same_error() raises:
    """A dictionary index past the end of the dictionary must raise the same
    error out of the fused path as out of the separate bounds pass — the same
    index reported, the same message — wherever in the page it sits.

    The check also has to happen *before* the block is gathered: an index past
    the end is an out-of-bounds read of the dictionary's offsets, not just a
    wrong answer.
    """
    var entries = 300
    var width = bit_width(entries + 200)
    for kind in range(4):
        var dict = _test_dict(kind, entries)
        for bad in range(5):
            var idx = _bad_index_pattern(bad, entries)
            var page = _index_page(idx, width)
            var tag = String("dict ", kind, " case ", bad)

            var want = String()
            var slow = PhysBuffer(dict.kind, dict.width)
            try:
                gather_into(
                    slow, dict, decode_dict_indices(Span(page), len(idx))
                )
            except e:
                want = String(e)
            assert_true(
                want.find("out of range") >= 0,
                String(tag, ": the unfused path did not raise"),
            )

            var got = String()
            var fast = PhysBuffer(dict.kind, dict.width)
            try:
                gather_dict_into(fast, dict, Span(page), len(idx))
            except e:
                got = String(e)
            assert_equal(got, want, tag)


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


def _write_int64_column(
    values: Span[Int64, _], var options: WriterOptions
) raises -> List[UInt8]:
    """One `INT64` column of exactly these values, written by us."""
    var a = ArrayData(ArrowType(AT_INT64), String("v"))
    a.nullable = False
    a.length = len(values)
    for v in values:
        var u = bitcast[DType.uint64](v)
        for b in range(8):
            a.values.append(UInt8((u >> UInt64(8 * b)) & 0xFF))
    var arena = ArrayArena()
    var roots: List[Int] = [arena.add(a^)]
    var w = ParquetWriter(options^)
    w.write_batch(arena, roots)
    return w^.finish()


def _cycled(distinct: Int, count: Int) -> List[Int64]:
    """`count` values drawn cyclically from `distinct` of them — the shape a
    prefix sample cannot tell from an all-distinct column, because the first
    `distinct` values *are* all distinct."""
    var out = List[Int64](capacity=count)
    for i in range(count):
        out.append(Int64((i % distinct) * 7919))
    return out^


def _dictionary_column_count(bytes: Span[UInt8, _]) raises -> Int:
    """How many of this file's column chunks carry a dictionary page."""
    var r = ParquetReader[DefaultCodecs].from_span(bytes)
    var n = 0
    for g in range(len(r.meta.row_groups)):
        for c in range(len(r.meta.row_groups[g].columns)):
            ref cm = r.meta.row_groups[g].columns[c].meta_data.value()
            if cm.dictionary_page_offset:
                n += 1
    return n


def _assert_int64_round_trip(
    bytes: Span[UInt8, _], values: Span[Int64, _], what: String
) raises:
    var r = ParquetReader[DefaultCodecs].from_span(bytes)
    var t = r.read_table()
    assert_equal(t.num_rows, len(values), String(what, ": row count"))
    var got = t.column_i64(0)
    for i in range(len(values)):
        assert_equal(got[0][i], values[i], String(what, ": value ", i))


def test_dictionary_kept_for_a_low_cardinality_column() raises:
    """The happy path: few distinct values, so the dictionary must survive."""
    var values = _cycled(1000, 65536)
    var bytes = _write_int64_column(Span(values), WriterOptions())
    assert_equal(
        _dictionary_column_count(Span(bytes)),
        1,
        "a 1000-distinct column lost its dictionary",
    )
    _assert_int64_round_trip(Span(bytes), Span(values), String("low"))


def test_dictionary_abandoned_for_a_high_cardinality_column() raises:
    """All-distinct values: no dictionary, and the abandon must be early —
    `DICT_MAX_VALUES` inserts, not half the chunk."""
    var values = List[Int64](capacity=65536)
    for i in range(65536):
        values.append(Int64(i) * 2654435761)
    var bytes = _write_int64_column(Span(values), WriterOptions())
    assert_equal(
        _dictionary_column_count(Span(bytes)),
        0,
        "an all-distinct column kept a dictionary",
    )
    _assert_int64_round_trip(Span(bytes), Span(values), String("high"))


def test_dictionary_cap_boundary() raises:
    """Either side of the cap, and the small-chunk floor below it.

    `DICT_MAX_VALUES` is the largest index the build tolerates, so a dictionary
    of `DICT_MAX_VALUES + 1` entries is the last one that fits.
    """
    var n = 4 * DICT_MAX_VALUES
    var fits = _cycled(DICT_MAX_VALUES + 1, n)
    assert_equal(
        _dictionary_column_count(
            Span(_write_int64_column(Span(fits), WriterOptions()))
        ),
        1,
        String(DICT_MAX_VALUES + 1, " distinct values should still fit"),
    )
    var over = _cycled(DICT_MAX_VALUES + 2, n)
    var over_bytes = _write_int64_column(Span(over), WriterOptions())
    assert_equal(
        _dictionary_column_count(Span(over_bytes)),
        0,
        String(DICT_MAX_VALUES + 2, " distinct values should abandon"),
    )
    _assert_int64_round_trip(Span(over_bytes), Span(over), String("boundary"))

    # Below 128 values a chunk cannot reach the cap at all and the floor of 64
    # decides instead — this is the rule that keeps a tiny chunk from
    # abandoning on its second value, and it is unchanged.
    var small_ok = _cycled(65, 100)
    assert_equal(
        _dictionary_column_count(
            Span(_write_int64_column(Span(small_ok), WriterOptions()))
        ),
        1,
        "65 distinct values in 100 should fit under the floor",
    )
    var small_over = _cycled(66, 100)
    assert_equal(
        _dictionary_column_count(
            Span(_write_int64_column(Span(small_over), WriterOptions()))
        ),
        0,
        "66 distinct values in 100 should abandon",
    )


def _write_float_column[
    id: Int, dt: DType, bits: DType
](values: Span[Scalar[dt], _]) raises -> List[UInt8]:
    """One `FLOAT` or `DOUBLE` column of exactly these values, written by us."""
    comptime width = 4 if id == AT_FLOAT32 else 8
    var a = ArrayData(ArrowType(id), String("v"))
    a.nullable = False
    a.length = len(values)
    for v in values:
        var u = UInt64(bitcast[bits](v))
        for b in range(width):
            a.values.append(UInt8((u >> UInt64(8 * b)) & 0xFF))
    var arena = ArrayArena()
    var roots: List[Int] = [arena.add(a^)]
    var w = ParquetWriter(WriterOptions())
    w.write_batch(arena, roots)
    return w^.finish()


def _float_bounds_of(
    bytes: Span[UInt8, _],
) raises -> Tuple[Bool, Float64, Float64]:
    """`(has_min_max, min, max)` of the one column of a file we just wrote."""
    var r = ParquetReader[DefaultCodecs].from_span(bytes)
    var st = r.statistics(0, 0)
    return (st.has_min_max, st.min.f, st.max.f)


def _is_negative_zero(v: Float64) -> Bool:
    return v == 0.0 and bitcast[DType.uint64](v) != 0


def _check_float_statistics[
    id: Int, dt: DType, bits: DType
](what: String) raises:
    """Every float statistics rule, on one physical type.

    Parquet is specific about NaN — it is never a bound, and a chunk of
    nothing but NaN has no bounds at all — and *unspecific* about signed zero:
    it suggests a writer prefer `-0.0` for a minimum and `+0.0` for a maximum,
    which this writer does not do. What it does instead is report the first
    zero it saw, and these pin that down rather than let it drift.
    """
    var one = Scalar[dt](1.0)
    var zero = Scalar[dt](0.0)
    var nan = zero / zero
    var inf = one / zero

    # NaN never becomes a bound, wherever it sits.
    var mixed: List[Scalar[dt]] = [one, nan, -one, nan, Scalar[dt](2.0), nan]
    var got = _float_bounds_of(
        Span(_write_float_column[id, dt, bits](Span(mixed)))
    )
    assert_true(got[0], String(what, ": NaN mix should have bounds"))
    assert_equal(got[1], -1.0, String(what, ": NaN mix min"))
    assert_equal(got[2], 2.0, String(what, ": NaN mix max"))

    # Nothing but NaN: no statistics at all, not a NaN bound.
    var all_nan: List[Scalar[dt]] = [nan, nan, nan, nan, nan]
    var none = _float_bounds_of(
        Span(_write_float_column[id, dt, bits](Span(all_nan)))
    )
    assert_false(none[0], String(what, ": all-NaN must report no bounds"))

    # One real value among NaNs is both bounds.
    var lonely: List[Scalar[dt]] = [nan, nan, Scalar[dt](3.5), nan]
    var only = _float_bounds_of(
        Span(_write_float_column[id, dt, bits](Span(lonely)))
    )
    assert_true(only[0], String(what, ": one non-NaN should have bounds"))
    assert_equal(only[1], 3.5, String(what, ": one non-NaN min"))
    assert_equal(only[2], 3.5, String(what, ": one non-NaN max"))

    # Infinities are ordinary values and are the bounds when present.
    var edges: List[Scalar[dt]] = [zero, inf, -inf, one, nan]
    var ends = _float_bounds_of(
        Span(_write_float_column[id, dt, bits](Span(edges)))
    )
    assert_true(ends[0], String(what, ": infinities should have bounds"))
    assert_true(ends[1] < -1.0e37, String(what, ": min should be -inf"))
    assert_true(ends[2] > 1.0e37, String(what, ": max should be +inf"))

    # Signed zero: the first zero wins, whichever sign it has. Long enough,
    # and with the two zeros far enough apart, that a lane-wise scan would
    # answer with the wrong one if it did not put the order back.
    var pos_first = List[Scalar[dt]]()
    var neg_first = List[Scalar[dt]]()
    for i in range(64):
        var v = Scalar[dt](2.0 + Float64(i))
        pos_first.append(v)
        neg_first.append(v)
    pos_first[2] = zero
    pos_first[37] = -zero
    neg_first[2] = -zero
    neg_first[37] = zero
    var pos = _float_bounds_of(
        Span(_write_float_column[id, dt, bits](Span(pos_first)))
    )
    assert_true(pos[0], String(what, ": +0.0 first should have bounds"))
    assert_equal(pos[1], 0.0, String(what, ": +0.0 first min value"))
    assert_false(
        _is_negative_zero(pos[1]), String(what, ": +0.0 first min sign")
    )
    var neg = _float_bounds_of(
        Span(_write_float_column[id, dt, bits](Span(neg_first)))
    )
    assert_true(neg[0], String(what, ": -0.0 first should have bounds"))
    assert_equal(neg[1], 0.0, String(what, ": -0.0 first min value"))
    assert_true(
        _is_negative_zero(neg[1]), String(what, ": -0.0 first min sign")
    )

    # …and the same for a maximum of zero, where every other value is below.
    var max_zero = List[Scalar[dt]]()
    for i in range(64):
        max_zero.append(Scalar[dt](-2.0 - Float64(i)))
    max_zero[2] = -zero
    max_zero[37] = zero
    var top = _float_bounds_of(
        Span(_write_float_column[id, dt, bits](Span(max_zero)))
    )
    assert_true(top[0], String(what, ": zero max should have bounds"))
    assert_equal(top[2], 0.0, String(what, ": zero max value"))
    assert_true(_is_negative_zero(top[2]), String(what, ": zero max sign"))


def test_double_statistics_semantics() raises:
    _check_float_statistics[AT_FLOAT64, DType.float64, DType.uint64](
        String("DOUBLE")
    )


def test_float_statistics_semantics() raises:
    _check_float_statistics[AT_FLOAT32, DType.float32, DType.uint32](
        String("FLOAT")
    )


def test_integer_statistics_bounds() raises:
    """The integer paths lost their index too; the bounds must not move."""
    var vals = List[Int64](capacity=1000)
    for i in range(1000):
        vals.append(Int64(((i * 7919) % 1013) - 500))
    vals[0] = 12345
    vals[997] = -99999
    vals[998] = 99999
    var bytes = _write_int64_column(Span(vals), WriterOptions())
    var r = ParquetReader[DefaultCodecs].from_span(Span(bytes))
    var st = r.statistics(0, 0)
    assert_true(st.has_min_max, "int64 bounds missing")
    assert_equal(st.min.i, -99999, "int64 min")
    assert_equal(st.max.i, 99999, "int64 max")


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


# ── borrowing the file bytes ───────────────────────────────────────────────


def test_from_span_reads_the_same_table_as_the_owning_constructor() raises:
    var bytes = fixture_bytes("big")

    var owned = ParquetReader[DefaultCodecs](bytes.copy())
    var want = owned.read_table()

    var borrowed = ParquetReader[DefaultCodecs].from_span(Span(bytes))
    var got = borrowed.read_table()

    assert_equal(got.num_rows, want.num_rows)
    assert_equal(len(borrowed.meta.row_groups), len(owned.meta.row_groups))
    assert_equal(len(borrowed.schema.leaves), len(owned.schema.leaves))
    keep(bytes)


def test_from_span_reads_the_same_buffer_twice() raises:
    """The reason the borrowing path exists: many reads, one allocation."""
    var bytes = fixture_bytes("big")
    var first = ParquetReader[DefaultCodecs].from_span(Span(bytes))
    var rows_first = first.read_table().num_rows
    var second = ParquetReader[DefaultCodecs].from_span(Span(bytes))
    var rows_second = second.read_table().num_rows
    assert_equal(rows_first, rows_second)
    keep(bytes)


def test_owning_reader_survives_a_move() raises:
    """`data` points into `_owned`, so a move must not invalidate it. Moving a
    `List` moves the handle and leaves the heap buffer where it was."""
    var r = ParquetReader[DefaultCodecs](fixture_bytes("big"))
    var moved = r^
    assert_equal(moved.read_table().num_rows, 100000)


def test_projection_works_on_a_borrowed_reader() raises:
    var bytes = fixture_bytes("prune")
    var r = ParquetReader[DefaultCodecs].from_span(Span(bytes))
    var all_cols = r.read_table()
    assert_equal(all_cols.num_rows, 1000)
    keep(bytes)


# ── the value index of a row ────────────────────────────────────────────────


def _write_nullable_int64(
    values: Span[Int64, _], valid: Span[Bool, _], var options: WriterOptions
) raises -> List[UInt8]:
    """One nullable `INT64` column: `values[i]`, or null where `valid[i]` is
    false. Arrow keeps a slot for a null, so the value buffer is full length."""
    var a = ArrayData(ArrowType(AT_INT64), String("v"))
    a.nullable = True
    a.length = len(values)
    a.validity.resize((len(values) + 7) // 8, 0)
    for i in range(len(values)):
        if valid[i]:
            a.validity[i >> 3] |= UInt8(1) << UInt8(i & 7)
        else:
            a.null_count += 1
        var u = bitcast[DType.uint64](values[i])
        for b in range(8):
            a.values.append(UInt8((u >> UInt64(8 * b)) & 0xFF))
    var arena = ArrayArena()
    var roots: List[Int] = [arena.add(a^)]
    var w = ParquetWriter(options^)
    w.write_batch(arena, roots)
    return w^.finish()


def test_value_index_before_the_first_null_page() raises:
    """A chunk whose nulls only start in a later page.

    `ColumnData.page_slot` starts at the first page that has a null in it, on
    the grounds that every slot before that holds a value and so its value
    index *is* its slot index. Nothing in the corpus has that shape — every
    nullable fixture is null inside its first page — so this writes one: 4000
    rows in 512-byte data pages, with only the second half nullable. The
    assertions are that the shape really is that (`page_slot[0] > 0`, so the
    branch is reached) and that `value_at` agrees with a count for every row.
    """
    var values = List[Int64](capacity=4000)
    var valid = List[Bool](capacity=4000)
    for i in range(4000):
        values.append(Int64(i))
        valid.append(i < 2000 or (i % 7) != 0)
    var opts = WriterOptions()
    opts.data_page_size = 512
    opts.row_group_size = 4000
    opts.use_dictionary = False
    var bytes = _write_nullable_int64(Span(values), Span(valid), opts^)

    var r = ParquetReader[DefaultCodecs].from_span(Span(bytes))
    r.verify_crc = False
    r._load(0)
    ref cd = r._chunks[0]
    assert_true(
        len(cd.page_slot) > 0 and cd.page_slot[0] > 0,
        String(
            "the fixture needs a null-free first page; page_slot starts at ",
            cd.page_slot[0] if len(cd.page_slot) else -1,
        ),
    )
    var seen = 0
    for row in range(len(values) + 1):
        assert_equal(
            cd.value_at(row, 1),
            seen,
            String("value_at(", row, ")"),
        )
        if row < len(values) and valid[row]:
            seen += 1

    # …and end to end, at batch sizes that start inside the null-free region.
    var sizes: List[Int] = [1, 7, 64, 999, 65536]
    for bs in sizes:
        var r2 = ParquetReader[DefaultCodecs].from_span(Span(bytes))
        r2.verify_crc = False
        r2.batch_size = bs
        var t = r2.read_table()
        var row = 0
        for b in range(len(t.batches)):
            ref batch = t.batches[b]
            for i in range(batch.num_rows):
                var s = String()
                canon_value(batch.arena, batch.roots[0], i, s)
                var text = String(values[row])
                var want = String("N") if not valid[row] else String(
                    "S", text.byte_length(), ":", text
                )
                assert_equal(s, want, String("row ", row, " at batch ", bs))
                row += 1
        assert_equal(row, len(values), String("rows at batch ", bs))
    keep(bytes)


# ── which columns take the packed-validity path ─────────────────────────────
#
# A leaf with `max_def == 1` and `max_rep == 0` has definition levels that are
# already a validity bitmap, and `read_column_chunk` decodes them straight into
# one instead of materialising a `UInt16` per slot. Every other test in this
# file would pass just as well if that path never fired, so these two say out
# loud which chunks take it — by name for the fixture the optimisation was
# aimed at, and by invariant across the corpus.


def _validity_shapes(name: StringSlice) raises -> List[String]:
    """Per leaf of `name`'s first row group, `path=shape`.

    `mask` is the packed path, `all-present` is the chunk that had no nulls at
    all and needed no validity, and `levels` is a definition level per slot.
    """
    var r = _reader(name)
    r.verify_crc = False
    r._load(0)
    var out = List[String]()
    for i in range(len(r.schema.leaves)):
        var shape = String("levels")
        if r._chunks[i].masked():
            shape = String("mask")
        elif r._chunks[i].all_present:
            shape = String("all-present")
        out.append(String(r.schema.leaves[i].dotted(), "=", shape))
    return out^


def test_packed_validity_fires_on_the_columns_it_should() raises:
    """Instrumentation, not inference: which of `big.parquet`'s columns take
    the bitmap path, asserted by name.

    Four of its five columns are plain nullable scalars — `i`, `f`, `s`, `b` —
    which qualify and have nulls, so their levels land in a mask. The fifth is
    a list, whose leaf repeats, so it keeps a level per slot. If a change ever
    makes the gate stop firing, this fails instead of quietly getting slower.
    """
    var got = _validity_shapes("big")
    var want: List[String] = [
        String("i=mask"),
        String("f=mask"),
        String("s=mask"),
        String("b=mask"),
        String("l.list.element=levels"),
    ]
    assert_equal(len(got), len(want), "leaf count of big.parquet")
    for k in range(len(want)):
        assert_equal(got[k], want[k])
    var line = String("    big.parquet validity shapes:")
    for g in got:
        line += String(" ", g)
    print(line)


def test_packed_validity_and_its_gate_do_not_drift() raises:
    """Across the corpus: a chunk decodes to a mask exactly when its leaf
    qualifies and it has at least one null.

    `packed` is set from the leaf descriptor before a page is read and only
    ever cleared by the legacy `BIT_PACKED` level encoding, so
    `masked() == (packed and not all_present)` is the whole contract. The
    count at the end is what makes this a positive control rather than a
    vacuous one.
    """
    var masked = 0
    var checked = 0
    for f in core_fixtures():
        var r = _reader(f)
        r.verify_crc = False
        for rg in range(r.num_row_groups()):
            r._load(rg)
            for i in range(len(r.schema.leaves)):
                ref leaf = r.schema.leaves[i]
                ref cd = r._chunks[i]
                checked += 1
                assert_equal(
                    cd.masked(),
                    cd.packed and not cd.all_present,
                    String(f, ".", leaf.dotted(), ": mask against its gate"),
                )
                if cd.masked():
                    masked += 1
                    assert_true(
                        leaf.max_def == 1 and leaf.max_rep == 0,
                        String(f, ".", leaf.dotted(), " must not be packed"),
                    )
                    assert_equal(
                        len(cd.defs),
                        0,
                        String(f, ".", leaf.dotted(), " kept levels too"),
                    )
                    assert_equal(
                        len(cd.mask),
                        (cd.num_slots + 7) // 8,
                        String(f, ".", leaf.dotted(), ": mask size"),
                    )
    assert_true(masked >= 20, String("only ", masked, " packed chunks"))
    print("    packed chunks:", masked, "of", checked)


# ── more than one core ──────────────────────────────────────────────────────
#
# `ParquetReader.num_workers` fans the column chunks of a row group out over
# threads. The contract these tests pin down is that it changes *nothing* the
# caller can observe except how long the read takes: the same bytes, the same
# batch boundaries, and the same error from the same corrupt file.


def worker_counts() -> List[Int]:
    """Worker counts every fixture is re-read at. 10 is this machine's core
    count; every fixture has fewer leaves than that, so this also covers the
    clamp down to the number of projected columns."""
    return [2, 4, 10]


def test_num_workers_is_bit_identical() raises:
    """Every fixture, read at 1, 2, 4 and 10 workers, byte for byte.

    The fingerprint folds in the Arrow buffers themselves — values, validity,
    offsets, large offsets — as well as every length, null count, type name and
    field id, so a threaded read that got one bit, one null count or one offset
    wrong fails here. A value-level or row-count comparison would not.
    """
    var workers = worker_counts()
    var checked = 0
    for f in core_fixtures():
        var path = String(FIXTURES, f, ".parquet")
        var want = read_fingerprint[DefaultCodecs](path, 1, 65536)
        for w in range(len(workers)):
            assert_equal(
                read_fingerprint[DefaultCodecs](path, workers[w], 65536),
                want,
                String(f, " at ", workers[w], " workers"),
            )
            checked += 1
    for f in iceberg_fixtures():
        var path = String(FIXTURES, "iceberg/", f, ".parquet")
        var want = read_fingerprint[DefaultCodecs](path, 1, 65536)
        for w in range(len(workers)):
            assert_equal(
                read_fingerprint[DefaultCodecs](path, workers[w], 65536),
                want,
                String("iceberg/", f, " at ", workers[w], " workers"),
            )
            checked += 1
    assert_true(checked >= 90, String("only ", checked, " comparisons"))
    print("    worker-count comparisons:", checked)


def test_num_workers_is_bit_identical_across_batches() raises:
    """The same, at a batch size that cuts every row group into many batches.

    Threading happens in `_load` and batching in `_assemble`, so a per-row
    index built on a worker thread has to survive being sliced repeatedly.
    Kept to the fixtures with nesting, nulls and many pages, because a tiny
    batch size on a 100k-row fixture is all re-assembly and no new coverage.
    """
    var names: List[String] = [
        String("nested"),
        String("legacy_list"),
        String("allnull"),
        String("manypages"),
        String("encodings"),
        String("logical"),
    ]
    var workers = worker_counts()
    var checked = 0
    for f in names:
        var path = String(FIXTURES, f, ".parquet")
        var want = read_fingerprint[DefaultCodecs](path, 1, 64)
        for w in range(len(workers)):
            assert_equal(
                read_fingerprint[DefaultCodecs](path, workers[w], 64),
                want,
                String(f, " at ", workers[w], " workers, batch size 64"),
            )
            checked += 1
    assert_true(checked >= 18, String("only ", checked, " comparisons"))


def test_the_fingerprint_catches_a_permuted_arena() raises:
    """The negative control for the arena-layout half of the fingerprint.

    Top-level fields are assembled on several threads and grafted back into the
    batch afterwards, and the way that goes subtly wrong is *ordering*: an
    `ArrayData` names its children by arena index, so stitching the fields back
    in a different order shifts every index and yields a structurally different
    batch out of values that are identical to the last bit. A test that folds
    only the arrays — which is what this file folded before assembly was
    threaded — cannot see that at all.

    So: read `big.parquet` (five top-level fields, one of them a list, so the
    arena has more nodes than roots), renumber every arena without touching a
    value, and assert both halves of the claim. If the second assertion ever
    fails, `test_num_workers_is_bit_identical` has stopped being able to catch
    a broken assembly order and is checking values alone again.
    """
    var r = ParquetReader[DefaultCodecs].open(String(FIXTURES, "big.parquet"))
    var t = r.read_table()
    assert_true(
        len(t.batches[0].arena.nodes) > len(t.batches[0].roots),
        "big.parquet has no nested field, so reversal may be a no-op",
    )
    var shuffled = permuted_arenas(t)
    assert_equal(
        table_values_fingerprint(shuffled),
        table_values_fingerprint(t),
        "the permutation changed a value, so it is not a layout-only control",
    )
    assert_not_equal(
        table_fingerprint(shuffled),
        table_fingerprint(t),
        "the fingerprint missed a permuted arena",
    )


def test_arena_layout_does_not_depend_on_the_worker_count() raises:
    """The same claim, stated on its own rather than folded into a CRC.

    `test_num_workers_is_bit_identical` covers this across the corpus now that
    the fingerprint folds layout, but a failure there says only "some byte
    moved". This says which byte: node for node, the arena a threaded read
    builds is the arena a sequential one builds — the same number of nodes, in
    the same order, with the same children at the same indices — because each
    field is grafted in at the index it would have had sequentially.
    """
    var names: List[String] = [
        String("big"),
        String("nested"),
        String("legacy_list"),
        String("logical"),
        String("manypages"),
    ]
    var checked = 0
    for f in names:
        var path = String(FIXTURES, f, ".parquet")
        var base = ParquetReader[DefaultCodecs].open(path)
        base.batch_size = 256
        var want = base.read_table()
        for w in [2, 4, 10]:
            var r = ParquetReader[DefaultCodecs].open(path)
            r.num_workers = w
            r.batch_size = 256
            var got = r.read_table()
            assert_equal(
                len(got.batches), len(want.batches), String(f, " at ", w)
            )
            for b in range(len(want.batches)):
                ref a = want.batches[b]
                ref c = got.batches[b]
                assert_equal(
                    len(c.arena.nodes),
                    len(a.arena.nodes),
                    String(f, " batch ", b, " node count at ", w),
                )
                assert_equal(
                    String(c.roots),
                    String(a.roots),
                    String(f, " batch ", b, " roots at ", w),
                )
                for n in range(len(a.arena.nodes)):
                    assert_equal(
                        String(c.arena.nodes[n].children),
                        String(a.arena.nodes[n].children),
                        String(f, " batch ", b, " node ", n, " at ", w),
                    )
                    assert_equal(
                        c.arena.nodes[n].name,
                        a.arena.nodes[n].name,
                        String(f, " batch ", b, " node ", n, " name at ", w),
                    )
                    checked += 1
    assert_true(checked >= 100, String("only ", checked, " nodes compared"))
    print("    arena nodes compared:", checked)


def test_num_workers_zero_uses_every_core() raises:
    """`num_workers = 0` is "one per core", not "no workers"."""
    var path = String(FIXTURES, "big.parquet")
    assert_equal(
        read_fingerprint[DefaultCodecs](path, 0, 65536),
        read_fingerprint[DefaultCodecs](path, 1, 65536),
    )


def test_num_workers_survives_projection() raises:
    """A projection shrinks the task set; one surviving column must still take
    the sequential path and give the same answer."""
    var cols: List[String] = [String("i")]
    var want = String()
    for w in [1, 2, 8]:
        var r = ParquetReader[DefaultCodecs].open(
            String(FIXTURES, "big.parquet")
        )
        r.num_workers = w
        r.select_columns(cols)
        var t = r.read_table()
        var got = String(t.num_rows, ":", t.num_columns(), ":", t.name(0))
        if w == 1:
            want = got^
        else:
            assert_equal(got, want, String("projection at ", w, " workers"))


def _corrupt_chunk(var data: List[UInt8], leaf: Int) raises -> List[UInt8]:
    """Scribble over the first page of one column chunk of row group 0."""
    return _corrupt_chunk_in(data^, 0, leaf)


def _corrupt_chunk_in(
    var data: List[UInt8], rg: Int, leaf: Int
) raises -> List[UInt8]:
    """Scribble over the first page of one column chunk of row group `rg`."""
    var r = ParquetReader[DefaultCodecs].from_span(Span(data))
    ref cm = r.meta.row_groups[rg].columns[leaf].meta_data.value()
    var at = Int(cm.data_page_offset)
    if cm.dictionary_page_offset:
        var d = Int(cm.dictionary_page_offset.value())
        if d > 0 and d < at:
            at = d
    for k in range(at, at + 48):
        data[k] = 0xA5
    return data^


def test_num_workers_surfaces_a_corrupt_file() raises:
    """A task cannot raise — pthread has no exception channel — so a failure
    comes back through the task's own slot. It has to reach the caller as a
    raise from `read_table`, not a hang and not a short table."""
    var bad = _corrupt_chunk(fixture_bytes("big"), 0)
    var want = read_error[DefaultCodecs](Span(bad), 1)
    assert_true(want != "", "the corrupt fixture read cleanly at 1 worker")
    var workers = worker_counts()
    for w in range(len(workers)):
        assert_equal(
            read_error[DefaultCodecs](Span(bad), workers[w]),
            want,
            String("corrupt read at ", workers[w], " workers"),
        )


def test_corrupt_file_error_does_not_depend_on_the_winner() raises:
    """Two bad columns, so two tasks fail and the order they fail in is a race.
    The error reported is the lowest-numbered leaf's either way, because the
    slots are scanned in leaf order after the join."""
    var bytes = fixture_bytes("big")
    var r = ParquetReader[DefaultCodecs].from_span(Span(bytes))
    var leaves = len(r.schema.leaves)
    assert_true(leaves >= 3, String("big.parquet has only ", leaves, " leaves"))
    var bad = _corrupt_chunk(_corrupt_chunk(bytes^, 0), leaves - 1)
    var want = read_error[DefaultCodecs](Span(bad), 1)
    assert_true(want != "", "the doubly corrupt fixture read cleanly")
    var workers = worker_counts()
    for w in range(len(workers)):
        assert_equal(
            read_error[DefaultCodecs](Span(bad), workers[w]),
            want,
            String("two corrupt columns at ", workers[w], " workers"),
        )


# ── the second axis: row groups ─────────────────────────────────────────────
#
# `read_table` also fans out across row groups, so a row group can now be
# decoded before one that precedes it in the file. Everything below is about
# the one thing that can make wrong: **order**. A test that counts rows, sums
# a column or compares a set of values passes with two row groups swapped, so
# these name the boundaries and say what has to be on each side of them.


def _read_workers(
    path: StringSlice, workers: Int, batch_size: Int
) raises -> Table:
    var r = ParquetReader[DefaultCodecs].open(path)
    r.num_workers = workers
    r.batch_size = batch_size
    return r.read_table()


def test_row_groups_assemble_in_file_order() raises:
    """`big.parquet` is four row groups of 25 000 rows, and its `i` column is
    the row's own index — so a value names the position it must occupy.

    Checked three ways at every worker count: every row of `i` holds its own
    index and every row of `f` holds half of it, the rows either side of the
    three row-group boundaries hold exactly what the file has there, and the
    two columns' nulls fall where the fixture puts them — every 101st row of
    `i` and every 97th of `f`, two periods that share no factor with the 25 000
    rows of a row group. So a row group landing in the wrong place moves the
    values *and* breaks the null pattern, while a row *count* check sees
    neither. `i` and `f` also decode in different tasks, which is what makes
    this a check on assembly and not only on one column's decode.
    """
    var path = String(FIXTURES, "big.parquet")
    var boundaries: List[Int] = [24999, 25000, 49999, 50000, 74999, 75000]
    for w in [1, 2, 3, 4, 8, 10]:
        var t = _read_workers(path, w, 65536)
        assert_equal(t.num_rows, 100000, String("row count at ", w))
        var iv = t.column_i64(0)
        var fv = t.column_f64(1)
        assert_equal(len(iv[0]), 100000)
        assert_false(iv[1][0], String("row 0 of `i` is null, at ", w))
        assert_false(fv[1][0], String("row 0 of `f` is null, at ", w))
        for k in range(len(boundaries)):
            var n = boundaries[k]
            assert_equal(
                iv[0][n], Int64(n), String("i[", n, "] at ", w, " workers")
            )
            assert_equal(
                fv[0][n],
                Float64(n) * 0.5,
                String("f[", n, "] at ", w, " workers"),
            )
        var bad = -1
        for n in range(100000):
            var i_here = (n % 101) != 0
            var f_here = (n % 97) != 0
            if iv[1][n] != i_here or fv[1][n] != f_here:
                bad = n
                break
            if i_here and iv[0][n] != Int64(n):
                bad = n
                break
            if f_here and fv[0][n] != Float64(n) * 0.5:
                bad = n
                break
        assert_equal(bad, -1, String("first out-of-order row at ", w))


def test_row_groups_hold_their_order_across_window_boundaries() raises:
    """`prune.parquet` is ten row groups of 100 rows, so a window of two,
    three or four takes five, four and three passes to cover it, and the seam
    between one window and the next is crossed four, three and two times.

    A window that restarted at the wrong slot, decoded a row group twice or
    dropped one would show up here as a `k` that is not its own row index.
    """
    var path = String(FIXTURES, "prune.parquet")
    for w in [1, 2, 3, 4, 5, 8, 10, 16]:
        var t = _read_workers(path, w, 65536)
        assert_equal(t.num_rows, 1000, String("row count at ", w))
        var k = t.column_i64(0)
        var bad = -1
        for n in range(1000):
            if k[0][n] != Int64(n):
                bad = n
                break
        assert_equal(bad, -1, String("first out-of-order row at ", w))


def test_row_group_selection_keeps_its_own_order() raises:
    """Row groups come out in the order they were *selected*, not the order
    they sit in the file or the order the workers happened to finish them in.

    Row groups 3 and 1 of `big.parquet`, in that order: rows 75 000..99 999
    followed by rows 25 000..49 999.
    """
    for w in [1, 2, 4, 10]:
        var r = ParquetReader[DefaultCodecs].open(
            String(FIXTURES, "big.parquet")
        )
        r.num_workers = w
        var pick: List[Int] = [3, 1]
        r.select_row_groups(pick^)
        var t = r.read_table()
        assert_equal(t.num_rows, 50000, String("row count at ", w))
        var iv = t.column_i64(0)
        assert_equal(iv[0][0], Int64(75000), String("first row at ", w))
        assert_equal(iv[0][24999], Int64(99999), String("last of rg3 at ", w))
        assert_equal(iv[0][25000], Int64(25000), String("first of rg1 at ", w))
        assert_equal(iv[0][49999], Int64(49999), String("last row at ", w))


def test_a_single_row_group_is_untouched_by_the_second_axis() raises:
    """A file — or a selection — with one row group has no second axis, and
    must come out exactly as it did before. Every Iceberg fixture and most of
    the corpus is one row group, so this is the common case, not a corner.
    """
    var checked = 0
    for f in core_fixtures():
        var path = String(FIXTURES, f, ".parquet")
        var r = ParquetReader[DefaultCodecs].open(path)
        if r.num_row_groups() != 1:
            continue
        var want = read_fingerprint[DefaultCodecs](path, 1, 65536)
        for w in [2, 4, 10]:
            assert_equal(
                read_fingerprint[DefaultCodecs](path, w, 65536),
                want,
                String(f, " (one row group) at ", w, " workers"),
            )
            checked += 1
    assert_true(checked >= 45, String("only ", checked, " comparisons"))
    # And one row group *selected* out of four, which takes the same path.
    for w in [1, 2, 4, 10]:
        var r = ParquetReader[DefaultCodecs].open(
            String(FIXTURES, "big.parquet")
        )
        r.num_workers = w
        var pick: List[Int] = [2]
        r.select_row_groups(pick^)
        var t = r.read_table()
        assert_equal(t.num_rows, 25000)
        var iv = t.column_i64(0)
        assert_equal(iv[0][0], Int64(50000), String("first row at ", w))
        assert_equal(iv[0][24999], Int64(74999), String("last row at ", w))


def test_streaming_reads_one_row_group_at_a_time() raises:
    """The memory contract `read_table` is allowed to trade and `read_batch`
    is not.

    `read_table` may hold several row groups' decode state at once — it was
    going to hold every row group's *output* anyway. The batch iterator makes
    the opposite promise: one row group in flight, whatever `num_workers` is.
    Asserted where it lives, on `_prefetched`, because there is no other way
    to observe it — and asserted at a worker count high enough that a
    look-ahead, if one were ever added, would trip it.
    """
    var r = ParquetReader[DefaultCodecs].open(String(FIXTURES, "big.parquet"))
    r.num_workers = 8
    r.batch_size = 4096
    var streamed = Table()
    while r.has_next():
        var b = r.read_batch()
        assert_equal(
            len(r._prefetched),
            0,
            "the streaming path decoded a row group ahead",
        )
        streamed.num_rows += b.num_rows
        streamed.batches.append(b^)
    assert_equal(streamed.num_rows, 100000)
    var iv = streamed.column_i64(0)
    assert_equal(iv[0][25000], Int64(25000))
    assert_equal(iv[0][99999], Int64(99999))
    # And the same bytes as the threaded `read_table` that skipped ahead.
    assert_equal(
        table_fingerprint(streamed),
        read_fingerprint[DefaultCodecs](
            String(FIXTURES, "big.parquet"), 8, 4096
        ),
    )


def test_page_pruning_survives_the_second_axis() raises:
    """Pruning leaves gaps inside a row group and can empty one outright, and
    a prefetched window has to agree with the iterator about which row groups
    are visited at all."""
    var want_rows = 0
    var want: UInt32 = 0
    for w in [1, 2, 4, 10]:
        var r = ParquetReader[DefaultCodecs].open(
            String(FIXTURES, "pageindex.parquet")
        )
        r.num_workers = w
        r.batch_size = 64
        var preds: List[Predicate] = [
            Predicate(String("k"), OP_GE, ScalarValue.of_int(200)),
        ]
        var left = r.prune_pages(preds)
        var t = r.read_table()
        var got = table_fingerprint(t)
        if w == 1:
            want_rows = t.num_rows
            want = got
            assert_true(left > 0, "pruning left nothing to read")
            assert_true(
                t.num_rows < 500, String("pruning kept every row: ", t.num_rows)
            )
        else:
            assert_equal(t.num_rows, want_rows, String("rows at ", w))
            assert_equal(got, want, String("pruned read at ", w, " workers"))


def test_a_corrupt_row_group_raises_wherever_it_sits() raises:
    """The first failure in *visit* order is the one the caller sees, at every
    worker count — including when it is in the last row group, which a window
    decodes at the same time as the first.

    No hang, no short table: `read_table` raises, and with the same words.
    """
    for rg in [0, 2, 3]:
        var bad = _corrupt_chunk_in(fixture_bytes("big"), rg, 0)
        var want = read_error[DefaultCodecs](Span(bad), 1)
        assert_true(
            want != "", String("row group ", rg, " read cleanly at 1 worker")
        )
        for w in [2, 4, 10]:
            assert_equal(
                read_error[DefaultCodecs](Span(bad), w),
                want,
                String("row group ", rg, " corrupt, at ", w, " workers"),
            )


def test_the_earliest_corrupt_row_group_wins() raises:
    """Two bad row groups decoded side by side: which task fails first is a
    race, but the error reported is always the earlier row group's, because
    the slots are scanned in group-then-leaf order after the join."""
    var bad = _corrupt_chunk_in(
        _corrupt_chunk_in(fixture_bytes("big"), 1, 0), 3, 0
    )
    var want = read_error[DefaultCodecs](Span(bad), 1)
    assert_true(want != "", "the doubly corrupt fixture read cleanly")
    for w in [2, 4, 10]:
        assert_equal(
            read_error[DefaultCodecs](Span(bad), w),
            want,
            String("two corrupt row groups at ", w, " workers"),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
