"""The optional codec set: ZSTD, BROTLI, LZ4_RAW and Hadoop-framed LZ4.

`pixi run -e codecs test-codecs` (nightly) or `-e codecs-stable` (Mojo 1.0.0).
These need `zstd.mojo`, `brotli.mojo` and `lz4.mojo` checked out next door and
their shims installed, which the `codecs` environment does.
"""

from fixtures_list import (
    core_fixtures,
    full_codec_columns,
    iceberg_fixtures,
    iceberg_zstd_fixtures,
)
from oracle import load_oracle
from parity import check_fixture, check_path, check_table
from parquet import ParquetReader, ParquetWriter, WriterOptions
from thrift import CompressionCodec
from parquet.ext_full import AllCodecs
from std.testing import TestSuite, assert_true


def test_all_codecs_read_every_column() raises:
    var n = check_fixture[AllCodecs](
        String("codecs"), full_codec_columns(), 65536
    )
    assert_true(n > 1800, String("only ", n, " values checked"))


def test_the_full_set_reads_everything_the_default_set_does() raises:
    var total = 0
    for f in core_fixtures():
        total += check_fixture[AllCodecs](f, List[String](), 65536)
    assert_true(total > 20000, String("only ", total, " values checked"))


def test_ffi_codec_columns_match_one_by_one() raises:
    """One column at a time, so a codec that decodes to plausible-but-wrong
    bytes cannot hide behind the others in the all-columns check above."""
    for name in [String("zstd"), String("lz4"), String("brotli")]:
        var only: List[String] = [name]
        assert_true(
            check_fixture[AllCodecs](String("codecs"), only, 65536) > 300,
            String("codec column ", name),
        )


def test_write_and_read_back_with_ffi_codecs() raises:
    var codecs: List[Int32] = [
        CompressionCodec.ZSTD.value,
        CompressionCodec.BROTLI.value,
        CompressionCodec.LZ4_RAW.value,
    ]
    var total = 0
    for c in codecs:
        var r = ParquetReader[AllCodecs].open("tests/fixtures/nested.parquet")
        var t = r.read_table()
        var opts = WriterOptions()
        opts.codec = c
        opts.row_group_size = 4
        var w = ParquetWriter[AllCodecs](opts^)
        for b in t.batches:
            w.write_batch(b.arena, b.roots)
        var bytes = w^.finish()
        var back = ParquetReader[AllCodecs](bytes^)
        var t2 = back.read_table()
        var doc = load_oracle("tests/fixtures/nested.parquet.oracle.json")
        total += check_table(doc, t2, "nested", List[String]())
    assert_true(total > 100, String("only ", total, " values round-tripped"))


def test_iceberg_zstd_data_files() raises:
    """The PyIceberg-written Iceberg fixtures use ZSTD."""
    var total = 0
    for f in iceberg_zstd_fixtures():
        total += check_path[AllCodecs](
            String("tests/fixtures/iceberg/", f), f, List[String](), 65536
        )
    for f in iceberg_fixtures():
        total += check_path[AllCodecs](
            String("tests/fixtures/iceberg/", f), f, List[String](), 65536
        )
    assert_true(total > 80, String("only ", total, " Iceberg values checked"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
