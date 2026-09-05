"""Decode and encode benchmarks.

    pixi run -e bench bench
    pixi run -e bench bench -- --json
    pixi run -e bench bench -- --only bench_read_big

The timed region is footer parse plus `read_table` -- everything from bytes to
Arrow. Reads go through `ParquetReader.from_span`, so the file bytes are
allocated once in setup and every iteration reads the same buffer. That is
what the borrowing constructor is for: the owning constructor would need a
fresh `List[UInt8]` per iteration, and on a 5 MB fixture that memcpy lands
inside the timer and inflates the number by 5-15%.

`tools/bench_pyarrow.py` prints the same figures for pyarrow with one thread,
so the README can quote both honestly. pyarrow hands its reader a
`pa.py_buffer` over the same bytes every time and does not copy either, which
is now also true here.

The `wide` benchmarks need `build/bench-wide.parquet`, which is generated
rather than checked in:

    python tools/bench_pyarrow.py --make

Without it the suite stops before running anything; `--skip bench_read_wide
bench_write_wide` if you would rather not build it.
"""

from bench import Benchmark, BenchSuite, Metric, keep

from parquet import DefaultCodecs, ParquetReader, ParquetWriter, WriterOptions
from thrift import CompressionCodec, read_parquet_file

comptime FIXTURES = "tests/fixtures/"
comptime WIDE = "build/bench-wide.parquet"


def _read_file(path: StringSlice) raises -> List[UInt8]:
    return read_parquet_file(String(path))


def _rows_in(data: List[UInt8]) raises -> Int:
    var r = ParquetReader[DefaultCodecs].from_span(Span(data))
    r.verify_crc = False
    return r.read_table().num_rows


# ── decode ──────────────────────────────────────────────────────────────────


def bench_read_big(mut b: Benchmark) raises:
    """100k rows, 5 mixed columns."""
    var data = _read_file(String(FIXTURES, "big.parquet"))
    b.throughput(Metric.elements(), _rows_in(data))

    @parameter
    def call() raises:
        var r = ParquetReader[DefaultCodecs].from_span(Span(data))
        r.verify_crc = False
        keep(r.read_table().num_rows)

    b.iter[call]()
    keep(data)


def bench_read_prune(mut b: Benchmark) raises:
    """1k rows, 3 columns."""
    var data = _read_file(String(FIXTURES, "prune.parquet"))
    b.throughput(Metric.elements(), _rows_in(data))

    @parameter
    def call() raises:
        var r = ParquetReader[DefaultCodecs].from_span(Span(data))
        r.verify_crc = False
        keep(r.read_table().num_rows)

    b.iter[call]()
    keep(data)


def bench_read_encodings(mut b: Benchmark) raises:
    """400 rows, 12 columns -- one per encoding."""
    var data = _read_file(String(FIXTURES, "encodings.parquet"))
    b.throughput(Metric.elements(), _rows_in(data))

    @parameter
    def call() raises:
        var r = ParquetReader[DefaultCodecs].from_span(Span(data))
        r.verify_crc = False
        keep(r.read_table().num_rows)

    b.iter[call]()
    keep(data)


def bench_read_v2pages(mut b: Benchmark) raises:
    """500 rows written with v2 data pages."""
    var data = _read_file(String(FIXTURES, "v2pages.parquet"))
    b.throughput(Metric.elements(), _rows_in(data))

    @parameter
    def call() raises:
        var r = ParquetReader[DefaultCodecs].from_span(Span(data))
        r.verify_crc = False
        keep(r.read_table().num_rows)

    b.iter[call]()
    keep(data)


def bench_read_wide(mut b: Benchmark) raises:
    """1M rows of int64 + double -- the one that behaves like real data."""
    var data = _read_file(WIDE)
    b.throughput(Metric.elements(), _rows_in(data))

    @parameter
    def call() raises:
        var r = ParquetReader[DefaultCodecs].from_span(Span(data))
        r.verify_crc = False
        keep(r.read_table().num_rows)

    b.iter[call]()
    keep(data)


# ── decode, on more than one core ───────────────────────────────────────────
#
# `ParquetReader.num_workers` fans the column chunks of a row group out over
# threads, so the ceiling is the number of projected columns: bench-wide has
# four and big.parquet five. `_w1` goes through the same helper as the rest,
# which makes it the honest baseline for the ladder — and comparing it with
# `bench_read_wide`/`bench_read_big` above shows what, if anything, the extra
# `num_workers` store and the refactored `_load` cost the sequential path.
#
# Run one ladder at a time when you want numbers to quote:
#
#     pixi run -e bench bench -- --only bench_read_wide_w1 bench_read_wide_w2 …


def _bench_read_workers(
    mut b: Benchmark, path: StringSlice, workers: Int
) raises:
    var data = _read_file(String(path))
    b.throughput(Metric.elements(), _rows_in(data))

    @parameter
    def call() raises:
        var r = ParquetReader[DefaultCodecs].from_span(Span(data))
        r.verify_crc = False
        r.num_workers = workers
        keep(r.read_table().num_rows)

    b.iter[call]()
    keep(data)


def bench_read_wide_w1(mut b: Benchmark) raises:
    _bench_read_workers(b, WIDE, 1)


def bench_read_wide_w2(mut b: Benchmark) raises:
    _bench_read_workers(b, WIDE, 2)


def bench_read_wide_w4(mut b: Benchmark) raises:
    _bench_read_workers(b, WIDE, 4)


def bench_read_wide_w8(mut b: Benchmark) raises:
    _bench_read_workers(b, WIDE, 8)


def bench_read_wide_w10(mut b: Benchmark) raises:
    _bench_read_workers(b, WIDE, 10)


def bench_read_big_w1(mut b: Benchmark) raises:
    _bench_read_workers(b, String(FIXTURES, "big.parquet"), 1)


def bench_read_big_w2(mut b: Benchmark) raises:
    _bench_read_workers(b, String(FIXTURES, "big.parquet"), 2)


def bench_read_big_w4(mut b: Benchmark) raises:
    _bench_read_workers(b, String(FIXTURES, "big.parquet"), 4)


def bench_read_big_w8(mut b: Benchmark) raises:
    _bench_read_workers(b, String(FIXTURES, "big.parquet"), 8)


def bench_read_big_w10(mut b: Benchmark) raises:
    _bench_read_workers(b, String(FIXTURES, "big.parquet"), 10)


# ── encode ──────────────────────────────────────────────────────────────────
#
# Decoding happens in setup, so what is timed is Arrow batches to Parquet
# bytes and nothing else.


def bench_write_wide(mut b: Benchmark) raises:
    var data = _read_file(WIDE)
    var r = ParquetReader[DefaultCodecs].from_span(Span(data))
    r.verify_crc = False
    var table = r.read_table()
    b.throughput(Metric.elements(), table.num_rows)

    @parameter
    def call() raises:
        var opts = WriterOptions()
        opts.codec = CompressionCodec.UNCOMPRESSED.value
        var w = ParquetWriter[DefaultCodecs](opts^)
        for batch in table.batches:
            w.write_batch(batch.arena, batch.roots)
        keep(w^.finish())

    b.iter[call]()
    keep(table)
    keep(data)


def bench_write_big(mut b: Benchmark) raises:
    var data = _read_file(String(FIXTURES, "big.parquet"))
    var r = ParquetReader[DefaultCodecs].from_span(Span(data))
    r.verify_crc = False
    var table = r.read_table()
    b.throughput(Metric.elements(), table.num_rows)

    @parameter
    def call() raises:
        var opts = WriterOptions()
        opts.codec = CompressionCodec.UNCOMPRESSED.value
        var w = ParquetWriter[DefaultCodecs](opts^)
        for batch in table.batches:
            w.write_batch(batch.arena, batch.roots)
        keep(w^.finish())

    b.iter[call]()
    keep(table)
    keep(data)


def _print_shape() raises:
    """File sizes and row counts, once, so a MB/s figure can be derived from
    the table without the benchmark computing one."""
    var names: List[String] = [
        String(FIXTURES, "big.parquet"),
        String(FIXTURES, "prune.parquet"),
        String(FIXTURES, "encodings.parquet"),
        String(FIXTURES, "v2pages.parquet"),
        String(WIDE),
    ]
    print("parquet.mojo benchmarks (the `_wN` ladders use N reader threads)")
    for i in range(len(names)):
        # A missing file is reported, not raised: the wide fixture is
        # generated rather than committed, and `--skip bench_read_wide
        # bench_write_wide` has to be able to run without it. A benchmark
        # that actually needs the file still fails when it is selected.
        try:
            var data = _read_file(names[i])
            print(
                "  ",
                names[i],
                ": ",
                _rows_in(data),
                " rows, ",
                len(data) // 1024,
                " KiB",
                sep="",
            )
        except:
            print(
                "  ",
                names[i],
                (
                    ": missing — generate it with"
                    " `python tools/bench_pyarrow.py --make`"
                ),
                sep="",
            )


def main() raises:
    _print_shape()
    # Three repetitions: the wide fixture is 1M rows and every body re-reads
    # its file once per phase.
    BenchSuite.run[__functions_in_module()](num_repetitions=3)
