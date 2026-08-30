"""Decode benchmarks. `pixi run bench`.

Reports rows/s and MB/s (of the *file*) for each fixture, single threaded.
`tools/bench_pyarrow.py` prints the same numbers for pyarrow with one thread,
so the README can quote both honestly.

The timed region is footer parse plus `read_table` — everything from bytes to
Arrow. `ParquetReader` takes ownership of the file bytes, so each repeat needs
its own copy; that copy is made *outside* the timer, because the pyarrow side
hands its reader a `pa.py_buffer` over the same bytes every time and does not
copy either.
"""

from parquet import DefaultCodecs, ParquetReader, ParquetWriter, WriterOptions
from thrift import CompressionCodec
from std.time import perf_counter_ns
from thrift import read_parquet_file


def _bench(path: StringSlice, label: StringSlice, repeats: Int) raises:
    var bytes = read_parquet_file(String(path))
    var size = len(bytes)
    var rows = 0
    var best = 1.0e30
    for _ in range(repeats):
        var buf = bytes.copy()
        var t0 = perf_counter_ns()
        var r = ParquetReader[DefaultCodecs](buf^)
        r.verify_crc = False
        var t = r.read_table()
        var t1 = perf_counter_ns()
        rows = t.num_rows
        var secs = Float64(t1 - t0) / 1e9
        if secs < best:
            best = secs
    print(
        String(
            label,
            ": ",
            rows,
            " rows, ",
            size // 1024,
            " KiB in ",
            Int(best * 1e6),
            " us -> ",
            Int(Float64(rows) / best),
            " rows/s, ",
            Int(Float64(size) / best / 1e6),
            " MB/s",
        )
    )


def _bench_write(path: StringSlice, label: StringSlice, repeats: Int) raises:
    """Read a file in, then time writing the same Arrow data back out."""
    var r = ParquetReader[DefaultCodecs](read_parquet_file(String(path)))
    r.verify_crc = False
    var t = r.read_table()
    var rows = t.num_rows
    var best = 1.0e30
    var size = 0
    for _ in range(repeats):
        var t0 = perf_counter_ns()
        var opts = WriterOptions()
        opts.codec = CompressionCodec.UNCOMPRESSED.value
        var w = ParquetWriter[DefaultCodecs](opts^)
        for b in t.batches:
            w.write_batch(b.arena, b.roots)
        var bytes = w^.finish()
        var t1 = perf_counter_ns()
        size = len(bytes)
        var secs = Float64(t1 - t0) / 1e9
        if secs < best:
            best = secs
    print(
        String(
            label,
            ": ",
            rows,
            " rows -> ",
            size // 1024,
            " KiB in ",
            Int(best * 1e6),
            " us -> ",
            Int(Float64(rows) / best),
            " rows/s, ",
            Int(Float64(size) / best / 1e6),
            " MB/s",
        )
    )


def main() raises:
    print("parquet.mojo decode benchmark (single threaded)\n")
    _bench(
        "tests/fixtures/big.parquet", "big.parquet (100k rows, 5 mixed cols)", 5
    )
    _bench(
        "tests/fixtures/prune.parquet", "prune.parquet (1k rows, 3 cols)", 20
    )
    _bench(
        "tests/fixtures/encodings.parquet",
        "encodings.parquet (400 rows, 12 cols)",
        50,
    )
    _bench(
        "tests/fixtures/v2pages.parquet",
        "v2pages.parquet (500 rows, v2 pages)",
        50,
    )
    var wide = "build/bench-wide.parquet"
    try:
        _bench(wide, "bench-wide.parquet (1M rows int64+double)", 3)
    except:
        print(
            "bench-wide.parquet: not built — run"
            " `python tools/bench_pyarrow.py --make` first"
        )
    print("\nparquet.mojo encode benchmark (single threaded)\n")
    try:
        _bench_write(wide, "bench-wide.parquet (1M rows int64/double/dict)", 3)
    except:
        print("bench-wide.parquet: not built")
    _bench_write(
        "tests/fixtures/big.parquet", "big.parquet (100k rows, 5 mixed cols)", 3
    )
