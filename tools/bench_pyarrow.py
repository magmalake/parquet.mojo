#!/usr/bin/env python3
"""The same decode benchmark in pyarrow, with one thread and with all of them.

    python tools/bench_pyarrow.py --make    # write build/bench-wide.parquet
    python tools/bench_pyarrow.py           # time pyarrow on each fixture

Two legs, reported side by side:

- **1 thread** — `set_cpu_count(1)`, `set_io_thread_count(1)`, `use_threads=
  False`. This is the leg the README's tables have always quoted, and it is
  the one to compare against a `num_workers = 1` parquet.mojo read.
- **threaded** — pyarrow's own default CPU count with `use_threads=True`. This
  is what pyarrow does out of the box, and it is the leg to compare against a
  `num_workers = N` read.

Percentiles, not a minimum: `--repeats` samples are taken per leg and p50 and
p90 are printed. A minimum flatters whichever side has the fatter tail, and a
mean hides the tail entirely.

pyarrow's thread pool is global and set once per process, so the two legs
cannot both run at their own CPU count in one interpreter. The threaded leg
therefore re-execs this script with `--leg threaded`, and the single-threaded
leg runs in the parent. `--leg` is an internal flag; run the script with no
arguments.
"""

import os
import statistics
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

WIDE = "build/bench-wide.parquet"

CASES = [
    ("tests/fixtures/big.parquet", 25),
    ("tests/fixtures/prune.parquet", 50),
    ("tests/fixtures/encodings.parquet", 50),
    ("tests/fixtures/v2pages.parquet", 50),
    (WIDE, 25),
]

WRITE_CASES = [(WIDE, 10), ("tests/fixtures/big.parquet", 10)]


def make_wide():
    os.makedirs("build", exist_ok=True)
    n = 1_000_000
    t = pa.table(
        {
            "a": pa.array(range(n), pa.int64()),
            "b": pa.array([float(i) * 0.5 for i in range(n)], pa.float64()),
            "c": pa.array([i % 1000 for i in range(n)], pa.int64()),
            "d": pa.array([f"k{i % 997}" for i in range(n)], pa.string()),
        }
    )
    pq.write_table(t, WIDE, compression="none", row_group_size=250_000,
                   use_dictionary=["c", "d"])
    print(f"{WIDE}: {os.path.getsize(WIDE):,} bytes")


def percentiles(samples):
    """p50 and p90 of a sample list, in seconds."""
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    # Nearest-rank p90, which is what a small sample can honestly support.
    p90 = ordered[max(0, min(len(ordered) - 1, round(0.9 * len(ordered)) - 1))]
    return p50, p90


def time_read(path, repeats, use_threads):
    with open(path, "rb") as fh:
        raw = fh.read()
    samples = []
    rows = 0
    for _ in range(repeats):
        buf = pa.py_buffer(raw)
        t0 = time.perf_counter()
        table = pq.read_table(pa.BufferReader(buf), use_threads=use_threads)
        t1 = time.perf_counter()
        rows = table.num_rows
        samples.append(t1 - t0)
    return rows, len(raw), percentiles(samples)


def time_write(path, repeats):
    table = pq.read_table(path, use_threads=False)
    samples = []
    size = 0
    for _ in range(repeats):
        sink = pa.BufferOutputStream()
        t0 = time.perf_counter()
        pq.write_table(table, sink, compression="none")
        t1 = time.perf_counter()
        size = sink.getvalue().size
        samples.append(t1 - t0)
    return table.num_rows, size, percentiles(samples)


def report(leg):
    threaded = leg == "threaded"
    if threaded:
        label = f"threaded, cpu_count={pa.cpu_count()}"
    else:
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        label = "1 thread"

    print(f"pyarrow decode benchmark ({label})\n")
    for path, repeats in CASES:
        if not os.path.exists(path):
            print(f"{path}: missing")
            continue
        rows, nbytes, (p50, p90) = time_read(path, repeats, threaded)
        print(
            f"{os.path.basename(path)}: {rows:,} rows, {nbytes // 1024} KiB — "
            f"p50 {p50 * 1000:.2f} ms, p90 {p90 * 1000:.2f} ms "
            f"-> {int(rows / p50):,} rows/s, {nbytes / p50 / 1e6:.1f} MB/s"
        )

    # Encoding is single threaded on both sides — pyarrow's writer does not
    # take use_threads — so it is reported once, from the single-thread leg.
    if threaded:
        return
    print(f"\npyarrow encode benchmark ({label})\n")
    for path, repeats in WRITE_CASES:
        if not os.path.exists(path):
            print(f"{path}: missing")
            continue
        rows, size, (p50, p90) = time_write(path, repeats)
        print(
            f"{os.path.basename(path)}: {rows:,} rows -> {size // 1024} KiB — "
            f"p50 {p50 * 1000:.2f} ms, p90 {p90 * 1000:.2f} ms "
            f"-> {int(rows / p50):,} rows/s, {size / p50 / 1e6:.1f} MB/s"
        )


def main():
    if "--make" in sys.argv:
        make_wide()
        return
    if "--leg" in sys.argv:
        report(sys.argv[sys.argv.index("--leg") + 1])
        return
    report("single")
    print()
    # Flush before handing stdout to the child: when this script is piped its
    # own writes are block-buffered and the child's are not, so without this
    # the two legs come out in the wrong order.
    sys.stdout.flush()
    # A fresh interpreter: pa.set_cpu_count(1) above is process-wide and there
    # is no way back to "however many pyarrow would have picked".
    subprocess.run(
        [sys.executable, __file__, "--leg", "threaded"], check=True
    )


if __name__ == "__main__":
    main()
