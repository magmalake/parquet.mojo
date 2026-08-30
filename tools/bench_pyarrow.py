#!/usr/bin/env python3
"""The same decode benchmark, in pyarrow with one thread, for the README.

    python tools/bench_pyarrow.py --make    # write build/bench-wide.parquet
    python tools/bench_pyarrow.py           # time pyarrow on each fixture
"""

import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(1)
pa.set_io_thread_count(1)

WIDE = "build/bench-wide.parquet"

if "--make" in sys.argv:
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
    sys.exit(0)

CASES = [
    ("tests/fixtures/big.parquet", 5),
    ("tests/fixtures/prune.parquet", 20),
    ("tests/fixtures/encodings.parquet", 50),
    ("tests/fixtures/v2pages.parquet", 50),
    (WIDE, 3),
]

print("pyarrow decode benchmark (single threaded)\n")
for path, repeats in CASES:
    if not os.path.exists(path):
        print(f"{path}: missing")
        continue
    with open(path, "rb") as fh:
        raw = fh.read()
    best = 1e30
    rows = 0
    for _ in range(repeats):
        buf = pa.py_buffer(raw)
        t0 = time.perf_counter()
        table = pq.read_table(pa.BufferReader(buf), use_threads=False)
        t1 = time.perf_counter()
        rows = table.num_rows
        best = min(best, t1 - t0)
    print(
        f"{os.path.basename(path)}: {rows:,} rows, {len(raw)//1024} KiB in "
        f"{best*1000:.2f} ms -> {int(rows/best):,} rows/s, "
        f"{len(raw)/best/1e6:.1f} MB/s"
    )
