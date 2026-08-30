#!/usr/bin/env python3
"""pyarrow must be able to read what parquet.mojo writes.

`tools/write_fixtures.mojo` writes every fixture back out through our writer
into `build/written/`. This reads both the original and our copy with pyarrow
and asserts that every value of every column is equal.

    python tools/verify_written.py
"""

import glob
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

WRITTEN = "build/written"
FIXTURES = "tests/fixtures"


def storage(t):
    while hasattr(t, "storage_type"):
        t = t.storage_type
    return t


def normalise(table):
    """Cast away the type differences Parquet cannot record."""
    fields = []
    arrays = []
    for i, field in enumerate(table.schema):
        col = table.column(i).combine_chunks()
        t = storage(field.type)
        if t != field.type:
            col = col.cast(t)
        if pa.types.is_large_string(t):
            t = pa.string()
            col = col.cast(t)
        elif pa.types.is_large_binary(t):
            t = pa.binary()
            col = col.cast(t)
        fields.append(pa.field(field.name, col.type))
        arrays.append(col)
    return pa.table(arrays, schema=pa.schema(fields))


failures = []
checked = 0
files = sorted(glob.glob(os.path.join(WRITTEN, "*.parquet")))
if not files:
    sys.exit(f"{WRITTEN} is empty — build and run tools/write_fixtures.mojo first")

for path in files:
    stem = os.path.basename(path)[: -len(".parquet")]
    base = stem.split("-")[0]
    original = os.path.join(FIXTURES, base + ".parquet")
    try:
        got = pq.read_table(path)
    except Exception as exc:  # pyarrow could not read our file at all
        failures.append(f"{stem}: pyarrow failed to read it — {exc}")
        continue
    want = pq.read_table(original)
    if got.num_rows != want.num_rows:
        failures.append(f"{stem}: {got.num_rows} rows, expected {want.num_rows}")
        continue
    if got.column_names != want.column_names:
        failures.append(
            f"{stem}: columns {got.column_names} != {want.column_names}"
        )
        continue
    g = normalise(got)
    w = normalise(want)
    for i, name in enumerate(g.column_names):
        gc = g.column(i)
        wc = w.column(i)
        if gc.type != wc.type:
            try:
                gc = gc.cast(wc.type)
            except pa.ArrowInvalid:
                failures.append(f"{stem}.{name}: type {gc.type} != {wc.type}")
                continue
        if not gc.equals(wc):
            failures.append(
                f"{stem}.{name}: values differ\n  got : {gc.slice(0, 8)}\n"
                f"  want: {wc.slice(0, 8)}"
            )
            continue
        checked += 1
    # Round-trip through our own metadata too.
    md = pq.ParquetFile(path).metadata
    if md.created_by is None or "parquet.mojo" not in md.created_by:
        failures.append(f"{stem}: created_by is {md.created_by!r}")

print(f"{len(files)} file(s), {checked} column(s) read by pyarrow and compared")
for f in failures:
    print("FAIL", f)
sys.exit(1 if failures else 0)
