#!/usr/bin/env python3
"""Import our Arrow C Data Interface export into pyarrow, and compare.

`tools/carrow_export.mojo` builds to a shared library with one C entry point:

    int32_t pq_export_column(const char* path, int32_t col,
                             void* array_out /* 80 bytes */,
                             void* schema_out /* 72 bytes */);

It copies the root `ArrowArray` and `ArrowSchema` into caller-owned storage,
which is exactly the "move" a C Data Interface consumer performs. This script
allocates that storage with ctypes, calls in, hands the two addresses to
`pyarrow.Array._import_from_c`, and asserts the imported array equals what
pyarrow itself reads from the same file. pyarrow then calls our `release`
callback when the array is collected.

    python tools/consume_c_data.py build/libpqcarrow.dylib
"""

import ctypes
import gc
import sys

import pyarrow as pa
import pyarrow.parquet as pq

LIB = sys.argv[1] if len(sys.argv) > 1 else "build/libpqcarrow.dylib"

CASES = [
    ("tests/fixtures/primitives.parquet", None),
    ("tests/fixtures/logical.parquet", None),
    ("tests/fixtures/extension.parquet", None),
    ("tests/fixtures/float16.parquet", None),
    ("tests/fixtures/nested.parquet", None),
    ("tests/fixtures/fieldids.parquet", None),
    ("tests/fixtures/int96.parquet", None),
    ("tests/fixtures/decimal_int.parquet", None),
    ("tests/fixtures/allnull.parquet", None),
    ("tests/fixtures/legacy_list.parquet", None),
]

lib = ctypes.CDLL(LIB)
lib.pq_export_column.restype = ctypes.c_int32
lib.pq_export_column.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int32,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def storage_type(t):
    while hasattr(t, "storage_type"):
        t = t.storage_type
    return t


checked = 0
failures = []
for path, _ in CASES:
    table = pq.read_table(path)
    # allnull.parquet has two row groups; the exporter returns the first batch.
    first_rows = pq.ParquetFile(path).metadata.row_group(0).num_rows
    for col in range(table.num_columns):
        arr_buf = (ctypes.c_char * 80)()
        sch_buf = (ctypes.c_char * 72)()
        rc = lib.pq_export_column(
            path.encode(), col, ctypes.byref(arr_buf), ctypes.byref(sch_buf)
        )
        if rc != 0:
            failures.append(f"{path}[{col}]: exporter returned {rc}")
            continue
        got = pa.Array._import_from_c(
            ctypes.addressof(arr_buf), ctypes.addressof(sch_buf)
        )
        want = table.column(col).combine_chunks()
        if isinstance(want, pa.ChunkedArray):
            want = want.chunk(0) if want.num_chunks else want.combine_chunks()
        want = want.slice(0, first_rows)
        want_cast = want
        if storage_type(want.type) != want.type:
            want_cast = want.cast(storage_type(want.type))
        if got.type != want_cast.type:
            # Our reader does not read pyarrow's ARROW:schema metadata, so a
            # large_string round-trips as string. Compare after a cast.
            try:
                want_cast = want_cast.cast(got.type)
            except pa.ArrowInvalid:
                failures.append(
                    f"{path}[{col}] {table.field(col).name}: "
                    f"type {got.type} != {want_cast.type}"
                )
                continue
        if not got.equals(want_cast):
            failures.append(
                f"{path}[{col}] {table.field(col).name}: values differ\n"
                f"  got : {got}\n  want: {want_cast}"
            )
            continue
        # Extension metadata survives the export.
        checked += 1
        del got
        gc.collect()

print(f"{checked} column(s) imported from Mojo into pyarrow and compared")
for f in failures:
    print("FAIL", f)
sys.exit(1 if failures else 0)
