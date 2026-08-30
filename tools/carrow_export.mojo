"""A tiny C shared library: read one column of a Parquet file and hand it to
the caller over the Arrow C Data Interface.

`tools/consume_c_data.py` dlopens this, passes two ctypes buffers, and feeds
the result straight to `pyarrow.Array._import_from_c` — which proves the
export is a real C Data Interface producer, release callback and all.

```console
mojo build --emit shared-lib tools/carrow_export.mojo -I src … -o build/libpqcarrow.so
```
"""

from parquet import ParquetReader, export_c


@export("pq_export_column")
def pq_export_column(
    path: UnsafePointer[UInt8, ImmUntrackedOrigin],
    col: Int32,
    arr_out: UnsafePointer[UInt8, MutUntrackedOrigin],
    sch_out: UnsafePointer[UInt8, MutUntrackedOrigin],
) abi("C") -> Int32:
    """Export column `col` of the NUL-terminated file `path`.

    The root `ArrowArray` (80 bytes) is copied into `arr_out` and the root
    `ArrowSchema` (72 bytes) into `sch_out`, exactly as a C producer would
    "move" them into caller-owned storage. Returns 0, or -1 on any error.
    """
    try:
        var name = String(unsafe_from_utf8_ptr=path)
        var r = ParquetReader.open(name)
        r.batch_size = 1 << 30
        var batch = r.read_batch()
        var ci = Int(col)
        if ci < 0 or ci >= batch.num_columns():
            return -1
        var e = export_c(batch.arena, batch.roots[ci])
        var raw = e.into_raw()
        var a = UnsafePointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=raw[0])
        var s = UnsafePointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=raw[1])
        for i in range(80):
            arr_out[unsafe_offset=i] = a[i]
        for i in range(72):
            sch_out[unsafe_offset=i] = s[i]
        return 0
    except:
        return -1
