"""A tiny C shared library: take Arrow data *in* over the C Data Interface.

The export side of this pair (`tools/carrow_export.mojo`) hands our arrays to
pyarrow. This one is the other direction, and it is the one that matters for
correctness: `tools/produce_c_data.py` has pyarrow build an array, calls
`_export_to_c` on it, and passes the two addresses in here — so the producer
under test is one we did not write, and a format string or buffer convention
we got subtly wrong shows up as a wrong value rather than as two of our own
modules agreeing with each other.

Three entry points, all of them *consumers*: each takes ownership of what it
is handed and releases it, which is what the C Data Interface says a consumer
does.

```console
mojo build --emit shared-lib tools/carrow_import.mojo -I src … -o build/libpqcimport.so
```
"""

from parquet import (
    ArrayArena,
    ImportedArray,
    ImportedStream,
    export_c,
    import_c,
)


def _write_out(
    text: StringSlice, buf: Pointer[UInt8, MutUntrackedOrigin], buflen: Int
):
    """Copy `text` into the caller's buffer, NUL-terminated and truncated."""
    var b = text.as_bytes()
    var n = min(len(b), buflen - 1)
    for i in range(n):
        buf[unsafe_offset=i] = b[i]
    buf[unsafe_offset=n] = 0


@export("pq_import_describe")
def pq_import_describe(
    array: Int,
    schema: Int,
    buf: Pointer[UInt8, MutUntrackedOrigin],
    buflen: Int,
) abi("C") -> Int32:
    """Import one array and report what we made of it.

    Writes `format|length|null_count|n_children|name` — the type as this
    library would spell it back out, so a format string that was misread turns
    into a mismatch on the Python side rather than into a plausible-looking
    number. On failure the error message goes into the same buffer and the
    result is -1, which is how the malformed-input cases assert *which* check
    fired.
    """
    var owned = ImportedArray(array, schema)
    try:
        var arena = ArrayArena()
        var root = import_c(arena, array, schema)
        ref a = arena.nodes[root]
        _write_out(
            String(
                a.type.format(),
                "|",
                a.length,
                "|",
                a.null_count,
                "|",
                len(a.children),
                "|",
                a.name,
            ),
            buf,
            buflen,
        )
        owned.release()
        return 0
    except e:
        owned.release()
        _write_out(String(e), buf, buflen)
        return -1


@export("pq_import_roundtrip")
def pq_import_roundtrip(
    array: Int,
    schema: Int,
    array_out: Pointer[UInt8, MutUntrackedOrigin],
    schema_out: Pointer[UInt8, MutUntrackedOrigin],
) abi("C") -> Int32:
    """Import one array and export it straight back.

    pyarrow then compares the array it gets back with the one it sent, which
    is the end-to-end check: every buffer had to survive being copied out of
    the producer's memory and written into ours in the layout the spec names.
    """
    var owned = ImportedArray(array, schema)
    try:
        var arena = ArrayArena()
        var root = import_c(arena, array, schema)
        owned.release()
        var e = export_c(arena, root)
        var raw = e.into_raw()
        var a = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=raw[0])
        var s = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=raw[1])
        for i in range(80):
            array_out[unsafe_offset=i] = a[unsafe_offset=i]
        for i in range(72):
            schema_out[unsafe_offset=i] = s[unsafe_offset=i]
        return 0
    except:
        owned.release()
        return -1


@export("pq_stream_describe")
def pq_stream_describe(
    stream: Int,
    buf: Pointer[UInt8, MutUntrackedOrigin],
    buflen: Int,
) abi("C") -> Int32:
    """Drain an `ArrowArrayStream` and report what came out of it.

    Writes `format|batches|rows|columns`. The stream is released here whatever
    happens, including on a raise, so the producer's memory does not outlive
    the call.
    """
    try:
        var s = ImportedStream(stream)
        var fmt = s.format()
        var batches = 0
        var rows = 0
        var columns = -1
        while True:
            var got = s.next()
            if not got:
                break
            ref batch = got.value()
            batches += 1
            rows += batch.num_rows
            if columns < 0:
                columns = batch.num_columns()
            elif columns != batch.num_columns():
                raise Error("the stream changed width mid-scan")
        _write_out(
            String(fmt, "|", batches, "|", rows, "|", max(columns, 0)),
            buf,
            buflen,
        )
        s.release()
        return 0
    except e:
        _write_out(String(e), buf, buflen)
        return -1
