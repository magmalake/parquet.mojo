#!/usr/bin/env python3
"""Have pyarrow produce Arrow data and import it in Mojo — the direction that
matters.

`tools/consume_c_data.py` checks that pyarrow can read what we export. That
proves the export is well formed but says nothing about the import, and an
import checked only against our own export would be two of our modules
agreeing with each other. So here pyarrow is the *producer*: it builds an
array, `_export_to_c` moves it into ctypes storage, and
`tools/carrow_import.mojo` takes ownership of it exactly as a consumer must.

Each case is checked twice, because the two checks fail differently.

* `pq_import_describe` reports the type as *we* would spell it back out, plus
  the length, the null count and the child count. A format string we misread
  fails here, loudly, with both spellings visible.
* `pq_import_roundtrip` imports and re-exports, and pyarrow compares what
  comes back with what it sent. That is the end-to-end check on the buffers:
  every byte had to survive being copied out of pyarrow's memory into ours in
  the layout the specification names.

`_export_to_c` on a `RecordBatchReader` gives the third struct,
`ArrowArrayStream`, which is what a scan actually returns — that is drained by
`pq_stream_describe`.

    python tools/produce_c_data.py build/libpqcimport.dylib
"""

import ctypes
import gc
import sys
from decimal import Decimal

import pyarrow as pa

LIB = sys.argv[1] if len(sys.argv) > 1 else "build/libpqcimport.dylib"

lib = ctypes.CDLL(LIB)
lib.pq_import_describe.restype = ctypes.c_int32
lib.pq_import_describe.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
]
lib.pq_import_roundtrip.restype = ctypes.c_int32
lib.pq_import_roundtrip.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
lib.pq_stream_describe.restype = ctypes.c_int32
lib.pq_stream_describe.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
]


def exported(arr):
    """`arr` moved into caller-owned C storage, as a producer would move it."""
    a = (ctypes.c_char * 80)()
    s = (ctypes.c_char * 72)()
    arr._export_to_c(ctypes.addressof(a), ctypes.addressof(s))
    return a, s


def describe(arr):
    a, s = exported(arr)
    buf = ctypes.create_string_buffer(4096)
    rc = lib.pq_import_describe(
        ctypes.addressof(a), ctypes.addressof(s), buf, len(buf)
    )
    return rc, buf.value.decode("utf-8", "replace")


def roundtrip(arr):
    a, s = exported(arr)
    out_a = (ctypes.c_char * 80)()
    out_s = (ctypes.c_char * 72)()
    rc = lib.pq_import_roundtrip(
        ctypes.addressof(a),
        ctypes.addressof(s),
        ctypes.addressof(out_a),
        ctypes.addressof(out_s),
    )
    if rc != 0:
        return None
    return pa.Array._import_from_c(
        ctypes.addressof(out_a), ctypes.addressof(out_s)
    )


CASES = [
    ("bool", pa.array([True, False, None, True]), "b"),
    ("int8", pa.array([-128, 0, 127, None], pa.int8()), "c"),
    ("uint8", pa.array([0, 255, None], pa.uint8()), "C"),
    ("int16", pa.array([-32768, 32767, None], pa.int16()), "s"),
    ("uint16", pa.array([0, 65535], pa.uint16()), "S"),
    ("int32", pa.array([-1, 0, 1, None], pa.int32()), "i"),
    ("uint32", pa.array([0, 4294967295], pa.uint32()), "I"),
    ("int64", pa.array([-(2**63), 2**63 - 1, None], pa.int64()), "l"),
    ("uint64", pa.array([0, 2**64 - 1], pa.uint64()), "L"),
    ("float16", pa.array([1.5, -2.25, None], pa.float16()), "e"),
    ("float32", pa.array([1.5, float("inf"), None], pa.float32()), "f"),
    ("float64", pa.array([1.5, float("-inf"), None], pa.float64()), "g"),
    ("string", pa.array(["", "héllo", None, "x" * 300]), "u"),
    ("large_string", pa.array(["a", None, "bb"], pa.large_string()), "U"),
    ("binary", pa.array([b"", b"\x00\xff", None], pa.binary()), "z"),
    ("large_binary", pa.array([b"z", None], pa.large_binary()), "Z"),
    (
        "fixed_size_binary",
        pa.array([b"0123456789abcdef", None], pa.binary(16)),
        "w:16",
    ),
    (
        "decimal128",
        pa.array(
            [Decimal("1.234567890"), None, Decimal("-9.999999999")],
            pa.decimal128(38, 9),
        ),
        "d:38,9",
    ),
    ("date32", pa.array([0, 19000, None], pa.date32()), "tdD"),
    ("time32_s", pa.array([0, 86399, None], pa.time32("s")), "tts"),
    ("time32_ms", pa.array([0, 1000], pa.time32("ms")), "ttm"),
    ("time64_us", pa.array([0, 123456789], pa.time64("us")), "ttu"),
    ("time64_ns", pa.array([0, None], pa.time64("ns")), "ttn"),
    ("timestamp_us", pa.array([0, None], pa.timestamp("us")), "tsu:"),
    (
        "timestamp_ns_utc",
        pa.array([0, 1], pa.timestamp("ns", tz="UTC")),
        "tsn:UTC",
    ),
    ("timestamp_s", pa.array([1, 2, 3], pa.timestamp("s")), "tss:"),
    ("null", pa.nulls(5), "n"),
    ("empty_int64", pa.array([], pa.int64()), "l"),
    ("empty_string", pa.array([], pa.string()), "u"),
    ("all_null_int64", pa.array([None, None, None], pa.int64()), "l"),
    ("list", pa.array([[1, 2], None, [], [3]], pa.list_(pa.int32())), "+l"),
    (
        "large_list",
        pa.array([["a"], None, []], pa.large_list(pa.string())),
        "+L",
    ),
    (
        "struct",
        pa.array(
            [{"a": 1, "b": "x"}, None, {"a": None, "b": "yy"}],
            pa.struct([("a", pa.int64()), ("b", pa.string())]),
        ),
        "+s",
    ),
    (
        "map",
        pa.array(
            [[("k", 1), ("j", 2)], None, []],
            pa.map_(pa.string(), pa.int32()),
        ),
        "+m",
    ),
    (
        "list_of_struct",
        pa.array(
            [[{"a": 1}], None, [{"a": None}, {"a": 3}]],
            pa.list_(pa.struct([("a", pa.int64())])),
        ),
        "+l",
    ),
    (
        "struct_of_list",
        pa.array(
            [{"xs": [1, 2]}, {"xs": None}, None],
            pa.struct([("xs", pa.list_(pa.int32()))]),
        ),
        "+s",
    ),
]

# A producer's `offset` is a slice, and `ArrayData` has no field for one, so
# every sliced case is a check that the importer materialised the window
# instead of reading from the front of the buffer. The unaligned starts are
# deliberate: a validity bitmap sliced at 3 or 5 has to be re-packed bit by
# bit, not memcpy'd.
SLICES = [
    ("int64", pa.array(list(range(20)) + [None], pa.int64()), "l"),
    ("string", pa.array([f"s{i}" * i for i in range(20)] + [None]), "u"),
    ("bool", pa.array([i % 3 == 0 for i in range(20)] + [None]), "b"),
    ("list", pa.array([[i, i + 1] for i in range(20)] + [None]), "+l"),
    (
        "struct",
        pa.array(
            [{"a": i, "b": f"v{i}"} for i in range(20)] + [None],
            pa.struct([("a", pa.int64()), ("b", pa.string())]),
        ),
        "+s",
    ),
    (
        "map",
        pa.array(
            [[("k", i)] for i in range(20)], pa.map_(pa.string(), pa.int32())
        ),
        "+m",
    ),
]

failures = []
checked = 0


def check_case(name, arr, want_format):
    global checked
    rc, text = describe(arr)
    if rc != 0:
        failures.append(f"{name}: describe failed: {text}")
        return
    parts = text.split("|")
    got_format = parts[0]
    length = int(parts[1])
    nulls = int(parts[2])
    children = int(parts[3])
    if got_format != want_format:
        failures.append(
            f"{name}: format {got_format!r} != {want_format!r}"
        )
    if length != len(arr):
        failures.append(f"{name}: length {length} != {len(arr)}")
    if nulls != arr.null_count:
        failures.append(f"{name}: null_count {nulls} != {arr.null_count}")
    want_children = arr.type.num_fields
    if children != want_children:
        failures.append(f"{name}: n_children {children} != {want_children}")
    got = roundtrip(arr)
    if got is None:
        failures.append(f"{name}: roundtrip returned an error")
        return
    if not got.equals(arr):
        failures.append(
            f"{name}: round trip differs\n  got : {got}\n  want: {arr}"
        )
    del got
    gc.collect()
    checked += 1


for name, arr, want in CASES:
    check_case(name, arr, want)

for name, arr, want in SLICES:
    for start, length in ((1, 5), (3, 7), (5, len(arr) - 5), (8, 0)):
        check_case(
            f"{name}[{start}:{start + length}]",
            arr.slice(start, length),
            want,
        )

# ── the stream ─────────────────────────────────────────────────────────────

schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
batches = [
    pa.record_batch(
        [pa.array([i * 10 + j for j in range(i + 1)], pa.int64()),
         pa.array([f"r{j}" for j in range(i + 1)])],
        schema=schema,
    )
    for i in range(4)
]
reader = pa.RecordBatchReader.from_batches(schema, batches)
stream = (ctypes.c_char * 40)()
reader._export_to_c(ctypes.addressof(stream))
buf = ctypes.create_string_buffer(4096)
rc = lib.pq_stream_describe(ctypes.addressof(stream), buf, len(buf))
text = buf.value.decode("utf-8", "replace")
want_rows = sum(b.num_rows for b in batches)
if rc != 0:
    failures.append(f"stream: {text}")
elif text != f"+s|{len(batches)}|{want_rows}|2":
    failures.append(f"stream: {text!r} != '+s|{len(batches)}|{want_rows}|2'")
else:
    checked += 1

# ── malformed input from a producer we do not control ──────────────────────
#
# The interface carries no buffer sizes, so a truncated `utf8` data buffer is
# undetectable by construction. What *is* checkable is checked: a buffer count
# that contradicts the format string, and a child shorter than the parent that
# indexes into it. Both must come back as an error naming what disagreed, not
# as a read into somebody else's memory.

arr = pa.array(["a", "bb", "ccc"])
a, s = exported(arr)
ctypes.c_int64.from_address(ctypes.addressof(a) + 24).value = 2  # n_buffers
buf = ctypes.create_string_buffer(4096)
rc = lib.pq_import_describe(ctypes.addressof(a), ctypes.addressof(s), buf, len(buf))
text = buf.value.decode("utf-8", "replace")
if rc == 0 or "needs 3 buffers" not in text:
    failures.append(f"malformed n_buffers: rc={rc} {text!r}")
else:
    checked += 1

arr = pa.array([[1, 2], [3], [4, 5, 6]], pa.list_(pa.int32()))
a, s = exported(arr)
children = ctypes.c_int64.from_address(ctypes.addressof(a) + 48).value
child_ptr = ctypes.c_int64.from_address(children).value
ctypes.c_int64.from_address(child_ptr).value = 1  # the child claims one row
buf = ctypes.create_string_buffer(4096)
rc = lib.pq_import_describe(ctypes.addressof(a), ctypes.addressof(s), buf, len(buf))
text = buf.value.decode("utf-8", "replace")
if rc == 0 or "run past the end of its child" not in text:
    failures.append(f"malformed child length: rc={rc} {text!r}")
else:
    checked += 1

print(f"{checked} case(s) produced by pyarrow and imported into Mojo")
for f in failures:
    print("FAIL", f)
sys.exit(1 if failures else 0)
