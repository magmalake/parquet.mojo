#!/usr/bin/env python3
"""Write the Parquet test fixtures with pyarrow.

Every fixture is small on purpose — the whole directory including the oracle
JSON stays well under 5 MB. `tools/oracle_pyarrow.py` then dumps the values
pyarrow reads back out of each one; the Mojo test suite has to reproduce
those exactly.

Run through `pixi run fixtures`, which builds a throwaway uv venv.
"""

import datetime
import decimal
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

OUT = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"
os.makedirs(OUT, exist_ok=True)

WRITTEN = []


def write(name, table, **kw):
    path = os.path.join(OUT, name)
    kw.setdefault("version", "2.6")
    pq.write_table(table, path, **kw)
    WRITTEN.append((name, os.path.getsize(path)))
    print(f"  {name:26s} {os.path.getsize(path):>9,d} bytes")


def supports(fn, kwarg):
    import inspect

    try:
        return kwarg in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


print(f"pyarrow {pa.__version__} -> {OUT}")

# ── 1. every physical type, with nulls ─────────────────────────────────────
n = 12
primitives = pa.table(
    {
        "b": pa.array([True, False, None, True, False, True, None, False, True, True, False, None], pa.bool_()),
        "i8": pa.array([-128, -1, 0, 1, 127, None, 3, -3, 42, -42, 7, None], pa.int8()),
        "i16": pa.array([-32768, -1, 0, 1, 32767, None, 300, -300, 4242, -4242, 77, None], pa.int16()),
        "i32": pa.array([-2147483648, -1, 0, 1, 2147483647, None, 70000, -70000, 5, -5, 9, None], pa.int32()),
        "i64": pa.array([-9223372036854775808, -1, 0, 1, 9223372036854775807, None,
                         5000000000, -5000000000, 12, -12, 99, None], pa.int64()),
        "u8": pa.array([0, 1, 127, 128, 255, None, 3, 9, 42, 200, 7, None], pa.uint8()),
        "u16": pa.array([0, 1, 32767, 32768, 65535, None, 300, 900, 4242, 20000, 77, None], pa.uint16()),
        "u32": pa.array([0, 1, 2147483647, 2147483648, 4294967295, None, 70000, 9, 5, 123456, 9, None], pa.uint32()),
        "u64": pa.array([0, 1, 9223372036854775807, 9223372036854775808, 18446744073709551615, None,
                         5000000000, 9, 12, 123456789, 99, None], pa.uint64()),
        "f32": pa.array([0.0, -0.0, 1.5, -1.5, 3.4028234663852886e38, None,
                         1.1754943508222875e-38, float("inf"), float("-inf"), 0.1, -0.25, None], pa.float32()),
        "f64": pa.array([0.0, -0.0, 1.5, -1.5, 1.7976931348623157e308, None,
                         2.2250738585072014e-308, float("inf"), float("-inf"), 0.1, -0.25, None], pa.float64()),
        "s": pa.array(["", "a", "hello world", None, "ünïcødé ☃", "x" * 200, "a", "b", None, "z", "hello world", ""], pa.string()),
        "bin": pa.array([b"", b"\x00\x01\x02", b"\xff" * 5, None, b"abc", b"\x00", b"\x01", b"\x02",
                         None, b"\xde\xad\xbe\xef", b"abc", b""], pa.binary()),
        "flba": pa.array([b"0123", b"abcd", b"\x00\x00\x00\x00", None, b"\xff\xff\xff\xff",
                          b"wxyz", b"1234", b"5678", None, b"aaaa", b"abcd", b"zzzz"], pa.binary(4)),
    }
)
write("primitives.parquet", primitives)

# ── 2. logical types ───────────────────────────────────────────────────────
logical = pa.table(
    {
        "dec_flba": pa.array([decimal.Decimal("0.00"), decimal.Decimal("-1.23"), None,
                              decimal.Decimal("123456789012345678901234567.89"),
                              decimal.Decimal("-99999999999999999999999999.99")] * 2, pa.decimal128(29, 2)),
        "dec_small": pa.array([decimal.Decimal("0.0"), decimal.Decimal("-1.2"), None,
                               decimal.Decimal("999.9"), decimal.Decimal("-42.5")] * 2, pa.decimal128(4, 1)),
        "dec_mid": pa.array([decimal.Decimal("0.000"), decimal.Decimal("-1.234"), None,
                             decimal.Decimal("999999999999.999"), decimal.Decimal("-4.500")] * 2, pa.decimal128(15, 3)),
        "date": pa.array([datetime.date(1970, 1, 1), datetime.date(2026, 8, 29), None,
                          datetime.date(1900, 3, 4), datetime.date(2262, 4, 11)] * 2, pa.date32()),
        "time_ms": pa.array([0, 1000, None, 86399999, 43200000] * 2, pa.time32("ms")),
        "time_us": pa.array([0, 1000, None, 86399999999, 43200000000] * 2, pa.time64("us")),
        "time_ns": pa.array([0, 1000, None, 86399999999999, 43200000000000] * 2, pa.time64("ns")),
        "ts_ms": pa.array([0, 1, None, -1, 1756400000000] * 2, pa.timestamp("ms")),
        "ts_us": pa.array([0, 1, None, -1, 1756400000000000] * 2, pa.timestamp("us")),
        "ts_ns": pa.array([0, 1, None, -1, 1756400000000000000] * 2, pa.timestamp("ns")),
        "ts_ms_utc": pa.array([0, 1, None, -1, 1756400000000] * 2, pa.timestamp("ms", tz="UTC")),
        "ts_us_utc": pa.array([0, 1, None, -1, 1756400000000000] * 2, pa.timestamp("us", tz="UTC")),
        "ts_ns_utc": pa.array([0, 1, None, -1, 1756400000000000000] * 2, pa.timestamp("ns", tz="UTC")),
        "lstr": pa.array(["", "large", None, "string", "here"] * 2, pa.large_string()),
        "lbin": pa.array([b"", b"large", None, b"binary", b"here"] * 2, pa.large_binary()),
    }
)
write("logical.parquet", logical)

# ── 3. extension logical types ─────────────────────────────────────────────
import uuid as _uuid

uuids = [_uuid.UUID(int=i * 0x1111111111111111111111111111111 % (1 << 128)).bytes for i in range(5)]
extension = pa.table(
    {
        "uuid": pa.array(uuids + [None] * 2 + uuids[:3], pa.binary(16)),
        "json": pa.array(['{"a":1}', "[]", None, '"str"', "3.5"] * 2, pa.string()),
        "f16": pa.array([b"\x00\x00", b"\x00\x3c", None, b"\x00\xbc", b"\x00\x7c"] * 2, pa.binary(2)),
        "enum": pa.array(["RED", "GREEN", None, "BLUE", "RED"] * 2, pa.string()),
    },
    schema=pa.schema(
        [
            pa.field("uuid", pa.uuid() if hasattr(pa, "uuid") else pa.binary(16)),
            pa.field("json", pa.json_() if hasattr(pa, "json_") else pa.string()),
            pa.field("f16", pa.float16() if False else pa.binary(2)),
            pa.field("enum", pa.string()),
        ]
    )
    if hasattr(pa, "uuid")
    else None,
)
write("extension.parquet", extension)

# a real float16 column, written straight from a float16 array
f16 = pa.table({"h": pa.array([0.0, 1.0, None, -1.0, 65504.0, 0.5, None, -0.5, 2.0, 3.0], pa.float16())})
write("float16.parquet", f16)

# ── 4. nested ──────────────────────────────────────────────────────────────
nested = pa.table(
    {
        "li": pa.array([[1, 2, 3], [], None, [4], [None, 5], [6, None, 7], [], None, [8, 9], [10]],
                       pa.list_(pa.int32())),
        "lls": pa.array([[["a"], ["b", "c"]], [], None, [[]], [None], [["d"]], [[], ["e"]], None,
                         [["f", None]], [["g"]]], pa.list_(pa.list_(pa.string()))),
        "m": pa.array([[("k", 1)], [], None, [("a", 1), ("b", 2)], [("z", None)],
                       [("q", 9)], [], None, [("y", 3)], [("x", 4)]],
                      pa.map_(pa.string(), pa.int64())),
        "st": pa.array([{"a": 1, "b": "x"}, {"a": None, "b": "y"}, None, {"a": 3, "b": None},
                        {"a": 4, "b": "z"}, {"a": 5, "b": "w"}, None, {"a": None, "b": None},
                        {"a": 8, "b": "v"}, {"a": 9, "b": "u"}],
                       pa.struct([("a", pa.int32()), ("b", pa.string())])),
        "lst": pa.array([[{"n": 1}], [], None, [{"n": None}, {"n": 2}], [None],
                         [{"n": 3}], [], None, [{"n": 4}], [{"n": 5}]],
                        pa.list_(pa.struct([("n", pa.int64())]))),
        "lb": pa.array([[b"\x01"], [], None, [b"\x02\x03", None], [b""],
                        [b"\xff"], [], None, [b"ab"], [b"cd"]], pa.list_(pa.binary())),
    }
)
write("nested.parquet", nested)

# ── 5. encodings ───────────────────────────────────────────────────────────
N = 400
enc = pa.table(
    {
        "plain_i64": pa.array([i * 7 - 100 if i % 13 else None for i in range(N)], pa.int64()),
        "dict_str": pa.array([["alpha", "beta", "gamma", "delta"][i % 4] if i % 11 else None for i in range(N)],
                             pa.string()),
        "delta_i32": pa.array([i * 3 - 5 if i % 7 else None for i in range(N)], pa.int32()),
        "delta_i64": pa.array([i * 1_000_003 - 17 if i % 5 else None for i in range(N)], pa.int64()),
        "delta_str": pa.array([f"prefix-{i:06d}-suffix" if i % 9 else None for i in range(N)], pa.string()),
        "bss_f32": pa.array([float(i) * 1.25 if i % 6 else None for i in range(N)], pa.float32()),
        "bss_f64": pa.array([float(i) * 1.25 if i % 6 else None for i in range(N)], pa.float64()),
        "bss_i32": pa.array([i * 11 if i % 4 else None for i in range(N)], pa.int32()),
        "bss_i64": pa.array([i * 111111 if i % 4 else None for i in range(N)], pa.int64()),
        "bss_flba": pa.array([bytes([i % 256, (i >> 8) % 256, 7, 9]) if i % 4 else None for i in range(N)],
                             pa.binary(4)),
        "rle_bool": pa.array([(i // 20) % 2 == 0 if i % 3 else None for i in range(N)], pa.bool_()),
        "plain_bool": pa.array([i % 2 == 0 if i % 3 else None for i in range(N)], pa.bool_()),
    }
)
write(
    "encodings.parquet",
    enc,
    use_dictionary=["dict_str"],
    column_encoding={
        "plain_i64": "PLAIN",
        "delta_i32": "DELTA_BINARY_PACKED",
        "delta_i64": "DELTA_BINARY_PACKED",
        "delta_str": "DELTA_BYTE_ARRAY",
        "bss_f32": "BYTE_STREAM_SPLIT",
        "bss_f64": "BYTE_STREAM_SPLIT",
        "bss_i32": "BYTE_STREAM_SPLIT",
        "bss_i64": "BYTE_STREAM_SPLIT",
        "bss_flba": "BYTE_STREAM_SPLIT",
        "rle_bool": "RLE",
        "plain_bool": "PLAIN",
    },
    compression="none",
)

# DELTA_LENGTH_BYTE_ARRAY needs its own file (pyarrow allows it per column).
dlba = pa.table({"s": pa.array([f"row-{i}" if i % 8 else None for i in range(200)], pa.string())})
write("delta_length.parquet", dlba, column_encoding={"s": "DELTA_LENGTH_BYTE_ARRAY"},
      use_dictionary=False, compression="none")

# ── 5b. the Parquet 1.0 spellings: PLAIN_DICTIONARY and v1 data pages ──────
v1legacy = pa.table(
    {
        "d": pa.array([["alpha", "beta", "gamma"][i % 3] if i % 7 else None for i in range(120)], pa.string()),
        "i": pa.array([i % 5 for i in range(120)], pa.int32()),
        "l": pa.array([[i % 3] if i % 4 else ([] if i % 8 else None) for i in range(120)], pa.list_(pa.int64())),
    }
)
write("v1legacy.parquet", v1legacy, version="1.0", data_page_version="1.0", compression="none")

# ── 6. codecs ──────────────────────────────────────────────────────────────
M = 300
codec_cols = {}
for c in ["none", "snappy", "gzip", "zstd", "lz4", "brotli"]:
    codec_cols[c] = pa.array([f"{c}-value-{i % 37}" for i in range(M)], pa.string())
codec_cols["ints"] = pa.array(list(range(M)), pa.int64())
codecs = pa.table(codec_cols)
write(
    "codecs.parquet",
    codecs,
    compression={"none": "none", "snappy": "snappy", "gzip": "gzip", "zstd": "zstd",
                 "lz4": "lz4", "brotli": "brotli", "ints": "gzip"},
    use_dictionary=False,
)

# ── 7. v1 and v2 data pages, several pages per chunk, several row groups ───
K = 500
pages = pa.table(
    {
        "k": pa.array([i if i % 17 else None for i in range(K)], pa.int64()),
        "v": pa.array([f"value-{i}" if i % 19 else None for i in range(K)], pa.string()),
        "d": pa.array([float(i) / 3.0 if i % 23 else None for i in range(K)], pa.float64()),
        "l": pa.array([[i, i + 1] if i % 5 else ([] if i % 10 else None) for i in range(K)],
                      pa.list_(pa.int32())),
    }
)
write("v1pages.parquet", pages, data_page_version="1.0", data_page_size=1024, row_group_size=180,
      compression="snappy")
write("v2pages.parquet", pages, data_page_version="2.0", data_page_size=1024, row_group_size=180,
      compression="snappy")
write("v2pages_uncompressed.parquet", pages, data_page_version="2.0", data_page_size=1024,
      row_group_size=180, compression="none")

# ── 8. page index + checksums + bloom filters ──────────────────────────────
idx_kw = dict(data_page_size=512, row_group_size=150, write_statistics=True)
if supports(pq.write_table, "write_page_index"):
    idx_kw["write_page_index"] = True
if supports(pq.write_table, "write_page_checksum"):
    idx_kw["write_page_checksum"] = True
write("pageindex.parquet", pages, **idx_kw)

# `bloom_filter_options` is a {column: {ndv, fpp}} dict on recent pyarrow.
bloom_kw = {}
if supports(pq.write_table, "bloom_filter_options"):
    bloom_kw["bloom_filter_options"] = {
        "s": {"ndv": 200, "fpp": 0.01},
        "i": {"ndv": 200, "fpp": 0.01},
        "d": {"ndv": 200, "fpp": 0.01},
    }
elif supports(pq.write_table, "write_bloom_filter"):
    bloom_kw["write_bloom_filter"] = True
bloom = pa.table(
    {
        "s": pa.array([f"key-{i:04d}" for i in range(200)], pa.string()),
        "i": pa.array([i * 3 for i in range(200)], pa.int64()),
        "d": pa.array([float(i) for i in range(200)], pa.float64()),
    }
)
write("bloom.parquet", bloom, use_dictionary=False, **bloom_kw)

# ── 9. no statistics, no page index ────────────────────────────────────────
write("nostats.parquet", pages, write_statistics=False)

# ── 10. INT96 timestamps ───────────────────────────────────────────────────
i96 = pa.table(
    {
        "t": pa.array([0, 1, None, -1, 1756400000000000000, 1000000000, None, -1000000000,
                       86400000000000, 2] , pa.timestamp("ns")),
        "n": pa.array(list(range(10)), pa.int64()),
    }
)
write("int96.parquet", i96, use_deprecated_int96_timestamps=True)

# ── 11. decimals backed by INT32 / INT64 ───────────────────────────────────
dec_int = pa.table(
    {
        "d32": pa.array([decimal.Decimal("0.0"), decimal.Decimal("-1.2"), None,
                         decimal.Decimal("999.9"), decimal.Decimal("-42.5")] * 2, pa.decimal128(4, 1)),
        "d64": pa.array([decimal.Decimal("0.000"), decimal.Decimal("-1.234"), None,
                         decimal.Decimal("999999999999.999"), decimal.Decimal("-4.500")] * 2,
                        pa.decimal128(15, 3)),
    }
)
if supports(pq.write_table, "store_decimal_as_integer"):
    write("decimal_int.parquet", dec_int, store_decimal_as_integer=True)
else:
    print("  (skipped decimal_int.parquet — pyarrow has no store_decimal_as_integer)")

# ── 12. legacy 2-level lists ───────────────────────────────────────────────
legacy = pa.table(
    {
        "li": pa.array([[1, 2], [], None, [3], [4, 5, 6], [], [7], None, [8], [9]], pa.list_(pa.int32())),
        "lm": pa.array([[("a", 1)], [], None, [("b", 2)], [("c", 3)], [], [("d", 4)], None,
                        [("e", 5)], [("f", 6)]], pa.map_(pa.string(), pa.int64())),
    }
)
write("legacy_list.parquet", legacy, use_compliant_nested_type=False)

# ── 13. degenerate files ───────────────────────────────────────────────────
empty = pa.table({"a": pa.array([], pa.int64()), "b": pa.array([], pa.string())})
write("empty.parquet", empty)

allnull = pa.table(
    {
        "a": pa.array([None] * 20, pa.int64()),
        "b": pa.array([None] * 20, pa.string()),
        "c": pa.array([None] * 20, pa.list_(pa.int32())),
    }
)
write("allnull.parquet", allnull, row_group_size=10)

# ── 14. field ids, for Iceberg-style projection ────────────────────────────
fid_schema = pa.schema(
    [
        pa.field("id", pa.int64(), metadata={b"PARQUET:field_id": b"1"}),
        pa.field("name", pa.string(), metadata={b"PARQUET:field_id": b"2"}),
        pa.field("score", pa.float64(), metadata={b"PARQUET:field_id": b"3"}),
        pa.field(
            "tags",
            pa.list_(pa.field("element", pa.string(), metadata={b"PARQUET:field_id": b"5"})),
            metadata={b"PARQUET:field_id": b"4"},
        ),
    ]
)
fids = pa.table(
    {
        "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
        "name": pa.array(["a", "b", None, "d", "e"], pa.string()),
        "score": pa.array([1.5, None, 3.5, 4.5, 5.5], pa.float64()),
        "tags": pa.array([["x"], [], None, ["y", "z"], ["w"]], pa.list_(pa.string())),
    },
    schema=fid_schema,
)
write("fieldids.parquet", fids)

# ── 15. many row groups with disjoint ranges, for statistics pruning ───────
P = 1000
prune = pa.table(
    {
        "k": pa.array(list(range(P)), pa.int64()),
        "s": pa.array([f"s{i:05d}" for i in range(P)], pa.string()),
        "f": pa.array([float(i) for i in range(P)], pa.float64()),
    }
)
write("prune.parquet", prune, row_group_size=100, write_statistics=True)

# ── 16. 100k rows of mixed types, for batching and the bench ───────────────
B = 100_000
big = pa.table(
    {
        "i": pa.array([i if i % 101 else None for i in range(B)], pa.int64()),
        "f": pa.array([float(i) * 0.5 if i % 97 else None for i in range(B)], pa.float64()),
        "s": pa.array([f"str-{i % 5000}" if i % 89 else None for i in range(B)], pa.string()),
        "b": pa.array([i % 3 == 0 if i % 79 else None for i in range(B)], pa.bool_()),
        "l": pa.array([[i % 7, i % 11] if i % 5 else None for i in range(B)], pa.list_(pa.int32())),
    }
)
write("big.parquet", big, row_group_size=25_000, data_page_size=64 * 1024, compression="snappy")

total = sum(s for _, s in WRITTEN)
print(f"\n{len(WRITTEN)} fixtures, {total:,d} bytes total")
