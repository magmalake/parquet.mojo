# Fixture provenance

Every `.parquet` file here is written by `tools/gen_fixtures.py` with
pyarrow 25.0.1; every `.oracle.json` beside it is written by
`tools/oracle_pyarrow.py`, which reads the file back with pyarrow and
dumps each column's values. Regenerate both with `pixi run fixtures`
(needs `uv` on PATH).

`iceberg/` is different — see `iceberg/PROVENANCE.md`. Those are real
Apache Iceberg data files taken from the sibling tins' test warehouses,
most of them written by `parquet-rs` rather than pyarrow.

| file | bytes | rows | row groups | covers |
|---|---:|---:|---:|---|
| `allnull.parquet` | 1,204 | 20 | 2 | two row groups in which every value of every column is null |
| `big.parquet` | 1,407,737 | 100,000 | 4 | 100,000 rows of mixed types over 4 row groups — batching and the benchmark |
| `bloom.parquet` | 4,188 | 200 | 1 | split-block bloom filters on three columns |
| `codecs.parquet` | 8,404 | 300 | 1 | one column each of uncompressed, snappy, gzip, zstd, lz4 (`LZ4_RAW` on disk) and brotli |
| `decimal_int.parquet` | 792 | 10 | 1 | decimal backed by `INT32` and by `INT64` |
| `delta_length.parquet` | 1,577 | 200 | 1 | `DELTA_LENGTH_BYTE_ARRAY` |
| `empty.parquet` | 467 | 0 | 1 | zero rows |
| `encodings.parquet` | 19,830 | 400 | 1 | `PLAIN`, `RLE_DICTIONARY`, `DELTA_BINARY_PACKED`, `DELTA_BYTE_ARRAY`, `BYTE_STREAM_SPLIT` (f32/f64/i32/i64/FLBA), `RLE` booleans, plain booleans |
| `extension.parquet` | 1,623 | 10 | 1 | the `UUID` and `Json` logical types and a bare fixed-len 2-byte column |
| `fieldids.parquet` | 1,804 | 5 | 1 | Parquet field ids on flat and nested fields, for Iceberg-style projection |
| `float16.parquet` | 441 | 10 | 1 | the `Float16` logical type over `FIXED_LEN_BYTE_ARRAY(2)` |
| `int96.parquet` | 785 | 10 | 1 | deprecated `INT96` timestamps |
| `legacy_list.parquet` | 1,366 | 10 | 1 | `use_compliant_nested_type=False` — the inner list field is named `item`, not `element` |
| `logical.parquet` | 4,830 | 10 | 1 | decimal over FLBA at three widths, date32, time ms/µs/ns, timestamp ms/µs/ns with and without UTC, large string, large binary |
| `manypages.parquet` | 60,778 | 2,000 | 2 | 20 pages per row group with disjoint value ranges and no dictionary — page-level pruning |
| `nested.parquet` | 3,225 | 10 | 1 | list, list-of-list, map, struct, list-of-struct, list-of-binary — with empty and null containers and null elements |
| `nostats.parquet` | 11,784 | 500 | 1 | statistics and page index disabled — the absent-optional path |
| `pageindex.parquet` | 16,566 | 500 | 4 | `OffsetIndex` + `ColumnIndex`, page CRC32 checksums, 4 row groups, many pages per chunk |
| `primitives.parquet` | 4,130 | 12 | 1 | every physical type — boolean, int32/64, float, double, byte array, fixed-len byte array — plus int8…uint64 logical widths, nulls in every column |
| `prune.parquet` | 21,233 | 1,000 | 10 | 10 row groups with disjoint key ranges and statistics, for row-group pruning |
| `v1legacy.parquet` | 1,247 | 120 | 1 | Parquet 1.0 spellings — `PLAIN_DICTIONARY` and v1 data pages |
| `v1pages.parquet` | 13,585 | 500 | 3 | v1 data pages, 3 row groups, several pages per chunk, snappy |
| `v2pages.parquet` | 13,997 | 500 | 3 | v2 data pages (`DataPageHeaderV2`), 3 row groups, several pages per chunk, snappy |
| `v2pages_uncompressed.parquet` | 21,412 | 500 | 3 | the same, uncompressed — v2 pages whose levels are outside the compressed region |

Whole directory including the oracles and `iceberg/`: 1,983 KiB.

One gap worth naming: pyarrow 25 no longer writes **2-level** legacy
lists even with `use_compliant_nested_type=False` — it writes a 3-level
list whose inner field is named `item` (which `legacy_list.parquet`
does cover). The 2-level backward-compatibility rules are therefore
covered by a synthetic schema test instead.
