# Iceberg data files — provenance

These are **not** written by this repository's fixture generator. They are real
Apache Iceberg data files lifted out of the test warehouses of the two sibling
tins, so that parquet.mojo is checked against writers other than pyarrow:

| file | rows | writer | codec | table |
|---|---:|---|---|---|
| `unpartitioned.parquet` | 3 | parquet-rs 58.4.0 | uncompressed | `iceberg.mojo` `db.unpartitioned` |
| `ident_part_eu.parquet` | 2 | parquet-rs 58.4.0 | uncompressed | `iceberg.mojo` `db.ident_part`, `region=eu` |
| `day_part.parquet` | 1 | parquet-rs 58.4.0 | uncompressed | `iceberg.mojo` `db.day_part`, `ts_day=2023-11-15` |
| `bucket_part.parquet` | 2 | parquet-rs 58.4.0 | uncompressed | `iceberg.mojo` `db.bucket_part`, `id_bucket=0` |
| `nullable.parquet` | 3 | parquet-rs 58.4.0 | uncompressed | `iceberg-rs.mojo` `sales.nullable` |
| `orders_apac.parquet` | 1 | parquet-rs 58.4.0 | uncompressed | `iceberg-rs.mojo` `sales.orders`, `region=apac` |
| `deletes_data.parquet` | 3 | parquet-cpp-arrow 25.0.1 (PyIceberg) | **zstd** | `iceberg.mojo` `db.deletes_v2` |
| `evolved.parquet` | 2 | parquet-cpp-arrow 25.0.1 (PyIceberg) | **zstd** | `iceberg.mojo` `db.evolved` |
| `position_deletes.parquet` | 2 | parquet-cpp-arrow 25.0.1 (PyIceberg) | snappy | `iceberg.mojo` `db.deletes_v2` position deletes |

What they are worth testing against:

* **a second writer.** `parquet-rs` names the root schema element
  `arrow_schema` rather than `schema`, and makes its own dictionary and
  page-layout decisions — none of which pyarrow's fixtures exercise.
* **Iceberg field ids on every column**, which is how a reader is supposed to
  project. The position-delete file uses the reserved ids 2147483546
  (`file_path`) and 2147483545 (`pos`).
* **the `required` repetition type at the top level**, which pyarrow never
  writes — every pyarrow column is optional.
* **timestamps without a time zone at microsecond precision**, Iceberg's
  `timestamp` type.

The oracles beside them are produced the same way as every other fixture's, by
`tools/oracle_pyarrow.py`.

The comparison this repository performs is against **pyarrow reading the same
files**, not against `iceberg-rs.mojo`'s `ib_scan_next`: pyarrow is the oracle
everywhere else in this suite, and the point of these fixtures is the *writer*
and the *schema shape*, which pyarrow reports faithfully.
