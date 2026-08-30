# parquet.mojo

[![mojoshelf](https://mojoshelf.org/badge/parquet-mojo.svg)](https://mojoshelf.org/tins/parquet-mojo) [![mojo nightly](https://mojoshelf.org/badge/parquet-mojo/nightly.svg)](https://mojoshelf.org/tins/parquet-mojo)

[![CI](https://github.com/magmalake/parquet.mojo/actions/workflows/ci.yml/badge.svg)](https://github.com/magmalake/parquet.mojo/actions/workflows/ci.yml) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> Part of **magmalake** — data lake building blocks in Mojo.

**The first native Apache Parquet decoder in [Mojo](https://www.modular.com/mojo).**
Every other Mojo dataframe today shells out to PyArrow through the Python
interpreter to read a Parquet file. This one does not: it reads the footer, the
page headers, the levels and the values itself, and hands the result back as
Arrow arrays over the **Arrow C Data Interface**.

```mojo
from parquet import ParquetReader

var r = ParquetReader.open("part-0.parquet")
r.select_columns(["id", "name"])
var t = r.read_table()
print(t.num_rows, "rows,", t.num_columns(), "columns")

var ids = t.column_i64(0)        # (values, validity)
for i in range(len(ids[0])):
    if ids[1][i]:
        print(ids[0][i])
```

## What it does

* **Reads any Parquet file** pyarrow 25 can write: every physical and logical
  type, every encoding, v1 and v2 data pages, dictionary pages, page CRC32s,
  multiple row groups and multiple pages per chunk.
* **Rebuilds nested data** — lists, lists of lists, maps and structs — from
  definition and repetition levels, including the two-level backward-compatible
  list layouts old writers produced.
* **Projects** by column name, by dotted leaf path, or by **Parquet field id**,
  which is what Apache Iceberg needs for schema evolution.
* **Skips row groups and pages** whose statistics or `ColumnIndex` bounds
  prove no row can match a `column op literal` predicate, and **probes bloom
  filters** for equality.
* **Exports Arrow** over the C Data Interface, with a real `release` callback —
  `pyarrow.Array._import_from_c` takes the result directly.
* **Writes Parquet** back out — `PLAIN` and `RLE_DICTIONARY`, all the codecs,
  statistics, field ids and a page index — and pyarrow reads what it writes.

## Install

```toml
[dependencies]
parquet-mojo = { git = "https://github.com/magmalake/parquet.mojo" }
```

The package's `.mojopkg` is built with stable Mojo 1.0.0 and the nightly
compiler will not load it, so a nightly consumer should put the source on the
include path instead and check the sibling tins out next to it:

```console
mojo build app.mojo \
    -I ../parquet.mojo/src -I ../thrift.mojo/src -I ../hashes.mojo/src \
    -I ../snappy.mojo/src -I ../avro.mojo/src
```

The four core dependencies are all pure Mojo:
[thrift.mojo](https://github.com/magmalake/thrift.mojo) for the metadata,
[hashes.mojo](https://github.com/magmalake/hashes.mojo) for page CRC32 and the
bloom filters' XXH64, [snappy.mojo](https://github.com/magmalake/snappy.mojo)
for the SNAPPY codec, and
[avro.mojo](https://github.com/magmalake/avro.mojo)'s `deflate.inflate` for the
DEFLATE half of GZIP. `ZSTD` and `LZ4` are optional and pull in the two FFI
tins — see [Codecs](#codecs).

## API

### `ParquetReader[Codecs: CodecSet = DefaultCodecs]`

| | |
|---|---|
| `ParquetReader.open(path)` | read a file |
| `ParquetReader(bytes)` | read an owned `List[UInt8]` |
| `.num_rows()`, `.num_row_groups()`, `.created_by()` | file metadata |
| `.key_value_metadata()` | `List[Tuple[String, String]]` |
| `.split_offsets()` | Iceberg's `data_file.split_offsets` |
| `.statistics(rg, leaf)` | `TypedStats` — decoded min/max/null count |
| `.offset_index(rg, leaf)`, `.column_index(rg, leaf)` | the page index |
| `.schema` | the `ParquetSchema` (see below) |
| `.select_columns([...])` | project by top-level name |
| `.select_field_ids([...])` | project by Parquet field id |
| `.select_fields([...])` | project by Arrow field index, **sub-fields included** |
| `.select_field_ids_deep([...])` | the same, addressed by Parquet field id |
| `.select_row_groups([...])` | read only these row groups |
| `.prune_row_groups(predicates)` | drop row groups by statistics |
| `.prune_pages(predicates)` | drop *pages* by the `ColumnIndex`; returns the rows left |
| `.page_row_ranges(rg, predicates)` | the row ranges those pages cover |
| `.batch_size` | rows per batch (default 65536) |
| `.verify_crc` | check page CRC32s when present (default true) |
| `.read_batch()` / `.has_next()` / `.rewind()` | iterate batches |
| `.read_table()` | every selected row group, as a `Table` |

A batch never spans two row groups, and never crosses a gap left by page
pruning, so the last batch of each range can be shorter than `batch_size`.

### `RecordBatch` and `Table`

`RecordBatch` has `num_rows`, `num_columns()`, `name(i)`, `type(i)`,
`column(i) -> ArrayData`, `export_c(i)`, the four typed accessors below, and
`arena` / `roots` for the nested case. `Table` is the list of batches a read
produced, with the same accessors running across all of them:

```mojo
t.column_i64(i)   -> (List[Int64],   List[Bool])   # any integer/date/time/timestamp
t.column_f64(i)   -> (List[Float64], List[Bool])   # float16/32/64
t.column_bool(i)  -> (List[Bool],    List[Bool])
t.column_str(i)   -> (List[String],  List[Bool])   # utf8 or binary
```

Nested values are reached through the arena: a list's `children[0]` is its
element array and its `offsets` say which elements belong to which row; a map's
`children[0]` is the `key_value` struct, whose two children are the keys and
the values.

### Arrow C Data Interface — the output contract

```mojo
var e = batch.export_c(0)            # or export_c(batch.arena, batch.roots[0])
# e.array and e.schema are the addresses of a C ArrowArray and ArrowSchema
var raw = e.into_raw()          # hand ownership to the consumer
```

`export_c` makes two allocations — one for the whole schema tree, one for the
whole array tree — laid out as the C structs, then the pointer arrays, then the
format and name strings and the metadata blocks, then **copies** of every
buffer. The root's `release` callback frees its block and is `abi("C")`, so it
can be called from C, Python or Rust; children carry a release that only marks
themselves released, because a consumer releases the root and the root owns
everything. Because buffers are copied, an export outlives the `RecordBatch` it
came from.

Extension types travel in the schema's `metadata` block as
`ARROW:extension:name`, so a `UUID` column arrives in pyarrow as
`extension<arrow.uuid>` and a `Json` column as `extension<arrow.json>`.

`ExportedArray` releases itself when it goes out of scope unless you call
`into_raw()` first, and `release()` is idempotent.

### `ParquetSchema`

`build_schema(file_metadata.schema)` produces three views:

* `nodes` — the Parquet tree, one `SchemaNode` per `SchemaElement`, each with
  its `max_def`, `max_rep`, `path` and `field_id`;
* `leaves` — the columns that hold data, in column-chunk order, each with its
  physical type, its Arrow type and its levels;
* `fields` / `roots` — the Arrow field tree, where a `LIST` collapses the
  annotated group and its repeated child into one node and a `MAP` keeps the
  `key_value` group as its single `STRUCT` child.

`field_by_name`, `field_by_id` and `leaf_by_path` find things.

### CLI

```console
$ parquet-mojo schema tests/fixtures/nested.parquet
message schema {
  li: list
    element: int32 <- INT32 def=3 rep=1
  lls: list
    element: list
      element: string <- BYTE_ARRAY def=5 rep=2
  m: map
    key_value: struct not null
      key: string not null <- BYTE_ARRAY def=2 rep=1
      value: int64 <- INT64 def=3 rep=1
  …
}

$ parquet-mojo cat tests/fixtures/nested.parquet --json --limit 2
{"li":[1,2,3],"lls":[["a"],["b","c"]],"m":{"k":1},"st":{"a":1,"b":"x"},…}
{"li":[],"lls":[],"m":{},"st":{"a":null,"b":"y"},"lst":[],"lb":[]}

$ parquet-mojo meta tests/fixtures/pageindex.parquet
```

`schema`, `meta` and `cat`, with `--columns a,b`, `--field-ids 1,2`,
`--limit N`, `--json` and `--no-crc`.

## Supported types, encodings and codecs

### Physical and logical types

| Parquet | Arrow | notes |
|---|---|---|
| `BOOLEAN` | `bool` | |
| `INT32` | `int32` | `Int(8\|16\|32, signed)` → `int8`…`uint32` |
| `INT64` | `int64` | `Int(64, false)` → `uint64` |
| `INT96` | `timestamp[ns]` | deprecated; Julian day + nanos, no time zone |
| `FLOAT` / `DOUBLE` | `float` / `double` | |
| `BYTE_ARRAY` | `binary` | `String`/`Enum` → `string`, `Json` → `string` + `arrow.json`, `Bson` → `binary` |
| `FIXED_LEN_BYTE_ARRAY` | `fixed_size_binary[n]` | `Float16` → `halffloat`, `UUID` → `fixed_size_binary[16]` + `arrow.uuid` |
| `Decimal` over `INT32`, `INT64`, `FLBA`, `BYTE_ARRAY` | `decimal128(p, s)` | sign-extended to Arrow's little-endian 16 bytes |
| `Date` | `date32[day]` | |
| `Time(ms)` / `Time(µs\|ns)` | `time32[ms]` / `time64[µs\|ns]` | |
| `Timestamp(ms\|µs\|ns, utc?)` | `timestamp[unit]`, `tz=UTC` when adjusted | |
| `List` | `list` | 3-level, and the 2-level backward-compatibility rules |
| `Map` / `MapKeyValue` | `map` | as `list<struct<key, value>>` |
| group | `struct` | |
| repeated primitive / repeated group | `list<T>` / `list<struct>` | |

`ConvertedType` is honoured wherever `LogicalType` is absent, which is what
files written before Parquet 2.4 have.

**Not mapped:** `Interval`, `Variant`, `Geometry`, `Geography` and `Unknown`
fall back to their physical type rather than raising, so a file that uses one
still reads. Large (64-bit offset) Arrow types are never produced: Parquet does
not record the distinction, so a pyarrow `large_string` column comes back as
`string` (pyarrow only restores it from its own `ARROW:schema` key/value
metadata, which this reader does not parse — see [Gaps](#gaps)).

### Encodings

| encoding | status |
|---|---|
| `PLAIN` | ✅ all physical types, booleans bit-packed, byte arrays length-prefixed |
| `PLAIN_DICTIONARY`, `RLE_DICTIONARY` | ✅ dictionary page + RLE indices with the bit-width byte |
| `RLE` (booleans) | ✅ with the 4-byte length prefix |
| `RLE` / `BIT_PACKED` (levels) | ✅ v1 length-prefixed, v2 lengths from the header, legacy MSB-first `BIT_PACKED` |
| `DELTA_BINARY_PACKED` | ✅ int32 and int64, blocks and miniblocks, 64-bit wrap-around |
| `DELTA_LENGTH_BYTE_ARRAY` | ✅ |
| `DELTA_BYTE_ARRAY` | ✅ byte arrays and `FIXED_LEN_BYTE_ARRAY` |
| `BYTE_STREAM_SPLIT` | ✅ float, double, int32, int64 and `FLBA` |
| `ALP` | ❌ not implemented (Parquet 2.12, no writer emits it yet) |

### Codecs

`ParquetReader` is parametrised on a `CodecSet`:

| codec | `DefaultCodecs` | `parquet.ext_full.AllCodecs` |
|---|---|---|
| `UNCOMPRESSED` | ✅ | ✅ |
| `SNAPPY` | ✅ (snappy.mojo, pure Mojo) | ✅ |
| `GZIP` | ✅ (avro.mojo's inflate; gzip, zlib and bare DEFLATE framing) | ✅ |
| `ZSTD` | ❌ | ✅ (zstd.mojo → libzstd) |
| `LZ4_RAW` | ❌ | ✅ (lz4.mojo → liblz4) |
| `LZ4` (Hadoop-framed, deprecated) | ❌ | ✅ |
| `BROTLI` | ❌ | ❌ |

```mojo
from parquet import ParquetReader
from parquet.ext_full import AllCodecs      # -I ../zstd.mojo/src -I ../lz4.mojo/src

var r = ParquetReader[AllCodecs].open("part-0.parquet")
```

**Brotli is not implemented** in either set: there is no Mojo Brotli and no
magmalake shim for one. A Brotli column raises a message that says so, and every
other column of the same file still reads.

## For Iceberg

`iceberg.mojo` reaches everything it needs through the package root:

```mojo
from parquet import (
    ParquetReader, DefaultCodecs, CodecSet,     # the reader
    Table, RecordBatch, ArrayData, ArrowType,   # what it returns
    export_c, ExportedArray,                    # Arrow C Data Interface
    ParquetSchema, build_schema, LeafColumn,    # schema and field ids
    TypedStats, ScalarValue, decode_stats,      # lower/upper bounds
    Predicate, OP_EQ, OP_LT, OP_LE, OP_GT, OP_GE, OP_NE,
)
from parquet.bloom import BloomFilter, read_bloom_filter
from parquet.ext_full import AllCodecs         # optional ZSTD / LZ4
```

The four things Iceberg specifically wants:

* **field-id projection** — `reader.select_field_ids([1, 3, 7])` resolves
  `SchemaElement.field_id` anywhere in the tree, including nested elements.
  `select_field_ids_deep` goes one step further and projects *into* a nested
  column: give it the id of a struct field and the batch comes back as that
  struct with only that field in it, having decoded only the leaves under it.
  Ancestors are kept as wrappers, a map always keeps its key, and the roots
  come out in the order they were asked for;
* **`split_offsets`** — `reader.split_offsets()` is each row group's
  `file_offset`, falling back to its first column chunk's starting page, which
  is what `data_file.split_offsets` records;
* **bounds** — `reader.statistics(rg, leaf)` returns `TypedStats` with `min` and
  `max` already decoded to `Int64` / `UInt64` / `Float64` / bytes against the
  column's physical *and* logical type, plus `null_count` and
  `distinct_count`. The deprecated `min`/`max` fields are only trusted for
  types whose old sort order was correct;
* **bloom filters** — `read_bloom_filter(file_bytes, column_metadata)` returns
  a split-block filter with `might_contain_string` / `_i64` / `_f64` /
  `might_contain(ScalarValue)`, hashing exactly as the spec's PLAIN-encoding
  XXH64 does.

Row-group pruning is `reader.prune_row_groups([Predicate("k", OP_GE, ...)])`,
which returns how many row groups survived.

## Tests

`pixi run test` — **48 tests**, on `default` (nightly) and `stable` (Mojo
1.0.0), Linux and macOS. `pixi run -e codecs test-codecs` adds 6 more for ZSTD
and LZ4, including a write/read round trip through each and the
ZSTD-compressed Iceberg fixtures.

**pyarrow is the oracle.** `tools/gen_fixtures.py` writes 24 fixtures with
pyarrow 25.0.1 and `tools/oracle_pyarrow.py` reads each one back and dumps
**every value of every column** to a JSON file beside it — nulls as `null`,
floats as their exact IEEE-754 bits, decimals as their unscaled 128-bit
integer, binary as hex, timestamps as the integer they store, lists as arrays,
structs as objects, maps as arrays of pairs. The Mojo suite reproduces each
oracle from its own decode, value by value *and* as a CRC32 over a canonical
serialisation of the whole column — **over 20,000 value assertions across 23
fixtures**, and again at batch sizes 1, 3, 64 and 997 to prove batching does
not change anything.

Beyond value parity the suite covers:

- the Arrow type each logical type maps to, integer widths and signs, the
  extension types, `INT96`, and the level and shape of every nested fixture;
- the **two-level list** backward-compatibility rules, built from synthetic
  `SchemaElement`s (pyarrow 25 no longer writes 2-level lists, even with
  `use_compliant_nested_type=False` — it writes a 3-level list whose inner
  field is named `item`, which `legacy_list.parquet` does cover);
- **nine real Iceberg data files** from the sibling tins' test warehouses, six
  of them written by `parquet-rs 58` rather than pyarrow — a second writer, a
  root schema element named `arrow_schema`, top-level `required` columns
  (which pyarrow never writes), Iceberg field ids on every column, and a
  position-delete file using the reserved ids 2147483546 and 2147483545;
- projection by name and by field id, row-group selection, and pruning by
  integer, string and float statistics — including that a file *without*
  statistics prunes nothing;
- **page-level pruning** against `manypages.parquet` (20 pages per row group,
  disjoint ranges): a `k in [1200, 1210)` predicate must keep every matching
  row and drop most of the rest, a string predicate must do the same, an
  impossible predicate must leave nothing, a file with no page index must be
  left alone, pruning must compose with batching — and the same again against
  a page index **our own writer** produced;
- statistics decoded to typed min/max compared against pyarrow's, per column
  chunk of five fixtures;
- `split_offsets`, `created_by` and key/value metadata against the oracle;
- the `OffsetIndex` and `ColumnIndex` of every chunk of `pageindex.parquet`,
  and that `nostats.parquet` has neither;
- bloom filters: all 200 keys of each of three columns must be reported
  present, and at least 400 of 500 absent keys must be ruled out;
- the C Data Interface: format strings, flags, buffer counts, offsets and
  values for flat, nested and extension columns, and that `release` is
  idempotent;
- unit tests for bit widths, ULEB128, zigzag, hybrid RLE runs, legacy
  `BIT_PACKED`, PLAIN, dictionary gather, `BYTE_STREAM_SPLIT` and
  `DELTA_BINARY_PACKED` headers;
- the writer: every fixture through write → read → the original oracle, at
  four different option sets, and the schema shape, field ids, statistics,
  split offsets, page index and key/value metadata of what it wrote;
- **hostile input**: an empty file, a 7-byte file, bad leading and trailing
  magic, every one of 23 truncations of a real file, a flipped byte inside a
  checksummed page, eleven single-byte corruptions of a page header, a
  dictionary index out of range, and a level above the column maximum — each
  must raise, and none may crash or read out of bounds.

### The C Data Interface, verified against pyarrow

`pixi run verify-c` builds `tools/carrow_export.mojo` into a shared library
with one C entry point, and `tools/consume_c_data.py` dlopens it, hands the two
structs straight to `pyarrow.Array._import_from_c`, and asserts the imported
array equals what pyarrow itself reads from the same file — then drops it, so
pyarrow calls our `release` callback.

**53 columns across 10 fixtures** — flat primitives, decimals over all three
physical layouts, `INT96` timestamps, `UUID` and `Json` extension types,
`halffloat`, lists, lists of lists, maps, structs, lists of structs, an
all-null file and a legacy list file — import into pyarrow and compare equal.

### Fixtures

`tests/fixtures/PROVENANCE.md` lists all 24 with their size, row count and what
each covers, and `tests/fixtures/iceberg/PROVENANCE.md` says where each of the
nine Iceberg data files came from and what it is worth testing against. The
whole directory including the oracles is under 2 MB. Regenerate the pyarrow
half with `pixi run fixtures` (needs `uv`); the Iceberg files are copied
verbatim and are not regenerated.

## Performance

`pixi run bench` against `pixi run bench-pyarrow`, single threaded, Apple M4,
CRC verification off on both sides. Both timers cover the same thing — bytes
in memory to Arrow arrays, and Arrow arrays to bytes — and neither side times
a copy of the input file: `ParquetReader` takes ownership of the bytes, so
each repeat gets its own copy, made outside the timed region, exactly as
pyarrow gets a fresh `pa.py_buffer` over the same bytes. Best of N repeats.

### Reading

| file | parquet.mojo | pyarrow 25.0.1 | ratio |
|---|---|---|---|
| 1M rows × (int64, double, dictionary int64, dictionary string), 18 MiB | **4.3 ms** — 232 M rows/s, 4.3 GB/s | 8.2 ms — 122 M rows/s, 2.3 GB/s | **parquet.mojo 1.9×** |
| 100k rows × 5 mixed types incl. a list column, ~1% nulls, snappy, 1.3 MiB | 4.3 ms — 23 M rows/s, 329 MB/s | 2.3 ms — 44 M rows/s, 616 MB/s | pyarrow 1.9× |
| 1,000 rows × 3 columns, 20 KiB | **0.17 ms** — 6.0 M rows/s | 0.35 ms | **parquet.mojo 2.1×** |
| 400 rows × 12 columns, every encoding, 19 KiB | 0.12 ms — 3.3 M rows/s | 0.17 ms | **parquet.mojo 1.4×** |
| 500 rows, v2 pages, 13 KiB | **0.086 ms** — 5.8 M rows/s | 0.18 ms | **parquet.mojo 2.1×** |

### Writing

| file | parquet.mojo | pyarrow 25.0.1 | ratio |
|---|---|---|---|
| 1M rows × (int64, double, dictionary int64, dictionary string) | 41.6 ms — 24 M rows/s, 451 MB/s | 30.7 ms — 33 M rows/s, 622 MB/s | pyarrow 1.4× |
| 100k rows × 5 mixed types incl. a list column | 13.5 ms — 7.4 M rows/s, 162 MB/s | 8.3 ms — 12 M rows/s, 290 MB/s | pyarrow 1.6× |

Nothing here is threaded — Mojo's nightly has no reachable thread pool — so
every number is one core. pyarrow's writer numbers are single-threaded too.

### Where the time goes

`pixi run profile` splits a read into stages and prints the table, so an
optimisation can be aimed rather than guessed at. For the 1M-row file, in
milliseconds:

| stage | before | now |
|---|---|---|
| footer + page headers | 0.7 | 0.3 |
| decompression | 0.8 | 0.4 |
| definition / repetition levels | 32.3 | 0.03 |
| value decoding | 27.8 | 1.2 |
| dictionary gather | 9.1 | 3.0 |
| per-page concatenation | 2.5 | — |
| row index | — | 0.0 |
| Dremel assembly | 18.1 | 1.0 |
| **total** | **76.0** | **4.3** |

What changed, in order of what the profile said to fix:

* **Levels.** A page whose definition levels are all the column maximum — any
  column with no nulls — is recognised from its run headers and never
  materialised at all; `ColumnData.all_present` stands in for the array. Pages
  that do have nulls decode whole runs at a time, a repeated run as a fill and
  a bit-packed run as one unaligned 64-bit load, shift and mask per value.
* **The row index.** A column with one slot per row was building a
  row-to-slot array holding `k` at `k`, and a column with no nulls a
  row-to-value array holding the same thing. Neither is built now.
* **Values.** Dictionary indices decode straight into a `UInt32` buffer with
  no `UInt64` detour; PLAIN values and the dictionary gather write onto the
  end of the chunk's own buffer, so a page's bytes are copied once instead of
  twice; byte arrays move eight bytes at a time into a buffer with slack.
* **Assembly.** When every slot of a range becomes a row — any top-level
  column, null or not — the array is built one buffer at a time: the validity
  bitmap in one pass over the levels, the values in one pass that copies the
  present ones into place. The value-by-value path is left for decimals,
  `INT96` timestamps and narrow integers carried in a wider physical type,
  where the Arrow bytes are not the Parquet bytes.
* **Snappy.** snappy.mojo 0.1.1 moves literals and non-overlapping copies 16
  bytes at a time; the compressed benchmark file spent a third of its read in
  that loop.

The writer got the same treatment: `shred_flat` moves a whole unnested column
in one pass instead of a recursive call per row, the dictionary builder is an
open-addressed table keyed on the integer value rather than a
`Dict[UInt64, List[Int]]`, and the chunk's statistics are folded from its
pages instead of scanning every value a second time. 584 ms to 41.6 ms.

What is left, if you want to take it further: the remaining gap on the
nullable/nested file is Dremel assembly of the list column and the per-page
allocation of the decompression buffer, and the biggest single item on the
wide file is the dictionary gather. A `keep_dictionary` option that handed
back an Arrow dictionary array would remove the gather entirely for consumers
that can take one.

## Gaps

* **Brotli** — no Mojo implementation and no shim yet. Named explicitly when a
  file needs it.
* **Encryption** — a `PARE` footer or an encrypted column raises. Parquet
  modular encryption is not implemented.
* **`ARROW:schema` metadata** — pyarrow stores an Arrow IPC schema in the
  file's key/value metadata and uses it to restore types Parquet cannot record
  (`large_string`, `fixed_size_list`, dictionary-encoded columns as Arrow
  dictionaries). Reading it would need a Flatbuffers parser; this reader
  derives everything from the Parquet schema alone, like `parquet-mr` does. The
  values are identical, only the Arrow type differs.
* **Page-level skipping for repeated columns** — `prune_pages` uses the page
  index for flat columns; a column with a maximum repetition level above zero
  is left alone, because its pages do not line up with rows one to one in a
  way a simple predicate can use.
* **`marrow`** — kszucs/marrow is the Arrow-in-Mojo library this would
  otherwise build on, but it pins `mojo == 0.26.3.0.dev2026032105`, far older
  than Mojo 1.0.0, so it does not compile on either supported toolchain. The
  Arrow buffers here are in-repo (`parquet.arrow`) and the C Data Interface is
  the interoperability boundary instead.
* **Writer** — see below.

## Writer

`ParquetWriter` takes exactly what the reader produces — an `ArrayArena` and
the indices of the top-level arrays in it — so a round trip is four lines:

```mojo
from parquet import ParquetReader, ParquetWriter, WriterOptions

var r = ParquetReader.open("in.parquet")
var t = r.read_table()

var opts = WriterOptions()
opts.codec = CompressionCodec.ZSTD.value   # with parquet.ext_full.AllCodecs
opts.row_group_size = 100_000
var w = ParquetWriter[AllCodecs](opts^)
w.add_metadata("iceberg.schema", schema_json)
for b in t.batches:
    w.write_batch(b.arena, b.roots)
var bytes = w^.finish()
```

`WriterOptions` carries `codec`, `row_group_size`, `data_page_size`,
`use_dictionary`, `write_statistics`, `write_page_index` and `created_by`.

What it writes:

| | |
|---|---|
| metadata | Parquet 2.6 `FileMetaData`, `created_by`, key/value metadata |
| pages | v1 data pages, split at record boundaries by `data_page_size`; a dictionary page when the dictionary pays for itself |
| values | `PLAIN` and `RLE_DICTIONARY` (the dictionary is dropped for a column where it would be bigger than the data) |
| levels | `RLE`, with the 4-byte length prefix a v1 page wants |
| types | every type the reader maps, written back to the physical type it came from; decimals as `FIXED_LEN_BYTE_ARRAY(16)`, which is legal at any precision Arrow can hold |
| nesting | three-level lists, `key_value` maps, structs — the compliant layouts |
| **field ids** | on every schema element that has one, which is what Iceberg needs |
| statistics | `min_value` / `max_value` / `null_count` per chunk, in the sort order each logical type requires (unsigned integers unsigned, decimals by sign then magnitude, NaNs excluded) |
| page index | an `OffsetIndex` and a `ColumnIndex` per chunk, with per-page bounds and null counts |

Not written: v2 data pages, the delta and byte-stream-split encodings, page
CRC32s, `INT96`, bloom filters, and encryption.

**pyarrow reads what we write.** `pixi run verify-written` writes all 22
fixtures back out through our writer — plus three of them again with the
dictionary off, uncompressed, and with GZIP at a 4-row row group and a 32-byte
page — then has pyarrow read every one and compare it with the original.
**29 files, 175 columns, all equal.** In the Mojo suite,
`test_write_round_trip` sends every fixture through write → read and checks the
result against the *original* pyarrow oracle, value by value.

## Tasks

| task | what it does |
|---|---|
| `pixi run test` | the test suite (`-e stable` for Mojo 1.0.0) |
| `pixi run check` | the tests plus a build of the CLI |
| `pixi run -e codecs test-codecs` | the ZSTD / LZ4 tests |
| `pixi run bench` | the decode and encode benchmarks |
| `pixi run profile` | per-stage decode profile: where a read spends its time |
| `pixi run bench-pyarrow` | pyarrow's numbers for the same files (needs `uv`) |
| `pixi run cli schema f.parquet` | build and run the CLI |
| `pixi run verify-c` | import the C Data Interface export into pyarrow |
| `pixi run verify-written` | write every fixture back out and have pyarrow read it |
| `pixi run fixtures` | regenerate the fixtures and their oracles |

## License

Apache-2.0. Fixtures are generated by pyarrow and carry no third-party code.
