# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases before 0.8.0 predate this file; their contents are in the commit log
(each release is one commit whose subject begins with its version).

## [Unreleased]

## [parquet-full-mojo 0.1.1] - 2026-09-11

`parquet-full-mojo` 0.1.0 was published by a `shelf` older than the one that
knows about subdirectory tins, so the registry recorded no subdirectory for it
and `pixi shelf add parquet-full-mojo` failed with "the package
'parquet-full-mojo' is not provided by the project located at
git+…/parquet.mojo". 0.1.1 is the same code, published by `shelf` 0.5.0, which
sends where the manifest sits.

0.1.0 resolves for nobody and should not be used. A direct git dependency
naming `subdirectory = "full"` was unaffected either way.

`parquet-mojo` 0.11.0 is unaffected — it publishes from the repository root,
where there is no subdirectory to record.


## [0.11.0] - 2026-09-11

Two dependency changes, both about what reading a Parquet file should oblige
you to build. Neither touches a Parquet API, codec behaviour or file output.

### Apache Avro is gone from the dependency list

The raw DEFLATE ([RFC 1951](https://www.rfc-editor.org/rfc/rfc1951)) behind the
`GZIP` codec had lived in `avro.mojo` since Avro's `deflate` block codec needed
it first, so every consumer of this tin resolved `avro-mojo` for one `inflate`.
It now has its own tin, [deflate.mojo](https://github.com/magmalake/deflate.mojo)
— still pure Mojo, still no FFI.

- `avro-mojo` is gone from `[package.host-dependencies]` and
  `[package.run-dependencies]`; `deflate-mojo` takes its place.
- Building from **source paths** swaps `-I ../avro.mojo/src` for
  `-I ../deflate.mojo/src`. The test suite keeps `-I ../avro.mojo/src`, because
  `tests/oracle.mojo` parses the oracle JSON with `avro.json`; nothing under
  `src/` does.

### The FFI codecs are a separate tin (breaking)

`ZSTD`, `BROTLI`, `LZ4_RAW` and the legacy Hadoop-framed `LZ4` — and with them
libzstd, libbrotli, liblz4 and the three cmake shims that dlopen them — are now
**`parquet-full-mojo`**, published from [`full/`](full) of this same repository.
`parquet-mojo` reads and writes Parquet with `UNCOMPRESSED`, `SNAPPY` and
`GZIP`, and links no third-party C at all.

`CodecSet` already made the codecs a compile-time choice; this carries that
choice up into the package graph, so a consumer who never needs ZSTD never
resolves or builds the C libraries either.

- **`parquet.ext_full` is now the `parquet_full` package.** One line:

  ```mojo
  from parquet_full import AllCodecs    # was: from parquet.ext_full import AllCodecs
  ```

- **`pixi shelf add parquet-mojo` no longer brings the FFI codecs.** A reader
  that meets one of them raises and names `AllCodecs`, as it always has for a
  `DefaultCodecs` reader. **If you read Parquet you did not write, you almost
  certainly want `pixi shelf add parquet-full-mojo`** — ZSTD is the common
  case, Iceberg's default included.
- Building from **source paths** needs `-I full/src` (or
  `-I ../parquet.mojo/full/src`) for `parquet_full`.

Both tins live on one commit and move together: `full/` depends on the root as
`parquet-mojo = { path = ".." }`, so there is no re-pinning between them and no
second repository, CI matrix or release. This needs mojoshelf's subdirectory
support (mojoshelf/mojoshelf#18).

## [0.9.1] - 2026-09-08

Builds on September nightlies again. 0.9.0 shipped with `nightly` and `gpu`
capped below `26.6.0.dev2026090105`, because threads-mojo reached the atomics
through compiler intrinsics that stopped parsing there. threads-mojo 0.5.0
moved onto `std.atomic`, so the cap is gone and both environments now take
`>=26.6.0.dev2026090705`.

Building from **source paths** needs one more include beside
`-I ../threads.mojo/src`: the compat directory holding the one line that
differs between Mojo 1.0.0 and nightly. This repository sets `$THREADS_COMPAT`
per feature, so no task names a toolchain. Consumers of the published tin need
nothing.

## [0.9.0] - 2026-09-08

The Arrow layer is no longer here. `ArrayData`, `ArrayArena`, `RecordBatch` and
both directions of the C Data Interface now live in
[arrow-mlake-mojo](https://mojoshelf.org/tins/arrow-mlake-mojo), which this tin
depends on; `parquet.arrow`, `parquet.carrow` and `parquet.carrow_import` are
re-export shims, so **no import path a consumer writes today changes**. That was
verified by running iceberg-mojo's full 203-test suite against this release with
no source edits at all.

Consumers building from **source paths** need one addition —
`-I ../arrow-mlake.mojo/src` — because a `-I` list has to name every package it
resolves. Consumers taking the published tin need nothing: it arrives as a run
dependency.

### Fixed
- **`mojo format` works again, and CI now gates on it.** Thirteen of this
  repo's files could not be formatted at all: the formatter aborted with
  `'_python_symbols' object has no attribute 'old_comptime_assert_stmt'`, on
  both toolchains, including files no recent change had touched. It was a
  defect in the formatter rather than anything about the code — the same files
  come back unchanged from a nightly at or after `26.6.0.dev2026090705`, which
  the lock now resolves. `format-check` runs on the nightly leg of CI; it
  cannot run on stable, which is pinned to Mojo 1.0.0 and still carries the
  bug. The formatter lives in its own `fmt` environment because the build
  environments had to be **capped below `26.6.0.dev2026090105`** in the same
  breath: `threads.mojo`'s `atomic.mojo` stopped compiling above it with
  `invalid MLIR attribute: expected '<'`, so moving one compiler to fix
  formatting broke every environment that builds against the sibling source
  paths. `mojo format` only parses, so the two can and now do move separately.

### Changed
- **The Arrow layer moved to
  [arrow-mlake.mojo](https://github.com/magmalake/arrow-mlake.mojo)**, and this
  library now depends on it. `ArrayData`, `ArrayArena`, `ArrowType`, the `AT_*`
  tags, the bitmap and unaligned-load helpers, both directions of the C Data
  Interface and `RecordBatch` all left `src/parquet`. The Arrow memory layout
  is not Parquet's; it lived here only because this is where it was first
  needed, and it moved out when `lancedb.mojo` needed the same C Data Interface
  and taking `parquet-mojo` for it would have turned a binding with no tin
  dependencies at all into one with nine, three of them compression codecs and
  one a Thrift implementation it would never execute.
  - **No consumer source has to change.** `parquet.arrow`, `parquet.carrow` and
    `parquet.carrow_import` are re-export shims, and `RecordBatch` and the
    `array_*` kernels are re-exported from `parquet` and `parquet.reader`
    alike, so `from parquet.arrow import ArrayData` and
    `from parquet.reader import RecordBatch` both still resolve to the same
    types. iceberg-mojo 0.7.2 builds against this unchanged, and its 203 tests
    pass.
  - A consumer that takes `parquet-mojo` as a tin gets `arrow-mlake-mojo` as a
    run dependency. A consumer that puts sources on the include path must add
    **`-I ../arrow-mlake.mojo/src`**; that is the one thing this change asks
    for.
  - `pixi run verify-c-import` and `tools/carrow_import.mojo` /
    `tools/produce_c_data.py` moved too — the pyarrow-as-producer gate belongs
    with the importer it gates. `verify-c` (pyarrow reading *our* export) stays
    here, because what it exercises is the Parquet decoder feeding the export.
  - Six unit tests went with the code: the format-string round trip, the
    formats we refuse, the struct-to-`RecordBatch` unwrap and the three
    `ArrowArrayStream` tests. What stays is the integration half — every column
    of every Parquet fixture exported and imported back — which is coverage of
    the decoder as much as of the interface. 101 tests here, 34 there.

### Added
- **The import half of the Arrow C Data Interface** (`parquet.carrow_import`).
  `parquet.carrow` could only ever hand arrays *out*; nothing could read a C
  `ArrowSchema`/`ArrowArray` back in, which meant no C or Rust producer could
  hand Arrow data into this stack — not DuckDB, not Arrow Flight, not an ADBC
  driver, and not LanceDB, whose analytical scan was blocked on exactly this.
  - `import_c(arena, array, schema)` copies one foreign array and everything
    under it into an `ArrayArena` and returns its root, in DFS **pre-order** —
    the order the exporter walks on the way out, so `export → import → export`
    reproduces the arena's shape and not merely its numbers.
  - `import_batch_c` unwraps a `+s` root into a `RecordBatch`, one column per
    field.
  - `ImportedArray` and `ImportedStream` own what a producer moved to them and
    implement the consumer's half of the release convention: the root is
    released exactly once, children never on their own, and a released struct
    is recognised by its null `release` pointer. `ImportedStream` drives
    `ArrowArrayStream` — `get_schema`, then `get_next` until a released array
    ends the stream, with `get_last_error` carried into the raise so a failed
    scan is not mistaken for a short one — and releases each batch as soon as
    it has been copied, so a long scan holds one batch of producer memory
    rather than the whole stream.
  - A producer's `offset` is honoured by materialising the window it names,
    for validity bitmaps (re-packed bit by bit when the start is unaligned),
    values, offsets, and recursively for the children a list's offsets point
    into. `ArrayData` has no offset field, so this is the only faithful
    reading of a sliced array.
  - **Nothing about a producer is trusted.** `n_buffers` is checked against
    the format string, `n_children` against the type, offsets for
    monotonicity and against the length of the child they index into, and
    metadata counts before they are used as loop bounds. Every format string
    this library cannot name raises **carrying the string**: unions, run-end
    encoding, list views, fixed-size lists, string views, `date64`,
    durations, intervals, decimal256 and dictionary-encoded arrays. The
    interface carries no buffer *sizes*, so a truncated `utf8` data buffer is
    undetectable by construction; that limit is documented rather than
    papered over.
  - Importing **copies**, because `ArrayData` owns its buffers. On a
    19-column, 65 536-row NYC-taxi batch (8.98 MB of Arrow buffers) that is
    2.2 ms at p50 and 2.4 ms at p90 — about 4 GB/s, and 6.6% of what decoding
    the same batch out of Parquet costs. A borrowing `ArrayData` would avoid
    it but changes the type every other module reads, and the measurement
    says it would be buying a rounding error.
- `pixi run verify-c-import` — **pyarrow as the producer**.
  `tools/produce_c_data.py` builds 63 cases, `_export_to_c`s them into ctypes
  storage and hands them to `tools/carrow_import.mojo`: every primitive width,
  utf8 and binary in both offset widths, `bool`, `null`, decimals, dates,
  times, timestamps with and without a zone, the four nested shapes, empty and
  all-null arrays, 24 slices at unaligned starts, a `RecordBatchReader` as an
  `ArrowArrayStream`, and two malformed producers. Each array is checked both
  on the type we parsed its format string into and by re-exporting it for
  pyarrow to compare. Runs in the `pyarrow parity` CI job.
- 14 tests in `tests/test_parquet.mojo` and the helpers in
  `tests/carrow_check.mojo`, including a hand-written C `ArrowArrayStream`
  producer. The round trip is compared on **values and arena layout
  separately**: a walk from the root cannot see a consistently permuted arena
  — the trap `tests/fingerprint.mojo` records — so `assert_preorder` checks
  the arena's own numbering, and `permuted_arena` is the negative control that
  shows the walk missing what the layout check catches. Three more negative
  controls corrupt an exported values byte, validity bit and offset and assert
  the comparison fails.

### Changed
- `parquet.carrow.n_buffers_for_type` splits the buffer count out of
  `n_buffers_for`, so the export and the import's validation read it from one
  place.

## [0.8.0] - 2026-09-07

A projected scan reads only the bytes it needs. On the eight-query
[taxibench](https://github.com/magmalake/taxibench.example) suite, through
iceberg-mojo, the single-threaded total went from 4054 ms to 3225 ms and the
ten-thread total from 1540 ms to 906 ms. The query that projects all nineteen
columns does not move, which is the check that this is a projection effect and
not an artefact.

Two null results came with it, and they are worth as much as the change: the
decoder itself had **no headroom worth taking** — column-for-column it is
already about 8.5% cheaper than pyarrow's — and neither late materialisation
nor the page index was worth building, because Arrow's own scanner does the
first not at all and ignores the second, and these files carry no page index
to read.

### Added
- **`ParquetReader.open_projected(path, columns)` — read only the bytes the
  projection needs.** `open` slurps the whole file, which is right when a
  caller wants all of it and pure waste when it does not. Four columns out of
  nineteen on a 60 MiB NYC-taxi file is 13 MiB of column chunks; reading the
  other 47 MiB cost **more than decompressing the 13** (`read_table` on that
  file: 13.8 ms → 7.7 ms for one column, 32.5 ms → 26.6 ms for four, p50 of
  five). The buffer is still the file's full length — Parquet addresses pages
  by absolute offset, so a sparse buffer decodes exactly as the whole file
  does — but the gap is left *uninitialised* rather than zeroed, so those
  pages are never faulted in. Zero-filling instead gives most of the saving
  back. The returned reader is projected already and must not be re-projected.
- **`ParquetReader.needed_byte_ranges()`** — the `(offset, length)` runs the
  current projection and row-group selection will actually read, coalesced and
  in file order. This is the piece a reader over a network wants: it turns
  "download the object" into one range request per run of wanted chunks.
  `iceberg-mojo` uses it with its own `FileIO`. The list covers each projected
  column chunk, the `ColumnIndex` and `OffsetIndex` of **every** column (page
  pruning reads the index of whatever the predicate names, which need not be a
  projected column), and a bloom filter when the file records its length.
- **`footer_start_of` / `footer_only_buffer` / `ParquetReader.fill_range`** —
  the same idea for a caller that owns its I/O: locate the footer from a
  file's last 8 bytes, build the file-length buffer that holds only the
  footer, resolve a projection against a reader over it, then write the
  fetched chunks in. Anything unexpected about the footer reports it rather
  than guessing, so the whole path is an optimisation and never a correctness
  condition.
- `bench/profile_scan.mojo` (`pixi run -e codecs profile-scan`) — a per-stage
  profile of a *projected* scan over a real file, with the bytes each stage
  moved and the bandwidth that implies printed beside the time, because a
  stage already at memory bandwidth cannot be made faster, only skipped. It
  also times `open` against `open_projected` on the same projection, having
  first checked the two agree on the row count.
