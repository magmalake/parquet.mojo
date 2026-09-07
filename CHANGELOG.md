# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases before 0.8.0 predate this file; their contents are in the commit log
(each release is one commit whose subject begins with its version).

## [Unreleased]

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
