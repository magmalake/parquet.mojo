"""Write every fixture out through our writer, for an encoding comparison.

Two Parquet files that pick different encodings for the same column are both
valid, so a change to the writer's dictionary rule can shrink or grow every
file it touches without a single test noticing. This writes the whole fixture
corpus — plus `build/bench-wide.parquet` when it exists — through the writer
with its default options, and `tools/encoding_report.py` then reports the
encoding and the size of every column chunk so two branches can be compared
column by column.

```console
pixi run -e codecs encodings                 # -> build/encodings/
pixi run -e codecs encodings /some/other/dir
```

`AllCodecs` rather than `DefaultCodecs` so `codecs.parquet`, whose columns are
one per codec, is in the corpus too.
"""

from fixtures_list import core_fixtures
from parquet import ParquetReader, ParquetWriter, WriterOptions
from parquet.ext_full import AllCodecs
from std.os.path import exists
from std.sys import argv


def write_bytes(path: StringSlice, data: Span[UInt8, _]) raises:
    var f = open(String(path), "w")
    f.write_bytes(data)
    f.close()


def round_trip(
    source: StringSlice, out_dir: StringSlice, name: StringSlice
) raises:
    var r = ParquetReader[AllCodecs].open(String(source))
    var t = r.read_table()
    var w = ParquetWriter[AllCodecs](WriterOptions())
    w.add_metadata("writer", "parquet.mojo")
    for b in t.batches:
        w.write_batch(b.arena, b.roots)
    var bytes = w^.finish()
    var out = String(out_dir, "/", name, ".parquet")
    write_bytes(out, Span(bytes))
    print(out, len(bytes), "bytes")


def main() raises:
    var args = argv()
    var out_dir = String("build/encodings")
    if len(args) > 1:
        out_dir = String(args[1])
    var names = core_fixtures()
    names.append(String("codecs"))
    for name in names:
        round_trip(String("tests/fixtures/", name, ".parquet"), out_dir, name)
    if exists("build/bench-wide.parquet"):
        round_trip("build/bench-wide.parquet", out_dir, "bench-wide")
    else:
        print("build/bench-wide.parquet: missing, skipped")
