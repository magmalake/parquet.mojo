"""Round-trip every fixture through our writer into `build/written/`.

`tools/verify_written.py` then has pyarrow read both the original and our copy
and asserts they hold the same values — the other half of the parity contract,
because a writer that only its own reader can read is not a Parquet writer.

```console
mojo build tools/write_fixtures.mojo -I src … -o build/write-fixtures
./build/write-fixtures
```
"""

from fixtures_list import core_fixtures
from parquet import ParquetReader, ParquetWriter, WriterOptions
from thrift import CompressionCodec


def write_bytes(path: StringSlice, data: Span[UInt8, _]) raises:
    var f = open(String(path), "w")
    f.write_bytes(data)
    f.close()


def round_trip(
    name: StringSlice, var options: WriterOptions, suffix: StringSlice
) raises:
    var r = ParquetReader.open(String("tests/fixtures/", name, ".parquet"))
    var t = r.read_table()
    var w = ParquetWriter(options^)
    w.add_metadata("writer", "parquet.mojo")
    for b in t.batches:
        w.write_batch(b.arena, b.roots)
    var bytes = w^.finish()
    var out = String("build/written/", name, suffix, ".parquet")
    write_bytes(out, Span(bytes))
    print(out, len(bytes), "bytes")


def main() raises:
    for name in core_fixtures():
        var snappy = WriterOptions()
        round_trip(name, snappy^, "")
    # A second pass with the other knobs, on a few representative fixtures.
    var names: List[String] = [
        String("primitives"),
        String("nested"),
        String("logical"),
    ]
    for name in names:
        var plain = WriterOptions()
        plain.use_dictionary = False
        plain.codec = CompressionCodec.UNCOMPRESSED.value
        round_trip(name, plain^, "-plain")
        var gz = WriterOptions()
        gz.codec = CompressionCodec.GZIP.value
        gz.row_group_size = 4
        gz.data_page_size = 32
        round_trip(name, gz^, "-gzip")
