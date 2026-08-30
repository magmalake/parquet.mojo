"""The codec set that includes ZSTD and LZ4.

`zstd.mojo` and `lz4.mojo` dlopen libzstd / liblz4, so a consumer that wants
these has to have those tins installed *and* on the include path:

```console
mojo build … -I ../zstd.mojo/src -I ../lz4.mojo/src
```

Everything `DefaultCodecs` handles is handled here too.
"""

import lz4
import zstd
from parquet.codec import CodecSet, DefaultCodecs, copy_of, gunzip, unsupported_codec
from snappy import decompress as snappy_decompress
from thrift import CompressionCodec


struct AllCodecs(CodecSet):
    """`UNCOMPRESSED`, `SNAPPY`, `GZIP`, `ZSTD`, `LZ4_RAW` and legacy `LZ4`."""

    @staticmethod
    def supports(codec: Int32) -> Bool:
        return (
            DefaultCodecs.supports(codec)
            or codec == CompressionCodec.ZSTD.value
            or codec == CompressionCodec.LZ4_RAW.value
            or codec == CompressionCodec.LZ4.value
        )

    @staticmethod
    def decompress(
        codec: Int32, data: Span[UInt8, _], uncompressed_size: Int
    ) raises -> List[UInt8]:
        if codec == CompressionCodec.UNCOMPRESSED.value:
            return copy_of(data)
        if codec == CompressionCodec.SNAPPY.value:
            return snappy_decompress(data)
        if codec == CompressionCodec.GZIP.value:
            return gunzip(data)
        if codec == CompressionCodec.ZSTD.value:
            return zstd.decompress(data)
        if codec == CompressionCodec.LZ4_RAW.value:
            # A raw LZ4 block: the uncompressed size comes from the page header.
            return lz4.decompress_block(data, uncompressed_size)
        if codec == CompressionCodec.LZ4.value:
            # The deprecated Hadoop framing: repeated
            # <big-endian uncompressed size><big-endian block size><block>.
            return lz4.decompress_hadoop(data, uncompressed_size)
        raise unsupported_codec(codec)
