"""The codec set that includes ZSTD, LZ4 and BROTLI.

`zstd.mojo`, `lz4.mojo` and `brotli.mojo` dlopen libzstd / liblz4 / libbrotli,
so a consumer that wants these has to have those tins installed *and* on the
include path:

```console
mojo build … -I ../zstd.mojo/src -I ../lz4.mojo/src -I ../brotli.mojo/src
```

Everything `DefaultCodecs` handles is handled here too. Between the two sets
that is every codec in the Parquet spec.
"""

import brotli
import lz4
import zstd
from parquet.codec import (
    CodecSet,
    DefaultCodecs,
    copy_of,
    gunzip,
    gzip_wrap,
    unsupported_codec,
)
from snappy import compress as snappy_compress, decompress as snappy_decompress
from thrift import CompressionCodec


struct AllCodecs(CodecSet):
    """`UNCOMPRESSED`, `SNAPPY`, `GZIP`, `ZSTD`, `BROTLI`, `LZ4_RAW` and legacy
    `LZ4` — every codec the Parquet spec defines."""

    @staticmethod
    def supports(codec: Int32) -> Bool:
        return (
            DefaultCodecs.supports(codec)
            or codec == CompressionCodec.ZSTD.value
            or codec == CompressionCodec.BROTLI.value
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
        if codec == CompressionCodec.BROTLI.value:
            # Brotli records no uncompressed size anywhere in the stream, so
            # the page header's is the only one there is.
            return brotli.decompress(data, uncompressed_size)
        if codec == CompressionCodec.LZ4_RAW.value:
            # A raw LZ4 block: the uncompressed size comes from the page header.
            return lz4.decompress_block(data, uncompressed_size)
        if codec == CompressionCodec.LZ4.value:
            # The deprecated `LZ4` codec never pinned its framing down, so
            # files in the wild carry either the Hadoop wrapper — repeated
            # <big-endian uncompressed size><big-endian block size><block> —
            # or a bare LZ4 block with no wrapper at all. Nothing in the file
            # says which, so the only way to tell is to try the wrapper and
            # fall back. parquet-testing ships one of each.
            try:
                return lz4.decompress_hadoop(data, uncompressed_size)
            except:
                return lz4.decompress_block(data, uncompressed_size)
        raise unsupported_codec(codec)

    @staticmethod
    def compress(codec: Int32, data: Span[UInt8, _]) raises -> List[UInt8]:
        if codec == CompressionCodec.UNCOMPRESSED.value:
            return copy_of(data)
        if codec == CompressionCodec.SNAPPY.value:
            return snappy_compress(data)
        if codec == CompressionCodec.GZIP.value:
            return gzip_wrap(data)
        if codec == CompressionCodec.ZSTD.value:
            return zstd.compress(data)
        if codec == CompressionCodec.BROTLI.value:
            # Quality 5, not libbrotli's default of 11: at 11 the encoder runs
            # at single-digit MB/s, which would make a Brotli write two orders
            # of magnitude slower than a ZSTD one for a few percent of size.
            return brotli.compress(data, quality=5)
        if codec == CompressionCodec.LZ4_RAW.value:
            return lz4.compress_block(data)
        raise unsupported_codec(codec)
