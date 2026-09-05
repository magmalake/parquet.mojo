"""Page compression codecs.

`ParquetReader` is parametrised on a `CodecSet`, a compile-time table of page
decompressors, exactly like `avro.mojo`'s block codecs. The default set covers
everything that needs no FFI:

| codec | where it comes from |
|---|---|
| `UNCOMPRESSED` | — |
| `SNAPPY` | `snappy.mojo`, pure Mojo |
| `GZIP` | `avro.mojo`'s pure-Mojo `deflate.inflate`, with the gzip wrapper parsed here |

`ZSTD`, `BROTLI`, `LZ4_RAW` and the legacy Hadoop-framed `LZ4` need `zstd.mojo`,
`brotli.mojo` and `lz4.mojo`, which dlopen libzstd, libbrotli and liblz4, so
they live in `parquet.ext_full` and its `AllCodecs`:

```mojo
from parquet import ParquetReader
# -I ../zstd.mojo/src -I ../lz4.mojo/src -I ../brotli.mojo/src
from parquet.ext_full import AllCodecs

var r = ParquetReader[AllCodecs].open("part-0.parquet")
```

Between the two sets that is every codec the Parquet spec defines. A reader on
`DefaultCodecs` that meets one of the four FFI codecs raises a message naming
it and pointing at `AllCodecs`; every other column of that file still reads.
"""

from avro.deflate import deflate, inflate, inflate_at
from hashes import crc32
from snappy import compress as snappy_compress, decompress as snappy_decompress
from thrift import CompressionCodec


trait CodecSet:
    """A compile-time table of Parquet page decompressors."""

    @staticmethod
    def supports(codec: Int32) -> Bool:
        """Whether this set can decompress `CompressionCodec` value `codec`."""
        ...

    @staticmethod
    def decompress[
        page_origin: ImmOrigin, scratch_origin: MutOrigin
    ](
        codec: Int32,
        data: Span[UInt8, page_origin],
        uncompressed_size: Int,
        ref[scratch_origin] scratch: List[UInt8],
    ) raises -> Span[UInt8, origin_of(page_origin, scratch_origin)].Immutable:
        """Decompress one page body, borrowing rather than allocating.

        The result is *a view*, never a fresh buffer. For `UNCOMPRESSED` it is
        `data` itself — the page's own bytes inside the file, handed back
        untouched, which is what `SerializedPageReader::DecompressIfNeeded`
        does in parquet-cpp. For every other codec it is `scratch`, the
        caller's one decompression buffer per column chunk.

        **Lifetime contract — the span is valid only until the next call that
        passes the same `scratch`.** That call overwrites the buffer and may
        reallocate it, so a span held across a page boundary reads another
        page's bytes. parquet-cpp documents the same rule at
        `column_reader.cc:251`: the previous page is invalidated by
        `NextPage()`. Copy out anything that has to outlive the page —
        `decode_plain_into` and its neighbours already do, which is why a
        dictionary page survives the data pages decompressed after it.

        Mojo's origins cannot make the compiler enforce that. The returned
        origin is the union of the file's and the scratch buffer's, so both are
        kept alive for as long as the span is and it can never dangle, but Mojo
        does not treat an outstanding immutable view as blocking a later
        mutation the way an exclusivity checker would — a held span is a wrong
        answer, not a compile error. So the rule is a contract, and
        `test_a_page_span_is_only_valid_until_the_next_page` is what states it.

        `uncompressed_size` is the page header's declared decompressed length,
        which for Brotli and raw LZ4 is the only size there is.
        """
        ...

    @staticmethod
    def compress(codec: Int32, data: Span[UInt8, _]) raises -> List[UInt8]:
        """Compress one page body. Used by `parquet.writer`."""
        ...


def copy_of(data: Span[UInt8, _]) -> List[UInt8]:
    var out = List[UInt8](capacity=len(data))
    out.extend(data)
    return out^


def page_span[origin: ImmOrigin](data: Span[UInt8, _]) -> Span[UInt8, origin]:
    """Re-label `data` with the origin `CodecSet.decompress` returns.

    Its two answers live in different places — the file for an uncompressed
    page, the caller's scratch buffer for a compressed one — and the declared
    return type is the union of those two origins, which keeps both alive for
    as long as the span is. Neither branch can produce that union type on its
    own, hence this one narrow re-labelling, and it is a no-op at run time: the
    same pointer and length.

    Sound because the union outlives either input, so the origin stamped on is
    never longer-lived than the one the bytes actually have.
    """
    return Span[UInt8, origin](
        unsafe_ptr=Pointer[UInt8, origin](
            unsafe_from_address=Int(data.unsafe_ptr())
        ),
        length=len(data),
    )


def codec_name(codec: Int32) -> String:
    return CompressionCodec(codec).name()


def unsupported_codec(codec: Int32) -> Error:
    return Error(
        String(
            "parquet: page codec ",
            codec_name(codec),
            (
                " is not available — use parquet.ext_full.AllCodecs for"
                " ZSTD/BROTLI/LZ4 (needs -I ../zstd.mojo/src"
                " -I ../brotli.mojo/src -I ../lz4.mojo/src)"
            ),
        )
    )


def _inflate_one_member(
    data: Span[UInt8, _], start: Int, mut out: List[UInt8]
) raises -> Int:
    """Inflate the gzip member at `start`, appending it to `out`.

    Returns the offset just past the member's 8-byte CRC32/ISIZE trailer,
    which is where the next member begins if there is one.
    """
    if start + 18 > len(data):
        raise Error("parquet.gzip: truncated gzip member")
    if data[start + 2] != 8:
        raise Error(
            String(
                "parquet.gzip: unsupported compression method ", data[start + 2]
            )
        )
    var flags = data[start + 3]
    var pos = start + 10
    if (flags & 0x04) != 0:  # FEXTRA
        if pos + 2 > len(data):
            raise Error("parquet.gzip: truncated FEXTRA")
        var xlen = Int(data[pos]) | (Int(data[pos + 1]) << 8)
        pos += 2 + xlen
    if (flags & 0x08) != 0:  # FNAME
        while pos < len(data) and data[pos] != 0:
            pos += 1
        pos += 1
    if (flags & 0x10) != 0:  # FCOMMENT
        while pos < len(data) and data[pos] != 0:
            pos += 1
        pos += 1
    if (flags & 0x02) != 0:  # FHCRC
        pos += 2
    if pos >= len(data):
        raise Error("parquet.gzip: gzip header runs past the page")
    var end = 0
    var body = inflate_at(data[pos:], end)
    out.extend(Span(body))
    # `end` is relative to the slice; + 8 steps over CRC32 and ISIZE.
    return pos + end + 8


def gunzip(data: Span[UInt8, _]) raises -> List[UInt8]:
    """Strip a gzip (RFC 1952) wrapper and inflate the DEFLATE stream inside.

    Parquet's `GZIP` codec is gzip-framed, not a bare DEFLATE stream, so the
    10-byte header and its optional extra/name/comment/CRC fields have to come
    off first. A zlib (RFC 1950) stream is accepted too — some writers emit
    one — and so is a bare DEFLATE stream.

    A gzip stream may be several members concatenated (RFC 1952 §2.2), and a
    page written that way decodes to the members joined in order. Stopping at
    the first one silently loses the rest, so the members are looped over here.
    """
    if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
        var out = List[UInt8]()
        var at = 0
        while at < len(data):
            # Trailing NUL padding after the last member is legal and some
            # writers emit it; anything else that is not a header is corrupt.
            if data[at] == 0:
                at += 1
                continue
            if at + 2 > len(data) or data[at] != 0x1F or data[at + 1] != 0x8B:
                raise Error(
                    "parquet.gzip: trailing bytes are not a gzip member"
                )
            at = _inflate_one_member(data, at, out)
        return out^
    if (
        len(data) >= 2
        and (data[0] & 0x0F) == 8
        and ((UInt16(data[0]) * 256 + UInt16(data[1])) % 31) == 0
    ):
        # RFC 1950 zlib wrapper: 2-byte header, 4-byte Adler-32 trailer.
        return inflate(data[2 : len(data) - 4])
    return inflate(data)


def gzip_wrap(data: Span[UInt8, _]) raises -> List[UInt8]:
    """Wrap a DEFLATE stream in the gzip (RFC 1952) framing Parquet's `GZIP`
    codec expects: a 10-byte header, then the deflate bits, then the CRC32 of
    the *uncompressed* data and its length modulo 2^32."""
    var out = List[UInt8]()
    out.append(0x1F)
    out.append(0x8B)
    out.append(8)  # deflate
    out.append(0)  # no optional fields
    for _ in range(4):
        out.append(0)  # mtime
    out.append(0)  # no extra flags
    out.append(0xFF)  # unknown OS
    out.extend(Span(deflate(data)))
    var sum = crc32(data)
    for k in range(4):
        out.append(UInt8((sum >> UInt32(8 * k)) & 0xFF))
    var n = UInt32(len(data) & 0xFFFFFFFF)
    for k in range(4):
        out.append(UInt8((n >> UInt32(8 * k)) & 0xFF))
    return out^


struct DefaultCodecs(CodecSet):
    """`UNCOMPRESSED`, `SNAPPY` and `GZIP` — no FFI, no shared libraries."""

    @staticmethod
    def supports(codec: Int32) -> Bool:
        return (
            codec == CompressionCodec.UNCOMPRESSED.value
            or codec == CompressionCodec.SNAPPY.value
            or codec == CompressionCodec.GZIP.value
        )

    @staticmethod
    def decompress[
        page_origin: ImmOrigin, scratch_origin: MutOrigin
    ](
        codec: Int32,
        data: Span[UInt8, page_origin],
        uncompressed_size: Int,
        ref[scratch_origin] scratch: List[UInt8],
    ) raises -> Span[UInt8, origin_of(page_origin, scratch_origin)].Immutable:
        comptime O = origin_of(origin_of(page_origin, scratch_origin))
        if codec == CompressionCodec.UNCOMPRESSED.value:
            return page_span[O](data)
        if codec == CompressionCodec.SNAPPY.value:
            scratch = snappy_decompress(data)
            return page_span[O](Span(scratch))
        if codec == CompressionCodec.GZIP.value:
            scratch = gunzip(data)
            return page_span[O](Span(scratch))
        raise unsupported_codec(codec)

    @staticmethod
    def compress(codec: Int32, data: Span[UInt8, _]) raises -> List[UInt8]:
        if codec == CompressionCodec.UNCOMPRESSED.value:
            return copy_of(data)
        if codec == CompressionCodec.SNAPPY.value:
            return snappy_compress(data)
        if codec == CompressionCodec.GZIP.value:
            return gzip_wrap(data)
        raise unsupported_codec(codec)
