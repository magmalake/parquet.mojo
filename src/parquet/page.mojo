"""Page decoding: headers, compression, levels, and the values inside.

`read_column_chunk` walks every page of one column chunk — the optional
dictionary page and then the data pages, v1 or v2 — and returns the whole
chunk as three parallel things:

* `defs` — one definition level per *slot*, absent when the leaf's maximum
  definition level is 0;
* `reps` — one repetition level per slot, absent when the maximum repetition
  level is 0;
* `values` — the non-null values, back to back, still physical.

Page CRC32s are verified when the writer stored them. Every length in a page
header is checked against the bytes that are actually there, so a truncated or
doctored file raises instead of reading past the end of the buffer.
"""

from std.memory import bitcast

from hashes import crc32
from parquet.bitio import HybridDecoder, bit_width, unpack_msb
from parquet.codec import CodecSet
from parquet.encoding import (
    PK_BOOL,
    PK_FIXED,
    PK_VAR,
    PhysBuffer,
    decode_byte_stream_split,
    decode_delta_binary_packed,
    decode_delta_byte_array,
    decode_delta_length_byte_array,
    decode_dict_indices,
    decode_plain,
    decode_rle_bool,
    gather,
    physical_kind,
    physical_width,
)
from parquet.schema import LeafColumn
from thrift import (
    ColumnChunk,
    ColumnMetaData,
    Encoding,
    PageHeader,
    PageType,
    Type,
    read_page_header,
)


struct ColumnData(Copyable, Movable, Defaultable):
    """One decoded column chunk: levels and non-null values."""

    var defs: List[UInt16]
    var reps: List[UInt16]
    var values: PhysBuffer
    var num_slots: Int
    """Level slots — the number of values including nulls."""

    def __init__(out self):
        self.defs = List[UInt16]()
        self.reps = List[UInt16]()
        self.values = PhysBuffer()
        self.num_slots = 0

    def __init__(out self, *, copy: Self):
        self.defs = copy.defs.copy()
        self.reps = copy.reps.copy()
        self.values = copy.values.copy()
        self.num_slots = copy.num_slots

    def __init__(out self, *, deinit move: Self):
        self.defs = move.defs^
        self.reps = move.reps^
        self.values = move.values^
        self.num_slots = move.num_slots


def _read_levels(
    data: Span[UInt8, _],
    pos: Int,
    encoding: Int32,
    max_level: Int,
    count: Int,
    length_prefixed: Bool,
    explicit_length: Int,
    mut out: List[UInt16],
) raises -> Int:
    """Decode `count` levels; return the offset just past them."""
    if max_level == 0:
        for _ in range(count):
            out.append(0)
        return pos
    var width = bit_width(max_level)
    if encoding == Encoding.BIT_PACKED.value:
        # The Parquet 1.0 legacy level encoding: most-significant bit first,
        # no length prefix, exactly ceil(count * width / 8) bytes.
        var raw = List[UInt64]()
        var end = unpack_msb(data, pos, width, count, raw)
        for v in raw:
            out.append(UInt16(v))
        return end
    if encoding != Encoding.RLE.value:
        raise Error(
            String(
                "parquet.page: level encoding ",
                Encoding(encoding).name(),
                " is not RLE or BIT_PACKED",
            )
        )
    var start = pos
    var length = explicit_length
    if length_prefixed:
        if pos + 4 > len(data):
            raise Error("parquet.page: truncated level length prefix")
        length = (
            Int(data[pos])
            | (Int(data[pos + 1]) << 8)
            | (Int(data[pos + 2]) << 16)
            | (Int(data[pos + 3]) << 24)
        )
        start = pos + 4
    if length < 0 or start + length > len(data):
        raise Error(
            String(
                "parquet.page: level block of ",
                length,
                " byte(s) at ",
                start,
                " runs past the ",
                len(data),
                "-byte page",
            )
        )
    var dec = HybridDecoder(data[start : start + length], width)
    out.reserve(len(out) + count)
    for _ in range(count):
        var v = dec.next()
        if Int(v) > max_level:
            raise Error(
                String(
                    "parquet.page: level ",
                    v,
                    " exceeds the column maximum of ",
                    max_level,
                )
            )
        out.append(UInt16(v))
    return start + length


def _decode_values(
    encoding: Int32,
    leaf: LeafColumn,
    data: Span[UInt8, _],
    count: Int,
    dict: PhysBuffer,
    has_dict: Bool,
) raises -> PhysBuffer:
    var phys = leaf.physical
    var width = physical_width(phys, leaf.type_length)
    if (
        encoding == Encoding.RLE_DICTIONARY.value
        or encoding == Encoding.PLAIN_DICTIONARY.value
    ):
        if not has_dict:
            raise Error(
                String(
                    "parquet.page: column '",
                    leaf.dotted(),
                    "' uses a dictionary encoding but has no dictionary page",
                )
            )
        var indices = decode_dict_indices(data, count)
        return gather(dict, indices)
    if encoding == Encoding.PLAIN.value:
        return decode_plain(phys, leaf.type_length, data, count)
    if encoding == Encoding.RLE.value:
        if phys != Type.BOOLEAN.value:
            raise Error(
                "parquet.page: the RLE value encoding is only for booleans"
            )
        return decode_rle_bool(data, count)
    if encoding == Encoding.DELTA_BINARY_PACKED.value:
        if phys != Type.INT32.value and phys != Type.INT64.value:
            raise Error(
                String(
                    "parquet.page: DELTA_BINARY_PACKED on physical type ",
                    Type(phys).name(),
                )
            )
        return decode_delta_binary_packed(data, count, width)
    if encoding == Encoding.DELTA_LENGTH_BYTE_ARRAY.value:
        return decode_delta_length_byte_array(data, count)
    if encoding == Encoding.DELTA_BYTE_ARRAY.value:
        var v = decode_delta_byte_array(data, count)
        if phys == Type.FIXED_LEN_BYTE_ARRAY.value:
            # Parquet 2.x allows DELTA_BYTE_ARRAY for FLBA; flatten it back to
            # the fixed layout the rest of the reader expects.
            var out = PhysBuffer(PK_FIXED, width)
            out.bytes.reserve(v.count * width)
            for i in range(v.count):
                var s = v.value_span(i)
                if len(s) != width:
                    raise Error(
                        String(
                            "parquet.page: DELTA_BYTE_ARRAY value ",
                            i,
                            " is ",
                            len(s),
                            " bytes, expected ",
                            width,
                        )
                    )
                out.bytes.extend(s)
            out.count = v.count
            return out^
        return v^
    if encoding == Encoding.BYTE_STREAM_SPLIT.value:
        if width == 0:
            raise Error(
                "parquet.page: BYTE_STREAM_SPLIT needs a fixed-width type"
            )
        return decode_byte_stream_split(data, count, width)
    raise Error(
        String(
            "parquet.page: value encoding ",
            Encoding(encoding).name(),
            " is not implemented",
        )
    )


def chunk_start(cm: ColumnMetaData) -> Int:
    """Where a column chunk's pages begin — the dictionary page if it has one."""
    if cm.dictionary_page_offset:
        var d = Int(cm.dictionary_page_offset.value())
        if d > 0 and d < Int(cm.data_page_offset):
            return d
    return Int(cm.data_page_offset)


def read_column_chunk[Codecs: CodecSet](
    file: Span[UInt8, _],
    cm: ColumnMetaData,
    leaf: LeafColumn,
    verify_crc: Bool,
) raises -> ColumnData:
    """Decode every page of one column chunk."""
    var out = ColumnData()
    out.values = PhysBuffer(physical_kind(leaf.physical), physical_width(leaf.physical, leaf.type_length))
    var offset = chunk_start(cm)
    var limit = offset + Int(cm.total_compressed_size)
    if offset < 0 or limit > len(file):
        raise Error(
            String(
                "parquet.page: column chunk at ",
                offset,
                " + ",
                cm.total_compressed_size,
                " runs past the ",
                len(file),
                "-byte file",
            )
        )
    var want = Int(cm.num_values)
    var dict = PhysBuffer()
    var has_dict = False
    var codec = cm.codec.value

    while out.num_slots < want and offset < limit:
        var hdr = read_page_header(file, offset)
        ref ph = hdr[0]
        var body_at = offset + hdr[1]
        var csize = Int(ph.compressed_page_size)
        var usize = Int(ph.uncompressed_page_size)
        if csize < 0 or usize < 0 or body_at + csize > len(file):
            raise Error(
                String(
                    "parquet.page: page at ",
                    offset,
                    " declares ",
                    csize,
                    " compressed byte(s) past the end of the file",
                )
            )
        var body = file[body_at : body_at + csize]
        if verify_crc and ph.crc:
            var want_crc = ph.crc.value()
            var got = crc32(body)
            if got != UInt32(bitcast[DType.uint32](want_crc)):
                raise Error(
                    String(
                        "parquet.page: CRC32 mismatch on the page at ",
                        offset,
                        " — header says ",
                        want_crc,
                        ", data hashes to ",
                        got,
                    )
                )
        offset = body_at + csize

        if ph.type_ == PageType.DICTIONARY_PAGE:
            ref dh = ph.dictionary_page_header
            if not dh:
                raise Error("parquet.page: dictionary page with no header")
            var n = Int(dh.value().num_values)
            var raw = Codecs.decompress(codec, body, usize)
            var enc = dh.value().encoding.value
            if (
                enc != Encoding.PLAIN.value
                and enc != Encoding.PLAIN_DICTIONARY.value
            ):
                raise Error(
                    String(
                        "parquet.page: dictionary page encoding ",
                        Encoding(enc).name(),
                        " is not PLAIN",
                    )
                )
            dict = decode_plain(leaf.physical, leaf.type_length, Span(raw), n)
            has_dict = True
            continue

        if ph.type_ == PageType.INDEX_PAGE:
            continue

        if ph.type_ == PageType.DATA_PAGE:
            ref h = ph.data_page_header
            if not h:
                raise Error("parquet.page: v1 data page with no header")
            var n = Int(h.value().num_values)
            var raw = Codecs.decompress(codec, body, usize)
            var buf = Span(raw)
            var pos = 0
            var before = len(out.defs)
            if leaf.max_rep > 0:
                pos = _read_levels(
                    buf,
                    pos,
                    h.value().repetition_level_encoding.value,
                    leaf.max_rep,
                    n,
                    True,
                    0,
                    out.reps,
                )
            if leaf.max_def > 0:
                pos = _read_levels(
                    buf,
                    pos,
                    h.value().definition_level_encoding.value,
                    leaf.max_def,
                    n,
                    True,
                    0,
                    out.defs,
                )
            else:
                for _ in range(n):
                    out.defs.append(0)
            var non_null = n
            if leaf.max_def > 0:
                non_null = 0
                for i in range(before, len(out.defs)):
                    if Int(out.defs[i]) == leaf.max_def:
                        non_null += 1
            var vals = _decode_values(
                h.value().encoding.value, leaf, buf[pos:], non_null, dict, has_dict
            )
            out.values.extend(vals)
            out.num_slots += n
            continue

        if ph.type_ == PageType.DATA_PAGE_V2:
            ref h = ph.data_page_header_v2
            if not h:
                raise Error("parquet.page: v2 data page with no header")
            var n = Int(h.value().num_values)
            var nulls = Int(h.value().num_nulls)
            var rep_len = Int(h.value().repetition_levels_byte_length)
            var def_len = Int(h.value().definition_levels_byte_length)
            if rep_len < 0 or def_len < 0 or rep_len + def_len > csize:
                raise Error(
                    String(
                        "parquet.page: v2 level lengths ",
                        rep_len,
                        "+",
                        def_len,
                        " exceed the ",
                        csize,
                        "-byte page",
                    )
                )
            # Levels are never compressed in a v2 page.
            if leaf.max_rep > 0:
                _ = _read_levels(
                    body[0:rep_len],
                    0,
                    Encoding.RLE.value,
                    leaf.max_rep,
                    n,
                    False,
                    rep_len,
                    out.reps,
                )
            if leaf.max_def > 0:
                _ = _read_levels(
                    body[rep_len : rep_len + def_len],
                    0,
                    Encoding.RLE.value,
                    leaf.max_def,
                    n,
                    False,
                    def_len,
                    out.defs,
                )
            else:
                for _ in range(n):
                    out.defs.append(0)
            var vbytes = body[rep_len + def_len :]
            var compressed = h.value().is_compressed.or_else(True)
            var non_null = n - nulls
            if non_null < 0 or non_null > n:
                raise Error(
                    String(
                        "parquet.page: v2 page claims ",
                        nulls,
                        " nulls out of ",
                        n,
                        " values",
                    )
                )
            if compressed:
                var raw = Codecs.decompress(
                    codec, vbytes, usize - rep_len - def_len
                )
                var vals = _decode_values(
                    h.value().encoding.value,
                    leaf,
                    Span(raw),
                    non_null,
                    dict,
                    has_dict,
                )
                out.values.extend(vals)
            else:
                var vals = _decode_values(
                    h.value().encoding.value, leaf, vbytes, non_null, dict, has_dict
                )
                out.values.extend(vals)
            out.num_slots += n
            continue

        raise Error(
            String("parquet.page: unknown page type ", ph.type_.name())
        )

    if out.num_slots != want:
        raise Error(
            String(
                "parquet.page: column '",
                leaf.dotted(),
                "' decoded ",
                out.num_slots,
                " value slot(s) but the chunk metadata says ",
                want,
            )
        )
    if leaf.max_rep == 0:
        out.reps.clear()
    return out^
