"""Real dictionary pages, pulled out of the fixture corpus.

The parity gate and the benchmark both need what the reader actually sees: a
dictionary page decoded by `decode_plain`, and the `RLE_DICTIONARY` index
stream of the data pages that reference it. This walks a column chunk the way
`parquet.page.read_column_chunk` does and stops at the values, handing back the
dictionary and one `List[UInt32]` per data page.

Nothing here is on the timed path — the benchmark loads its pages in setup.
"""

from parquet.bitio import bit_width, decode_levels_into
from parquet.codec import CodecSet
from parquet.encoding import (
    PK_BOOL,
    PK_VAR,
    PhysBuffer,
    decode_dict_indices,
    decode_plain,
)
from parquet.schema import LeafColumn, ParquetSchema
from thrift import (
    ColumnMetaData,
    Encoding,
    FileMetaData,
    PageType,
    read_page_header,
)


struct DictChunk(Movable):
    """One column chunk's dictionary and the index stream of each data page."""

    var dict: PhysBuffer
    var pages: List[List[UInt32]]
    var counts: List[Int]
    """How many non-null values each page holds."""
    var raw: List[List[UInt8]]
    """Each page's `RLE_DICTIONARY` index stream, exactly as `gather_dict_into`
    wants it — a bit-width byte and then the hybrid runs. The benchmark needs
    it to time the fused production path, which never materialises `pages`."""
    var path: String
    """The column's dotted path, for the gate's output."""

    def __init__(out self):
        """An empty chunk."""
        self.dict = PhysBuffer()
        self.pages = List[List[UInt32]]()
        self.raw = List[List[UInt8]]()
        self.counts = List[Int]()
        self.path = String()

    def __init__(out self, *, deinit move: Self):
        """Move.

        Args:
            move: The value being moved from.
        """
        self.dict = move.dict^
        self.pages = move.pages^
        self.raw = move.raw^
        self.counts = move.counts^
        self.path = move.path^

    def total_values(self) -> Int:
        """How many values every data page adds up to.

        Returns:
            The count.
        """
        var n = 0
        for i in range(len(self.pages)):
            n += len(self.pages[i])
        return n

    def all_indices(self) raises -> List[UInt32]:
        """Every page's indices, back to back — the whole column chunk.

        Returns:
            One list of indices for the chunk.
        """
        var out = List[UInt32]()
        out.reserve(self.total_values())
        for i in range(len(self.pages)):
            out.extend(Span(self.pages[i]))
        return out^


def _skip_defs(
    data: Span[UInt8, _], encoding: Int32, max_def: Int, count: Int
) raises -> Tuple[Int, Int]:
    """Walk a v1 page's definition levels; return where the values start and
    how many of them are non-null.

    Args:
        data: The whole decompressed page.
        encoding: The definition-level encoding from the page header.
        max_def: The leaf's maximum definition level.
        count: How many levels the page holds.

    Returns:
        The byte offset of the value stream, and the non-null value count.

    Raises:
        If the level block is malformed or uses an encoding this does not read.
    """
    if max_def == 0:
        return (0, count)
    if encoding != Encoding.RLE.value:
        raise Error(
            "experiments.gpu_gather: only RLE definition levels are read here"
        )
    if len(data) < 4:
        raise Error("experiments.gpu_gather: truncated level length prefix")
    var length = (
        Int(data[0])
        | (Int(data[1]) << 8)
        | (Int(data[2]) << 16)
        | (Int(data[3]) << 24)
    )
    if length < 0 or 4 + length > len(data):
        raise Error("experiments.gpu_gather: level block runs past the page")
    var levels = List[UInt16]()
    _ = decode_levels_into(
        data[4 : 4 + length], bit_width(max_def), count, max_def, levels
    )
    var non_null = 0
    for i in range(len(levels)):
        if Int(levels[i]) == max_def:
            non_null += 1
    return (4 + length, non_null)


def read_dict_chunk[
    Codecs: CodecSet
](
    file: Span[UInt8, _],
    cm: ColumnMetaData,
    leaf: LeafColumn,
) raises -> DictChunk:
    """Walk one column chunk and collect its dictionary and index streams.

    Returns an empty `DictChunk` when the chunk has no dictionary page, or when
    a data page turns out not to be dictionary encoded (a writer is free to
    fall back to PLAIN mid-chunk, and `big.parquet` does).

    Parameters:
        Codecs: The codec set to decompress pages with.

    Args:
        file: The whole file.
        cm: The chunk's metadata.
        leaf: The column the chunk belongs to.

    Returns:
        The dictionary and one index list per dictionary-encoded data page.

    Raises:
        If a page header or body is malformed.
    """
    var out = DictChunk()
    out.path = leaf.dotted()
    var offset = Int(cm.data_page_offset)
    if cm.dictionary_page_offset:
        var d = Int(cm.dictionary_page_offset.value())
        if d > 0 and d < Int(cm.data_page_offset):
            offset = d
    var limit = offset + Int(cm.total_compressed_size)
    if offset < 0 or limit > len(file):
        raise Error("experiments.gpu_gather: column chunk runs past the file")
    var codec = cm.codec.value
    var scratch = List[UInt8]()
    var seen = 0
    var want = Int(cm.num_values)
    var has_dict = False

    while seen < want and offset < limit:
        var hdr = read_page_header(file, offset)
        ref ph = hdr[0]
        var body_at = offset + hdr[1]
        var csize = Int(ph.compressed_page_size)
        var usize = Int(ph.uncompressed_page_size)
        if csize < 0 or usize < 0 or body_at + csize > len(file):
            raise Error("experiments.gpu_gather: page runs past the file")
        var body = file[body_at : body_at + csize]
        offset = body_at + csize

        if ph.type_ == PageType.DICTIONARY_PAGE:
            ref dh = ph.dictionary_page_header
            if not dh:
                raise Error(
                    "experiments.gpu_gather: dictionary page with no header"
                )
            var raw = Codecs.decompress(codec, body, usize, scratch)
            out.dict = decode_plain(
                leaf.physical, leaf.type_length, raw, Int(dh.value().num_values)
            )
            has_dict = True
            continue

        if ph.type_ == PageType.INDEX_PAGE:
            continue

        # Only the simple shape: v1 data pages of a non-repeated leaf. A
        # nullable leaf is fine — the definition levels are read here purely to
        # find where the value stream starts and how many non-null values it
        # holds, which is exactly what `read_column_chunk` hands `gather_into`.
        # Anything else (v2 pages, repetition) is skipped rather than
        # half-decoded; the gate has plenty of coverage without it.
        if ph.type_ != PageType.DATA_PAGE:
            return DictChunk()
        ref h = ph.data_page_header
        if not h:
            raise Error("experiments.gpu_gather: v1 data page with no header")
        var n = Int(h.value().num_values)
        seen += n
        if not has_dict:
            return DictChunk()
        var enc = h.value().encoding.value
        if (
            enc != Encoding.RLE_DICTIONARY.value
            and enc != Encoding.PLAIN_DICTIONARY.value
        ):
            return DictChunk()
        if leaf.max_rep > 0:
            return DictChunk()
        var raw = Codecs.decompress(codec, body, usize, scratch)
        var got = _skip_defs(
            raw,
            h.value().definition_level_encoding.value,
            leaf.max_def,
            n,
        )
        var stream = raw[got[0] :]
        out.pages.append(decode_dict_indices(stream, got[1]))
        var kept = List[UInt8]()
        kept.extend(stream)
        out.raw.append(kept^)
        out.counts.append(got[1])

    if not has_dict or len(out.pages) == 0:
        return DictChunk()
    if out.dict.kind == PK_BOOL:
        return DictChunk()
    return out^


def find_dict_chunks[
    Codecs: CodecSet
](
    file: Span[UInt8, _],
    meta: FileMetaData,
    schema: ParquetSchema,
    var_only: Bool,
    fixed_only: Bool,
) raises -> List[DictChunk]:
    """Every dictionary-encoded column chunk in a file, in visit order.

    Parameters:
        Codecs: The codec set to decompress pages with.

    Args:
        file: The whole file.
        meta: The file's footer.
        schema: The file's schema.
        var_only: Keep only `BYTE_ARRAY` dictionaries.
        fixed_only: Keep only fixed-width dictionaries.

    Returns:
        The chunks that have a usable dictionary.

    Raises:
        If a page header or body is malformed.
    """
    var out = List[DictChunk]()
    for g in range(len(meta.row_groups)):
        ref rg = meta.row_groups[g]
        for c in range(len(rg.columns)):
            ref cc = rg.columns[c]
            if not cc.meta_data:
                continue
            if c >= len(schema.leaves):
                continue
            var chunk = read_dict_chunk[Codecs](
                file, cc.meta_data.value(), schema.leaves[c]
            )
            if len(chunk.pages) == 0:
                continue
            if var_only and chunk.dict.kind != PK_VAR:
                continue
            if fixed_only and chunk.dict.kind == PK_VAR:
                continue
            out.append(chunk^)
    return out^
