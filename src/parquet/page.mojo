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

from std.bit import pop_count
from std.memory import bitcast

from hashes import crc32
from parquet.bitio import (
    HybridDecoder,
    bit_width,
    decode_levels_into,
    levels_all_equal,
    read_uleb128,
    unpack_msb,
)
from parquet.alp import decode_alp
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
    decode_plain_into,
    decode_rle_bool,
    gather,
    gather_dict_into,
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


# ── packed validity ────────────────────────────────────────────────────────
#
# A leaf with `max_def == 1` and `max_rep == 0` — every plain nullable scalar
# column there is — has definition levels that are already a validity bitmap:
# one bit per slot, 1 where the value is there. arrow-rs decodes exactly that
# case straight into a `BooleanBufferBuilder` (`BufferInner::Mask`,
# `parquet/src/arrow/record_reader/definition_levels.rs:31`, gated by
# `packed_null_mask`, `record_reader/mod.rs:400`) with a width-1-only run
# decoder, `PackedDecoder` (`:313`), that never materialises a level: a
# repeated run is a bit fill and a literal run is a shifted copy of the
# encoded bytes, because at bit width 1 those bytes *are* the mask.
#
# The three helpers below are that decoder's write end. They assume the caller
# has already sized the buffer and that new bytes came in zeroed, so setting a
# bit is an `or` and clearing one is nothing at all.


@always_inline
def _mask_set_ones(mut mask: List[UInt8], at: Int, n: Int):
    """Set `n` bits from bit `at`, the whole-bytes middle by fill."""
    if n <= 0:
        return
    var mp = mask.unsafe_ptr()
    var first = at >> 3
    var last = (at + n - 1) >> 3
    var lo = at & 7
    var hi = (at + n - 1) & 7
    if first == last:
        var m = ((1 << (hi - lo + 1)) - 1) << lo
        mp.unsafe_store(first, mp.unsafe_load(first) | UInt8(m))
        return
    mp.unsafe_store(first, mp.unsafe_load(first) | UInt8((0xFF << lo) & 0xFF))
    for k in range(first + 1, last):
        mp.unsafe_store(k, 0xFF)
    mp.unsafe_store(last, mp.unsafe_load(last) | UInt8((1 << (hi + 1)) - 1))


@always_inline
def _mask_or_bits(
    mut mask: List[UInt8], at: Int, data: Span[UInt8, _], pos: Int, n: Int
) -> Int:
    """OR `n` LSB-first bits of `data[pos:]` into `mask` at bit `at`; return
    how many of them are set.

    The literal run of a width-1 hybrid stream, which needs no unpacking: the
    bytes are the mask, so this is a shifted copy plus a popcount for the null
    count that `_take_defs` has to return anyway.
    """
    var nbytes = (n + 7) >> 3
    var shift = at & 7
    var base = at >> 3
    var mp = mask.unsafe_ptr()
    var tail = UInt8((1 << (n & 7)) - 1) if (n & 7) != 0 else UInt8(0xFF)
    var ones = 0
    if shift == 0:
        for i in range(nbytes):
            var b = data[pos + i]
            if i == nbytes - 1:
                b &= tail
            ones += Int(pop_count(b))
            mp.unsafe_store(base + i, mp.unsafe_load(base + i) | b)
        return ones
    for i in range(nbytes):
        var b = data[pos + i]
        if i == nbytes - 1:
            b &= tail
        ones += Int(pop_count(b))
        mp.unsafe_store(
            base + i, mp.unsafe_load(base + i) | (b << UInt8(shift))
        )
        mp.unsafe_store(
            base + i + 1,
            mp.unsafe_load(base + i + 1) | (b >> UInt8(8 - shift)),
        )
    return ones


def copy_mask_bits(
    src: List[UInt8], at: Int, mut dst: List[UInt8], n: Int
) -> Int:
    """Copy `n` bits from bit `at` of `src` to bit 0 of `dst`, and return how
    many of them are set.

    `dst` is already sized to `(n + 7) // 8` and zeroed. A batch that starts on
    a byte boundary — every batch whose size is a multiple of 8, which is every
    default one — makes this a `memcpy`; otherwise it is one shift per byte.
    This is how a leaf's Arrow validity buffer gets built when the chunk's
    definition levels were decoded as a bitmap in the first place: no pass over
    levels, and the null count comes out of the same popcount.
    """
    if n <= 0:
        return 0
    var sp = src.unsafe_ptr()
    var dp = dst.unsafe_ptr()
    var nbytes = (n + 7) >> 3
    var shift = at & 7
    var base = at >> 3
    var tail = UInt8((1 << (n & 7)) - 1) if (n & 7) != 0 else UInt8(0xFF)
    var ones = 0
    if shift == 0:
        for i in range(nbytes):
            var b = sp.unsafe_load(base + i)
            if i == nbytes - 1:
                b &= tail
            ones += Int(pop_count(b))
            dp.unsafe_store(i, b)
        return ones
    var nsrc = len(src)
    for i in range(nbytes):
        var b = sp.unsafe_load(base + i) >> UInt8(shift)
        if base + i + 1 < nsrc:
            b |= sp.unsafe_load(base + i + 1) << UInt8(8 - shift)
        if i == nbytes - 1:
            b &= tail
        ones += Int(pop_count(b))
        dp.unsafe_store(i, b)
    return ones


@always_inline
def _mask_count(mask: List[UInt8], at: Int, n: Int) -> Int:
    """How many of the `n` bits from bit `at` are set."""
    if n <= 0:
        return 0
    var mp = mask.unsafe_ptr()
    var first = at >> 3
    var last = (at + n - 1) >> 3
    var lo = at & 7
    var hi = (at + n - 1) & 7
    if first == last:
        var m = ((1 << (hi - lo + 1)) - 1) << lo
        return Int(pop_count(mp.unsafe_load(first) & UInt8(m)))
    var ones = Int(
        pop_count(mp.unsafe_load(first) & UInt8((0xFF << lo) & 0xFF))
    )
    for k in range(first + 1, last):
        ones += Int(pop_count(mp.unsafe_load(k)))
    return ones + Int(
        pop_count(mp.unsafe_load(last) & UInt8((1 << (hi + 1)) - 1))
    )


struct ColumnData(Copyable, Defaultable, Movable):
    """One decoded column chunk: levels and non-null values."""

    var defs: List[UInt16]
    var reps: List[UInt16]
    var values: PhysBuffer
    var num_slots: Int
    """Level slots — the number of values including nulls."""
    var all_present: Bool
    """Every slot is at the leaf's maximum definition level, so `defs` was
    never materialised. This is the usual case — a column with no nulls — and
    skipping the per-slot level array is worth a great deal on a wide read."""
    var page_slot: List[Int]
    """The first slot of each data page from the first one with a null in it,
    and one final entry equal to `num_slots`. Empty until then, and so empty
    altogether for a chunk with no nulls.

    A checkpoint table, sampled at the one boundary the page walk already
    knows: it turns "where do this row's values start" into a binary search
    plus a scan bounded by one page, instead of an entry per row. Every slot
    before `page_slot[0]` holds a value, so its value index *is* its slot
    index and there is nothing to store — which is why a chunk that never
    sees a null allocates neither list."""
    var page_value: List[Int]
    """The first value of each data page `page_slot` covers, and one final
    entry equal to the number of values in `values`."""
    var packed: Bool
    """This leaf's definition levels *are* a validity bitmap — `max_def == 1`
    and `max_rep == 0`, so a level is one bit — and `mask`, not `defs`, is
    where they land. Set by `read_column_chunk` from the leaf's descriptor
    before a single page is read, so it is a property of the column and not of
    what any one page happened to contain."""
    var mask: List[UInt8]
    """Packed validity when `packed`: bit `i`, LSB-first, is 1 when slot `i`
    is at `max_def`. Empty when the chunk turned out to have no nulls at all,
    which is what `all_present` then says."""

    def __init__(out self):
        self.defs = List[UInt16]()
        self.reps = List[UInt16]()
        self.values = PhysBuffer()
        self.num_slots = 0
        self.all_present = True
        self.page_slot = List[Int]()
        self.page_value = List[Int]()
        self.packed = False
        self.mask = List[UInt8]()

    def __init__(out self, *, copy: Self):
        self.defs = copy.defs.copy()
        self.reps = copy.reps.copy()
        self.values = copy.values.copy()
        self.num_slots = copy.num_slots
        self.all_present = copy.all_present
        self.page_slot = copy.page_slot.copy()
        self.page_value = copy.page_value.copy()
        self.packed = copy.packed
        self.mask = copy.mask.copy()

    def __init__(out self, *, deinit move: Self):
        self.defs = move.defs^
        self.reps = move.reps^
        self.values = move.values^
        self.num_slots = move.num_slots
        self.all_present = move.all_present
        self.page_slot = move.page_slot^
        self.page_value = move.page_value^
        self.packed = move.packed
        self.mask = move.mask^

    @always_inline
    def masked(self) -> Bool:
        """Validity for this chunk is a bitmap, not an array of levels."""
        return self.packed and len(self.mask) > 0

    @always_inline
    def value_at(self, row: Int, max_def: Int) -> Int:
        """How many of the first `row` slots hold a value.

        The answer for a slot that is a page boundary is stored; for anything
        else it is that answer plus a scan of at most one page — a popcount
        over the validity mask where there is one. Nothing here is per row,
        which is the point: the caller wants this at a handful of batch
        boundaries, not at every row.
        """
        if row <= 0:
            return 0
        # Before the first checkpoint every slot holds a value, so the value
        # index and the slot index are the same number.
        if len(self.page_slot) == 0 or row <= self.page_slot[0]:
            return row
        var lo = 0
        var hi = len(self.page_slot) - 1
        while lo < hi:
            var mid = (lo + hi + 1) >> 1
            if self.page_slot[mid] <= row:
                lo = mid
            else:
                hi = mid - 1
        var start = self.page_slot[lo]
        var v = self.page_value[lo]
        if self.masked():
            return v + _mask_count(self.mask, start, row - start)
        var defs = self.defs.unsafe_ptr()
        var md = UInt16(max_def)
        for k in range(start, row):
            if defs.unsafe_load(k) == md:
                v += 1
        return v

    @always_inline
    def def_at(self, i: Int, max_def: Int) -> Int:
        """The definition level of slot `i`."""
        if self.all_present:
            return max_def
        if self.packed:
            var b = (self.mask[i >> 3] >> UInt8(i & 7)) & 1
            return max_def if b == 1 else 0
        return Int(self.defs[i])

    @always_inline
    def rep_at(self, i: Int) -> Int:
        return Int(self.reps[i]) if len(self.reps) else 0


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
    _ = decode_levels_into(
        data[start : start + length], width, count, max_level, out
    )
    return start + length


def _mask_append_ones(mut out: ColumnData, count: Int):
    """Append `count` present slots to the mask a chunk has already started."""
    var at = out.num_slots
    out.mask.resize((at + count + 7) // 8 + 1, 0)
    _mask_set_ones(out.mask, at, count)


def _materialise_mask(mut out: ColumnData):
    """Back-fill the validity of the slots decoded so far as all present.

    The mirror of `_materialise_defs`: pages that are entirely at `max_def`
    write nothing, so the first page with a null in it pays for the slots that
    were skipped — one `_mask_set_ones` over a byte range, not a level each.
    """
    if len(out.mask) == 0 and out.num_slots > 0:
        out.mask.resize((out.num_slots + 7) // 8 + 1, 0)
        _mask_set_ones(out.mask, 0, out.num_slots)


def _unpack_mask_to_defs(mut out: ColumnData, max_def: Int):
    """Expand a packed mask back into one definition level per slot.

    Only the Parquet 1.0 `BIT_PACKED` level encoding needs this: it is not a
    hybrid stream, so it cannot be read as a mask, and a chunk that mixes it
    with RLE pages has to settle on one representation. Off the hot path in
    every sense — no writer in the last decade emits it.
    """
    out.packed = False
    if len(out.mask) == 0:
        return
    out.defs.resize(out.num_slots, 0)
    var dp = out.defs.unsafe_ptr()
    var mp = out.mask.unsafe_ptr()
    for i in range(out.num_slots):
        if ((mp.unsafe_load(i >> 3) >> UInt8(i & 7)) & 1) == 1:
            dp.unsafe_store(i, UInt16(max_def))
    out.mask.clear()


def _decode_mask_into(
    data: Span[UInt8, _], count: Int, mut out: ColumnData
) raises -> Int:
    """Decode `count` width-1 hybrid levels straight into `out.mask`.

    arrow-rs's `PackedDecoder::read` (`definition_levels.rs:427`) without the
    intermediate: a repeated run becomes one `_mask_set_ones`, and a literal
    run is copied wholesale because at bit width 1 the encoded bytes already
    *are* the bitmask. Returns how many of the `count` slots are present,
    which the caller needs and a popcount gives for free.
    """
    var at = out.num_slots
    out.mask.resize((at + count + 7) // 8 + 1, 0)
    var pos = 0
    var done = 0
    var nn = 0
    while done < count:
        if pos >= len(data):
            raise Error("parquet.rle: ran out of runs before all values")
        var head = read_uleb128(data, pos)
        pos = head[1]
        var indicator = head[0]
        if (indicator & 1) == 1:
            var n = Int(indicator >> 1) * 8
            if n <= 0:
                raise Error(String("parquet.rle: bit-packed run of ", n))
            if n > count - done:
                n = count - done
            var nbytes = (n + 7) // 8
            if pos + nbytes > len(data):
                raise Error(
                    String("parquet.rle: truncated bit-packed run at ", pos)
                )
            nn += _mask_or_bits(out.mask, at + done, data, pos, n)
            pos += nbytes
            done += n
        else:
            var n = Int(indicator >> 1)
            if n <= 0:
                raise Error(String("parquet.rle: RLE run of ", n, " values"))
            if pos >= len(data):
                raise Error(
                    String("parquet.rle: truncated RLE run value at ", pos)
                )
            var v = data[pos]
            pos += 1
            if n > count - done:
                n = count - done
            if v > 1:
                raise Error(
                    String(
                        "parquet.page: level ",
                        v,
                        " exceeds the column maximum of 1",
                    )
                )
            if v == 1:
                _mask_set_ones(out.mask, at + done, n)
                nn += n
            done += n
    return nn


def _materialise_defs(mut out: ColumnData, max_def: Int):
    """Back-fill the definition levels of the slots decoded so far.

    Pages whose levels are all `max_def` do not write anything into
    `out.defs`; the first page that actually has a null has to pay for the
    slots that were skipped.
    """
    if len(out.defs) == 0 and out.num_slots > 0:
        out.defs.resize(out.num_slots, UInt16(max_def))


def _take_defs(
    data: Span[UInt8, _],
    pos: Int,
    encoding: Int32,
    max_def: Int,
    count: Int,
    length_prefixed: Bool,
    explicit_length: Int,
    mut out: ColumnData,
) raises -> Tuple[Int, Int]:
    """Read one page's definition levels; return the offset past them and the
    number of non-null values.

    A page whose levels are all `max_def` — the overwhelmingly common case —
    is recognised from its run headers alone and never materialised. When the
    chunk is `packed`, what the rest of them land in is a validity bitmap
    rather than a `UInt16` per slot.
    """
    var width = bit_width(max_def)
    if encoding == Encoding.BIT_PACKED.value:
        # The one encoding a bitmap cannot absorb, so the chunk gives up on
        # packing and converts what it has.
        if out.packed:
            _unpack_mask_to_defs(out, max_def)
        _materialise_defs(out, max_def)
        var before = len(out.defs)
        var end = _read_levels(
            data,
            pos,
            encoding,
            max_def,
            count,
            length_prefixed,
            explicit_length,
            out.defs,
        )
        _ = width
        var nn = 0
        for i in range(before, len(out.defs)):
            if Int(out.defs[i]) > max_def:
                raise Error(
                    String(
                        "parquet.page: level ",
                        out.defs[i],
                        " exceeds the column maximum of ",
                        max_def,
                    )
                )
            if Int(out.defs[i]) == max_def:
                nn += 1
        return (end, nn)
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
    var block = data[start : start + length]
    if levels_all_equal(block, width, count, max_def):
        if out.packed:
            if len(out.mask) > 0:
                _mask_append_ones(out, count)
        elif len(out.defs) > 0:
            out.defs.resize(len(out.defs) + count, UInt16(max_def))
        return (start + length, count)
    if out.packed:
        _materialise_mask(out)
        var got = _decode_mask_into(block, count, out)
        return (start + length, got)
    _materialise_defs(out, max_def)
    var nn = decode_levels_into(block, width, count, max_def, out.defs)
    return (start + length, nn)


def _decode_values_into(
    mut out: PhysBuffer,
    encoding: Int32,
    leaf: LeafColumn,
    data: Span[UInt8, _],
    count: Int,
    dict: PhysBuffer,
    has_dict: Bool,
) raises:
    """Decode one page's values onto the end of the chunk's value buffer.

    PLAIN and the two dictionary encodings — between them, almost every page
    ever written — go straight in. The rest decode into their own buffer and
    are concatenated, which is what the whole chunk used to do.
    """
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
        gather_dict_into(out, dict, data, count)
        return
    if encoding == Encoding.PLAIN.value:
        decode_plain_into(out, leaf.physical, leaf.type_length, data, count)
        return
    var vals = _decode_values(encoding, leaf, data, count, dict, has_dict)
    out.extend(vals)


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
    if encoding == Encoding.ALP.value:
        if phys == Type.DOUBLE.value:
            return decode_alp[DType.float64](data, count)
        if phys == Type.FLOAT.value:
            return decode_alp[DType.float32](data, count)
        raise Error(
            String(
                "parquet.page: ALP on physical type ",
                Type(phys).name(),
                ", which is not FLOAT or DOUBLE",
            )
        )
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
    """Where a column chunk's pages begin — the dictionary page if it has one.
    """
    if cm.dictionary_page_offset:
        var d = Int(cm.dictionary_page_offset.value())
        if d > 0 and d < Int(cm.data_page_offset):
            return d
    return Int(cm.data_page_offset)


def read_column_chunk[
    Codecs: CodecSet
](
    file: Span[UInt8, _],
    cm: ColumnMetaData,
    leaf: LeafColumn,
    verify_crc: Bool,
) raises -> ColumnData:
    """Decode every page of one column chunk."""
    var out = ColumnData()
    # One optional level and no repetition: a definition level is a bit, so
    # the levels of this chunk are its validity bitmap and go straight there.
    out.packed = leaf.max_def == 1 and leaf.max_rep == 0
    out.values = PhysBuffer(
        physical_kind(leaf.physical),
        physical_width(leaf.physical, leaf.type_length),
    )
    # Size the chunk's value buffer once from the metadata, so a chunk of many
    # pages does not grow and copy it log-many times. The bounds keep a
    # doctored `num_values` from asking for an absurd allocation.
    var nv = Int(cm.num_values)
    if nv > 0 and nv < (1 << 30):
        if out.values.kind == PK_FIXED and out.values.width > 0:
            if nv * out.values.width <= (1 << 28):
                out.values.bytes.reserve(nv * out.values.width)
        elif out.values.kind == PK_VAR and nv <= (1 << 26):
            out.values.offsets.reserve(nv + 1)
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
    # Values written into `out.values` so far — the running count that
    # `page_value` samples once per page.
    var nvalues = 0

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
            var non_null = n
            if leaf.max_def > 0:
                var got = _take_defs(
                    buf,
                    pos,
                    h.value().definition_level_encoding.value,
                    leaf.max_def,
                    n,
                    True,
                    0,
                    out,
                )
                pos = got[0]
                non_null = got[1]
            _decode_values_into(
                out.values,
                h.value().encoding.value,
                leaf,
                buf[pos:],
                non_null,
                dict,
                has_dict,
            )
            if len(out.page_slot) > 0 or non_null < n:
                out.page_slot.append(out.num_slots)
                out.page_value.append(nvalues)
            nvalues += non_null
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
                _ = _take_defs(
                    body[rep_len : rep_len + def_len],
                    0,
                    Encoding.RLE.value,
                    leaf.max_def,
                    n,
                    False,
                    def_len,
                    out,
                )
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
            # A v2 page whose values are all null carries no value bytes at
            # all. Zero bytes is not a valid stream for any codec — Snappy
            # reads it as a truncated varint — so there is nothing to
            # decompress, and the empty span is already the answer.
            if compressed and len(vbytes) > 0:
                var raw = Codecs.decompress(
                    codec, vbytes, usize - rep_len - def_len
                )
                _decode_values_into(
                    out.values,
                    h.value().encoding.value,
                    leaf,
                    Span(raw),
                    non_null,
                    dict,
                    has_dict,
                )
            else:
                _decode_values_into(
                    out.values,
                    h.value().encoding.value,
                    leaf,
                    vbytes,
                    non_null,
                    dict,
                    has_dict,
                )
            if len(out.page_slot) > 0 or non_null < n:
                out.page_slot.append(out.num_slots)
                out.page_value.append(nvalues)
            nvalues += non_null
            out.num_slots += n
            continue

        raise Error(String("parquet.page: unknown page type ", ph.type_.name()))

    # The closing entry, so a lookup at the end of the chunk lands on a stored
    # answer rather than off the end of the table.
    if len(out.page_slot) > 0:
        out.page_slot.append(out.num_slots)
        out.page_value.append(nvalues)
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
    elif len(out.reps) > 0 and out.reps[0] != 0:
        # A repetition level says which level of nesting is being continued, so
        # the first slot of a chunk cannot continue anything — it has to start
        # a record, at level 0. Anything else means the levels are misaligned,
        # and assembly downstream would silently attach values to a record that
        # does not exist.
        raise Error(
            String(
                "parquet.page: column '",
                leaf.dotted(),
                "' starts with repetition level ",
                out.reps[0],
                ", so the chunk does not begin a record",
            )
        )
    # The mask is built a byte at a time with a spare byte of slack, so trim
    # it to exactly the slots the chunk has.
    if len(out.mask) > 0:
        out.mask.resize((out.num_slots + 7) // 8, 0)
    out.all_present = len(out.defs) == 0 and len(out.mask) == 0
    return out^
