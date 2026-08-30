"""Value encodings: PLAIN, dictionaries, the DELTA family, BYTE_STREAM_SPLIT.

Everything decodes into a `PhysBuffer` — values still in their *physical*
Parquet form (an `INT32` is four little-endian bytes whatever logical type
annotates it). `parquet.convert` is what turns a `PhysBuffer` plus definition
levels into an Arrow buffer.

Three physical shapes cover all eight Parquet types:

* `PK_BOOL` — a bit-packed bitmap, LSB first, one bit per value.
* `PK_FIXED` — `count * width` bytes; `width` is 4, 8 or 12 for the numeric
  types and `type_length` for `FIXED_LEN_BYTE_ARRAY`.
* `PK_VAR` — `count + 1` offsets into a data buffer, for `BYTE_ARRAY`.
"""

from std.memory import bitcast

from parquet.bitio import (
    HybridDecoder,
    bit_width,
    decode_hybrid_into,
    read_uleb128,
    read_zigzag,
    unpack_lsb,
    unpack_lsb_into,
    zigzag_decode,
)
from thrift import Type

comptime PK_BOOL = 0
comptime PK_FIXED = 1
comptime PK_VAR = 2


def physical_width(phys: Int32, type_length: Int) raises -> Int:
    """Bytes per value for a fixed-width physical type; 0 for the others."""
    if phys == Type.INT32.value or phys == Type.FLOAT.value:
        return 4
    if phys == Type.INT64.value or phys == Type.DOUBLE.value:
        return 8
    if phys == Type.INT96.value:
        return 12
    if phys == Type.FIXED_LEN_BYTE_ARRAY.value:
        if type_length <= 0:
            raise Error(
                String(
                    "parquet: FIXED_LEN_BYTE_ARRAY with type_length ",
                    type_length,
                )
            )
        return type_length
    return 0


def physical_kind(phys: Int32) -> Int:
    if phys == Type.BOOLEAN.value:
        return PK_BOOL
    if phys == Type.BYTE_ARRAY.value:
        return PK_VAR
    return PK_FIXED


struct PhysBuffer(Copyable, Defaultable, Movable):
    """Decoded values still in their physical Parquet representation."""

    var kind: Int
    var width: Int
    var count: Int
    var bytes: List[UInt8]
    var offsets: List[Int32]

    def __init__(out self):
        self.kind = PK_FIXED
        self.width = 0
        self.count = 0
        self.bytes = List[UInt8]()
        self.offsets = List[Int32]()

    def __init__(out self, kind: Int, width: Int):
        self.kind = kind
        self.width = width
        self.count = 0
        self.bytes = List[UInt8]()
        self.offsets = List[Int32]()
        if kind == PK_VAR:
            self.offsets.append(0)

    def __init__(out self, *, copy: Self):
        self.kind = copy.kind
        self.width = copy.width
        self.count = copy.count
        self.bytes = copy.bytes.copy()
        self.offsets = copy.offsets.copy()

    def __init__(out self, *, deinit move: Self):
        self.kind = move.kind
        self.width = move.width
        self.count = move.count
        self.bytes = move.bytes^
        self.offsets = move.offsets^

    def bool_at(self, i: Int) -> Bool:
        return ((self.bytes[i // 8] >> UInt8(i % 8)) & 1) == 1

    def append_bool(mut self, v: Bool):
        var byte = self.count // 8
        while len(self.bytes) <= byte:
            self.bytes.append(0)
        if v:
            self.bytes[byte] |= UInt8(1) << UInt8(self.count % 8)
        self.count += 1

    def append_bytes(mut self, src: Span[UInt8, _]):
        """Append one `PK_VAR` value."""
        self.bytes.extend(src)
        self.offsets.append(Int32(len(self.bytes)))
        self.count += 1

    def reserve_values(mut self, count: Int, bytes: Int):
        """Size the buffers for `count` more values and `bytes` more bytes, so
        that a chunk of many pages does not grow its buffer log-many times."""
        if self.kind == PK_VAR:
            self.offsets.reserve(len(self.offsets) + count)
        elif self.kind == PK_BOOL:
            self.bytes.reserve((self.count + count + 7) // 8)
        if bytes > 0:
            self.bytes.reserve(len(self.bytes) + bytes)

    def extend(mut self, other: PhysBuffer) raises:
        """Append every value of `other`, which must have the same shape."""
        if other.count == 0:
            return
        if self.count == 0 and self.kind == other.kind and len(self.bytes) == 0:
            self.width = other.width
        if self.kind != other.kind or (
            self.kind == PK_FIXED and self.width != other.width
        ):
            raise Error("parquet.encoding: cannot concatenate unlike pages")
        if self.kind == PK_BOOL:
            for i in range(other.count):
                self.append_bool(other.bool_at(i))
            return
        if self.kind == PK_VAR:
            var base = len(self.bytes)
            self.bytes.extend(Span(other.bytes))
            for i in range(1, len(other.offsets)):
                self.offsets.append(Int32(base) + other.offsets[i])
            self.count += other.count
            return
        self.bytes.extend(Span(other.bytes))
        self.count += other.count

    def value_span(self, i: Int) -> Span[UInt8, origin_of(self.bytes)]:
        """The bytes of value `i` — `PK_VAR` or `PK_FIXED` only."""
        if self.kind == PK_VAR:
            return Span(self.bytes)[
                Int(self.offsets[i]) : Int(self.offsets[i + 1])
            ]
        return Span(self.bytes)[i * self.width : (i + 1) * self.width]


def _need(data: Span[UInt8, _], want: Int, what: StringSlice) raises:
    if want > len(data):
        raise Error(
            String(
                "parquet.encoding: ",
                what,
                " needs ",
                want,
                " byte(s) but the page has ",
                len(data),
            )
        )


def decode_plain(
    phys: Int32, type_length: Int, data: Span[UInt8, _], count: Int
) raises -> PhysBuffer:
    """PLAIN: fixed-width values little-endian, byte arrays length-prefixed."""
    if count < 0:
        raise Error(String("parquet.encoding: negative value count ", count))
    var kind = physical_kind(phys)
    if kind == PK_BOOL:
        _need(data, (count + 7) // 8, "PLAIN booleans")
        var out = PhysBuffer(PK_BOOL, 0)
        out.bytes.extend(data[0 : (count + 7) // 8])
        out.count = count
        return out^
    if kind == PK_VAR:
        var out = PhysBuffer(PK_VAR, 0)
        if count == 0:
            return out^
        # One pass over the length prefixes fixes every offset and the total
        # size, so the bytes are copied once into a buffer that never grows.
        out.offsets.resize(count + 1, 0)
        var ooff = out.offsets.unsafe_ptr()
        var pos = 0
        var total = 0
        for i in range(count):
            _need(data, pos + 4, "PLAIN byte array length")
            var n = (
                Int(data[pos])
                | (Int(data[pos + 1]) << 8)
                | (Int(data[pos + 2]) << 16)
                | (Int(data[pos + 3]) << 24)
            )
            pos += 4
            if n < 0:
                raise Error(
                    String("parquet.encoding: negative byte array length ", n)
                )
            _need(data, pos + n, "PLAIN byte array body")
            pos += n
            total += n
            ooff.unsafe_store(i + 1, Int32(total))
        out.bytes.resize(total, 0)
        var dst = out.bytes.unsafe_ptr()
        var src = data.unsafe_ptr()
        pos = 0
        for i in range(count):
            var at = Int(ooff.unsafe_load(i))
            var n = Int(ooff.unsafe_load(i + 1)) - at
            pos += 4
            for b in range(n):
                dst.unsafe_store(at + b, src.unsafe_load(pos + b))
            pos += n
        out.count = count
        return out^
    var width = physical_width(phys, type_length)
    _need(data, count * width, "PLAIN fixed-width values")
    var out = PhysBuffer(PK_FIXED, width)
    out.bytes.extend(data[0 : count * width])
    out.count = count
    return out^


def gather(dict: PhysBuffer, indices: List[UInt32]) raises -> PhysBuffer:
    """Materialise dictionary-encoded values from a dictionary page.

    Fixed-width entries are a straight gather — one load and one store of the
    whole element per index. Byte arrays take two passes: lengths into the
    offsets, then the bytes.
    """
    var n = len(indices)
    var out = PhysBuffer(dict.kind, dict.width)
    if n == 0:
        return out^
    var idx = indices.unsafe_ptr()
    var entries = dict.count
    for i in range(n):
        if Int(idx.unsafe_load(i)) >= entries:
            raise Error(
                String(
                    "parquet.encoding: dictionary index ",
                    idx.unsafe_load(i),
                    " out of range (dictionary has ",
                    entries,
                    " entries)",
                )
            )

    if dict.kind == PK_BOOL:
        for i in range(n):
            out.append_bool(dict.bool_at(Int(idx.unsafe_load(i))))
        return out^

    if dict.kind == PK_VAR:
        var doff = dict.offsets.unsafe_ptr()
        out.offsets.resize(n + 1, 0)
        var ooff = out.offsets.unsafe_ptr()
        var total = 0
        for i in range(n):
            var k = Int(idx.unsafe_load(i))
            ooff.unsafe_store(i, Int32(total))
            total += Int(doff.unsafe_load(k + 1)) - Int(doff.unsafe_load(k))
        ooff.unsafe_store(n, Int32(total))
        out.bytes.resize(total, 0)
        var dst = out.bytes.unsafe_ptr()
        var src = dict.bytes.unsafe_ptr()
        for i in range(n):
            var k = Int(idx.unsafe_load(i))
            var at = Int(doff.unsafe_load(k))
            var end = Int(doff.unsafe_load(k + 1))
            var to = Int(ooff.unsafe_load(i))
            for b in range(end - at):
                dst.unsafe_store(to + b, src.unsafe_load(at + b))
        out.count = n
        return out^

    var w = dict.width
    out.bytes.resize(n * w, 0)
    var dst = out.bytes.unsafe_ptr()
    var src = dict.bytes.unsafe_ptr()
    if w == 8:
        var d8 = dst.bitcast[UInt64]()
        var s8 = src.bitcast[UInt64]()
        for i in range(n):
            d8.unsafe_store(
                i, s8.unsafe_load[alignment=1](Int(idx.unsafe_load(i)))
            )
    elif w == 4:
        var d4 = dst.bitcast[UInt32]()
        var s4 = src.bitcast[UInt32]()
        for i in range(n):
            d4.unsafe_store(
                i, s4.unsafe_load[alignment=1](Int(idx.unsafe_load(i)))
            )
    else:
        for i in range(n):
            var at = Int(idx.unsafe_load(i)) * w
            for b in range(w):
                dst.unsafe_store(i * w + b, src.unsafe_load(at + b))
    out.count = n
    return out^


def decode_dict_indices(
    data: Span[UInt8, _], count: Int
) raises -> List[UInt32]:
    """`RLE_DICTIONARY` / `PLAIN_DICTIONARY` data: a bit width byte, then a
    hybrid RLE run of that many indices."""
    if count == 0:
        return List[UInt32]()
    if len(data) == 0:
        raise Error("parquet.encoding: dictionary page data is empty")
    var width = Int(data[0])
    if width > 32:
        raise Error(
            String("parquet.encoding: dictionary index bit width ", width)
        )
    var out = List[UInt32](length=count, fill=0)
    decode_hybrid_into[DType.uint32](data[1:], width, count, out, 0)
    return out^


def decode_rle_bool(data: Span[UInt8, _], count: Int) raises -> PhysBuffer:
    """The `RLE` value encoding for booleans: a 4-byte length, then a hybrid
    run of 1-bit values."""
    _need(data, 4, "RLE boolean length")
    var n = (
        Int(data[0])
        | (Int(data[1]) << 8)
        | (Int(data[2]) << 16)
        | (Int(data[3]) << 24)
    )
    if n < 0 or 4 + n > len(data):
        raise Error(
            String(
                "parquet.encoding: RLE boolean length ",
                n,
                " runs past the page",
            )
        )
    var dec = HybridDecoder(data[4 : 4 + n], 1)
    var out = PhysBuffer(PK_BOOL, 0)
    for _ in range(count):
        out.append_bool(dec.next() == 1)
    return out^


def decode_delta_ints(
    data: Span[UInt8, _], count: Int, mut end: Int
) raises -> List[Int64]:
    """`DELTA_BINARY_PACKED`, the block / miniblock delta encoding.

    ```text
    header    := block_size_in_values, miniblocks_per_block, total_count, first_value
    block     := min_delta, list-of-bitwidths-of-miniblocks, miniblocks
    ```
    All four header fields are ULEB128, `first_value` and `min_delta` zigzag.
    Values wrap around at 64 bits, exactly as the spec says.
    """
    var pos = 0
    var h = read_uleb128(data, pos)
    var block_size = Int(h[0])
    pos = h[1]
    h = read_uleb128(data, pos)
    var miniblocks = Int(h[0])
    pos = h[1]
    h = read_uleb128(data, pos)
    var total = Int(h[0])
    pos = h[1]
    var z = read_zigzag(data, pos)
    var value = z[0]
    pos = z[1]

    if miniblocks <= 0 or block_size <= 0 or block_size % miniblocks != 0:
        raise Error(
            String(
                "parquet.delta: block size ",
                block_size,
                " is not a positive multiple of ",
                miniblocks,
                " miniblocks",
            )
        )
    if block_size % 128 != 0:
        raise Error(
            String(
                "parquet.delta: block size ",
                block_size,
                " is not a multiple of 128",
            )
        )
    var miniblock_size = block_size // miniblocks
    if total < 0:
        raise Error(String("parquet.delta: negative value count ", total))
    var want = count if count >= 0 else total
    if want > total:
        raise Error(
            String(
                "parquet.delta: page wants ",
                want,
                " values but the header declares ",
                total,
            )
        )

    var out = List[Int64](capacity=want)
    if total > 0:
        out.append(value)
    var produced = 1 if total > 0 else 0
    var widths = List[UInt64]()
    var raw = List[UInt64]()
    while produced < total:
        z = read_zigzag(data, pos)
        var min_delta = z[0]
        pos = z[1]
        _need(data, pos + miniblocks, "delta miniblock bit widths")
        widths.clear()
        for k in range(miniblocks):
            widths.append(UInt64(data[pos + k]))
        pos += miniblocks
        for m in range(miniblocks):
            if produced >= total:
                break
            var w = Int(widths[m])
            if w > 64:
                raise Error(String("parquet.delta: miniblock bit width ", w))
            raw.clear()
            pos = unpack_lsb(data, pos, w, miniblock_size, raw)
            for k in range(miniblock_size):
                if produced >= total:
                    break
                # 64-bit wrap-around is intentional and required by the spec.
                var acc = (
                    bitcast[DType.uint64](value)
                    + bitcast[DType.uint64](min_delta)
                    + raw[k]
                )
                value = bitcast[DType.int64](acc)
                if produced < want:
                    out.append(value)
                produced += 1
    end = pos
    return out^


def _store_ints(values: List[Int64], width: Int) -> PhysBuffer:
    var out = PhysBuffer(PK_FIXED, width)
    out.bytes.reserve(len(values) * width)
    for v in values:
        var u = bitcast[DType.uint64](v)
        for k in range(width):
            out.bytes.append(UInt8((u >> UInt64(8 * k)) & 0xFF))
    out.count = len(values)
    return out^


def decode_delta_binary_packed(
    data: Span[UInt8, _], count: Int, width: Int
) raises -> PhysBuffer:
    var end = 0
    var values = decode_delta_ints(data, count, end)
    return _store_ints(values, width)


def decode_delta_length_byte_array(
    data: Span[UInt8, _], count: Int
) raises -> PhysBuffer:
    """Lengths as `DELTA_BINARY_PACKED`, then the bytes back to back."""
    var end = 0
    var lengths = decode_delta_ints(data, count, end)
    var out = PhysBuffer(PK_VAR, 0)
    var pos = end
    for i in range(len(lengths)):
        var n = Int(lengths[i])
        if n < 0:
            raise Error(String("parquet.delta: negative string length ", n))
        _need(data, pos + n, "DELTA_LENGTH_BYTE_ARRAY body")
        out.append_bytes(data[pos : pos + n])
        pos += n
    return out^


def decode_delta_byte_array(
    data: Span[UInt8, _], count: Int
) raises -> PhysBuffer:
    """Prefix lengths, suffix lengths, then the suffix bytes: each value is
    the first `prefix` bytes of the value before it plus its own suffix."""
    var end1 = 0
    var prefixes = decode_delta_ints(data, count, end1)
    var end2 = 0
    var suffixes = decode_delta_ints(data[end1:], count, end2)
    if len(prefixes) != len(suffixes):
        raise Error(
            String(
                "parquet.delta: ",
                len(prefixes),
                " prefix lengths but ",
                len(suffixes),
                " suffix lengths",
            )
        )
    var out = PhysBuffer(PK_VAR, 0)
    var pos = end1 + end2
    var prev = List[UInt8]()
    for i in range(len(prefixes)):
        var p = Int(prefixes[i])
        var sfx = Int(suffixes[i])
        if p < 0 or sfx < 0 or p > len(prev):
            raise Error(
                String(
                    "parquet.delta: DELTA_BYTE_ARRAY value ",
                    i,
                    " has prefix ",
                    p,
                    " suffix ",
                    sfx,
                    " against a previous value of ",
                    len(prev),
                    " byte(s)",
                )
            )
        _need(data, pos + sfx, "DELTA_BYTE_ARRAY suffix")
        var cur = List[UInt8](capacity=p + sfx)
        for k in range(p):
            cur.append(prev[k])
        cur.extend(data[pos : pos + sfx])
        pos += sfx
        out.append_bytes(Span(cur))
        prev = cur^
    return out^


def decode_byte_stream_split(
    data: Span[UInt8, _], count: Int, width: Int
) raises -> PhysBuffer:
    """`BYTE_STREAM_SPLIT`: `width` streams of `count` bytes, the k-th stream
    holding byte k of every value."""
    if width <= 0:
        raise Error(String("parquet.encoding: BYTE_STREAM_SPLIT width ", width))
    _need(data, count * width, "BYTE_STREAM_SPLIT values")
    var out = PhysBuffer(PK_FIXED, width)
    out.bytes.resize(count * width, 0)
    for k in range(width):
        var base = k * count
        for i in range(count):
            out.bytes[i * width + k] = data[base + i]
    out.count = count
    return out^
