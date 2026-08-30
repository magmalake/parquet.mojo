"""Bit-level readers: little-endian bit packing and the RLE/bit-packed hybrid.

Parquet packs small integers two ways, and both are here:

* **bit packing** — values of a fixed bit width laid down least-significant
  bit first, running across byte boundaries. This is the packing used inside
  a hybrid run and inside a `DELTA_BINARY_PACKED` miniblock. (It is *not* the
  `BIT_PACKED` legacy level encoding, which packs most-significant bit first;
  that one has its own reader below.)
* **the RLE/bit-packed hybrid** — a sequence of runs, each either a repeated
  value or a block of bit-packed values, introduced by a ULEB128 header.
  Definition levels, repetition levels, dictionary indices and `RLE`-encoded
  booleans are all this.

Everything is bounds checked: a truncated buffer, a run that claims more
values than remain, or a bit width above 64 raises rather than reading past
the end of the span.
"""


def bit_width(max_value: Int) -> Int:
    """The number of bits needed to hold `max_value` (0 for 0)."""
    var w = 0
    var v = max_value
    while v > 0:
        w += 1
        v >>= 1
    return w


struct BitReader[origin: ImmOrigin](Copyable, Movable):
    """Reads little-endian bit-packed values out of a borrowed span."""

    var data: Span[UInt8, Self.origin]
    var bit_pos: Int

    def __init__(out self, data: Span[UInt8, Self.origin]):
        self.data = data
        self.bit_pos = 0

    def __init__(out self, *, copy: Self):
        self.data = copy.data
        self.bit_pos = copy.bit_pos

    def __init__(out self, *, deinit move: Self):
        self.data = move.data
        self.bit_pos = move.bit_pos

    def byte_pos(self) -> Int:
        return (self.bit_pos + 7) // 8

    def align(mut self):
        self.bit_pos = self.byte_pos() * 8

    def get(mut self, width: Int) raises -> UInt64:
        """Read one value of `width` bits, least-significant bit first."""
        if width == 0:
            return 0
        if width > 64:
            raise Error(String("parquet.bitio: bit width ", width, " > 64"))
        if self.bit_pos + width > len(self.data) * 8:
            raise Error(
                String(
                    "parquet.bitio: truncated bit stream, wanted ",
                    width,
                    " bits at bit ",
                    self.bit_pos,
                    " of ",
                    len(self.data) * 8,
                )
            )
        var out: UInt64 = 0
        var got = 0
        while got < width:
            var byte = self.bit_pos // 8
            var off = self.bit_pos % 8
            var avail = 8 - off
            var take = width - got
            if take > avail:
                take = avail
            var mask = (UInt64(1) << UInt64(take)) - 1
            var chunk = (UInt64(self.data[byte]) >> UInt64(off)) & mask
            out |= chunk << UInt64(got)
            got += take
            self.bit_pos += take
        return out


def unpack_lsb(
    data: Span[UInt8, _], start_byte: Int, width: Int, count: Int, mut out: List[UInt64]
) raises -> Int:
    """Append `count` little-endian bit-packed values; return the end byte."""
    var need = (count * width + 7) // 8
    if start_byte < 0 or start_byte + need > len(data):
        raise Error(
            String(
                "parquet.bitio: bit-packed run of ",
                count,
                " x ",
                width,
                " bits needs ",
                need,
                " byte(s) at ",
                start_byte,
                " but only ",
                len(data) - start_byte,
                " remain",
            )
        )
    var br = BitReader(data)
    br.bit_pos = start_byte * 8
    out.reserve(len(out) + count)
    for _ in range(count):
        out.append(br.get(width))
    return start_byte + need


def unpack_msb(
    data: Span[UInt8, _], start_byte: Int, width: Int, count: Int, mut out: List[UInt64]
) raises -> Int:
    """The legacy `BIT_PACKED` level encoding — most-significant bit first."""
    var need = (count * width + 7) // 8
    if start_byte < 0 or start_byte + need > len(data):
        raise Error(
            String(
                "parquet.bitio: BIT_PACKED run of ",
                count,
                " x ",
                width,
                " bits needs ",
                need,
                " byte(s) at ",
                start_byte,
                " but only ",
                len(data) - start_byte,
                " remain",
            )
        )
    out.reserve(len(out) + count)
    var bit = 0
    for _ in range(count):
        var v: UInt64 = 0
        for _ in range(width):
            var byte = data[start_byte + bit // 8]
            var b = (UInt64(byte) >> UInt64(7 - bit % 8)) & 1
            v = (v << 1) | b
            bit += 1
        out.append(v)
    return start_byte + need


def read_uleb128(data: Span[UInt8, _], pos: Int) raises -> Tuple[UInt64, Int]:
    """Decode a ULEB128 at `pos`; return the value and the byte after it."""
    var result: UInt64 = 0
    var shift = 0
    var i = pos
    while True:
        if i >= len(data):
            raise Error(
                String("parquet.bitio: truncated varint at offset ", pos)
            )
        if shift > 63:
            raise Error(
                String("parquet.bitio: varint longer than 10 bytes at ", pos)
            )
        var b = data[i]
        result |= UInt64(b & 0x7F) << UInt64(shift)
        i += 1
        if b < 0x80:
            break
        shift += 7
    return (result, i)


def zigzag_decode(u: UInt64) -> Int64:
    """Undo zigzag: 0,1,2,3,4 -> 0,-1,1,-2,2."""
    var half = Int64(u >> 1)
    if (u & 1) == 1:
        return -half - 1
    return half


def read_zigzag(data: Span[UInt8, _], pos: Int) raises -> Tuple[Int64, Int]:
    """Decode a zigzag ULEB128 at `pos`; return the value and the next byte."""
    var raw = read_uleb128(data, pos)
    return (zigzag_decode(raw[0]), raw[1])


struct HybridDecoder[origin: ImmOrigin](Copyable, Movable):
    """The RLE / bit-packed hybrid used for levels, dictionary indices and
    `RLE` booleans.

    ```text
    run        := bit-packed-run | rle-run
    bit-packed := <ULEB128 (groups << 1 | 1)> <groups * 8 values, bit packed>
    rle-run    := <ULEB128 (count << 1)> <value, ceil(width/8) bytes LE>
    ```
    """

    var data: Span[UInt8, Self.origin]
    var pos: Int
    var width: Int
    # The current run: `remaining` values still to hand out. `is_packed`
    # selects between `buffer` (bit-packed) and `value` (repeated).
    var buffer: List[UInt64]
    var buf_pos: Int
    var value: UInt64
    var remaining: Int
    var is_packed: Bool

    def __init__(out self, data: Span[UInt8, Self.origin], width: Int) raises:
        if width < 0 or width > 64:
            raise Error(String("parquet.rle: bad bit width ", width))
        self.data = data
        self.pos = 0
        self.width = width
        self.buffer = List[UInt64]()
        self.buf_pos = 0
        self.value = 0
        self.remaining = 0
        self.is_packed = False

    def __init__(out self, *, copy: Self):
        self.data = copy.data
        self.pos = copy.pos
        self.width = copy.width
        self.buffer = copy.buffer.copy()
        self.buf_pos = copy.buf_pos
        self.value = copy.value
        self.remaining = copy.remaining
        self.is_packed = copy.is_packed

    def __init__(out self, *, deinit move: Self):
        self.data = move.data
        self.pos = move.pos
        self.width = move.width
        self.buffer = move.buffer^
        self.buf_pos = move.buf_pos
        self.value = move.value
        self.remaining = move.remaining
        self.is_packed = move.is_packed

    def _load_run(mut self) raises:
        var head = read_uleb128(self.data, self.pos)
        self.pos = head[1]
        var indicator = head[0]
        if (indicator & 1) == 1:
            var groups = Int(indicator >> 1)
            var count = groups * 8
            if count <= 0 or count > (1 << 31):
                raise Error(
                    String("parquet.rle: bit-packed run of ", count, " values")
                )
            self.buffer.clear()
            self.pos = unpack_lsb(self.data, self.pos, self.width, count, self.buffer)
            self.buf_pos = 0
            self.remaining = count
            self.is_packed = True
        else:
            var count = Int(indicator >> 1)
            if count <= 0:
                raise Error(String("parquet.rle: RLE run of ", count, " values"))
            var nbytes = (self.width + 7) // 8
            if self.pos + nbytes > len(self.data):
                raise Error(
                    String(
                        "parquet.rle: truncated RLE run value at ", self.pos
                    )
                )
            var v: UInt64 = 0
            for k in range(nbytes):
                v |= UInt64(self.data[self.pos + k]) << UInt64(8 * k)
            self.pos += nbytes
            self.value = v
            self.remaining = count
            self.is_packed = False

    def next(mut self) raises -> UInt64:
        while self.remaining == 0:
            if self.pos >= len(self.data):
                raise Error("parquet.rle: ran out of runs before all values")
            self._load_run()
        self.remaining -= 1
        if self.is_packed:
            var v = self.buffer[self.buf_pos]
            self.buf_pos += 1
            return v
        return self.value

    def take(mut self, count: Int, mut out: List[UInt64]) raises:
        """Append the next `count` values."""
        out.reserve(len(out) + count)
        for _ in range(count):
            out.append(self.next())
