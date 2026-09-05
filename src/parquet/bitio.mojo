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
    data: Span[UInt8, _],
    start_byte: Int,
    width: Int,
    count: Int,
    mut out: List[UInt64],
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


@always_inline
def _u64_at(data: Span[UInt8, _], byte: Int) -> UInt64:
    """The 8 little-endian bytes at `byte`. The caller must have checked that
    8 bytes are readable there."""
    return (
        data.unsafe_ptr()
        .unsafe_offset(byte)
        .unsafe_bitcast[UInt64]()
        .unsafe_load[alignment=1](0)
    )


@always_inline
def _bits_slow(data: Span[UInt8, _], bitpos: Int, width: Int) -> UInt64:
    """`width` bits at `bitpos`, byte at a time — the tail of a run, where a
    64-bit window would read past the buffer."""
    var out: UInt64 = 0
    var got = 0
    var at = bitpos
    while got < width:
        var byte = at >> 3
        var off = at & 7
        var avail = 8 - off
        var take = width - got
        if take > avail:
            take = avail
        var mask = (UInt64(1) << UInt64(take)) - 1
        out |= ((UInt64(data[byte]) >> UInt64(off)) & mask) << UInt64(got)
        got += take
        at += take
    return out


def unpack_lsb_into[
    dt: DType
](
    data: Span[UInt8, _],
    start_byte: Int,
    width: Int,
    count: Int,
    mut out: List[Scalar[dt]],
    at: Int,
) raises -> Int:
    """Unpack `count` little-endian bit-packed values of `width` bits into
    `out[at : at + count]`, which the caller has already sized. Returns the
    byte just past the run.

    The body of the run is read one *unaligned 64-bit window* per value —
    a load, a shift and a mask — which is what makes bit-packed levels and
    dictionary indices cheap. Only the last few values, whose window would
    reach past the end of the buffer, take the byte-at-a-time path.
    """
    var need = (count * width + 7) // 8
    if width < 0 or width > 57:
        raise Error(String("parquet.bitio: bit width ", width, " unsupported"))
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
    if at < 0 or at + count > len(out):
        raise Error("parquet.bitio: unpack destination is too small")
    if count == 0:
        return start_byte + need
    var dst = out.unsafe_ptr()
    if width == 0:
        for i in range(count):
            dst.unsafe_store(at + i, Scalar[dt](0))
        return start_byte + need
    var mask = (UInt64(1) << UInt64(width)) - 1
    var bitpos = start_byte * 8
    # How many values keep their whole 64-bit window inside the buffer.
    var fast = 0
    var last_bit = (len(data) - 8) * 8
    if last_bit >= bitpos:
        fast = (last_bit - bitpos) // width + 1
        if fast > count:
            fast = count
    for i in range(fast):
        var v = (_u64_at(data, bitpos >> 3) >> UInt64(bitpos & 7)) & mask
        dst.unsafe_store(at + i, v.cast[dt]())
        bitpos += width
    for i in range(fast, count):
        dst.unsafe_store(at + i, _bits_slow(data, bitpos, width).cast[dt]())
        bitpos += width
    return start_byte + need


def levels_all_equal(
    data: Span[UInt8, _], width: Int, count: Int, value: Int
) raises -> Bool:
    """Is every one of the next `count` hybrid-encoded levels equal to `value`?

    Only run *headers* are read, so this is O(runs), not O(values). A
    bit-packed run answers `False` without being unpacked: a page whose levels
    are all the maximum is written as RLE runs by every writer in the wild,
    and that is the case worth the fast path.
    """
    if width == 0:
        return value == 0
    var pos = 0
    var done = 0
    var nbytes = (width + 7) // 8
    while done < count:
        if pos >= len(data):
            return False
        var head = read_uleb128(data, pos)
        pos = head[1]
        var indicator = head[0]
        if (indicator & 1) == 1:
            return False
        var n = Int(indicator >> 1)
        if n <= 0 or pos + nbytes > len(data):
            return False
        var v: UInt64 = 0
        for k in range(nbytes):
            v |= UInt64(data[pos + k]) << UInt64(8 * k)
        pos += nbytes
        if Int(v) != value:
            return False
        done += n
    return True


def decode_hybrid_into[
    dt: DType
](
    data: Span[UInt8, _],
    width: Int,
    count: Int,
    mut out: List[Scalar[dt]],
    at: Int,
) raises:
    """Decode `count` RLE/bit-packed hybrid values into `out[at : at + count]`,
    which the caller has already sized.

    Runs are handled whole — a repeated run is a fill, a bit-packed run is one
    `unpack_lsb_into` — so nothing goes through a per-value decoder. This is
    the hot path for dictionary indices.
    """
    if count == 0:
        return
    if at < 0 or at + count > len(out):
        raise Error("parquet.rle: hybrid destination is too small")
    var pos = 0
    var done = 0
    var dst = out.unsafe_ptr()
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
            pos = unpack_lsb_into[dt](data, pos, width, n, out, at + done)
            done += n
        else:
            var n = Int(indicator >> 1)
            if n <= 0:
                raise Error(String("parquet.rle: RLE run of ", n, " values"))
            if n > count - done:
                n = count - done
            var nbytes = (width + 7) // 8
            if pos + nbytes > len(data):
                raise Error(
                    String("parquet.rle: truncated RLE run value at ", pos)
                )
            var v: UInt64 = 0
            for k in range(nbytes):
                v |= UInt64(data[pos + k]) << UInt64(8 * k)
            pos += nbytes
            var sv = v.cast[dt]()
            for i in range(at + done, at + done + n):
                dst.unsafe_store(i, sv)
            done += n


comptime HYBRID_BLOCK = 1024
"""How many values `HybridRuns.next_block` hands out at a time.

Four kilobytes of `UInt32` scratch, which stays in L1 between the decode and
whatever consumes it. parquet-cpp's `ProcessValues` and arrow-rs's
`RleDecoder::get_batch_with_dict` both use exactly 1024. It has to be a
multiple of eight, for the reason `next_block` gives.
"""


struct HybridRuns[origin: ImmOrigin](Copyable, Movable):
    """An RLE/bit-packed hybrid stream, read a block at a time.

    `decode_hybrid_into` materialises the whole stream. A consumer that reads
    each value once and then throws it away — the dictionary gather — pays for
    that array twice, once written and once read straight back, and pays for
    the allocation under it. This hands the same values out in blocks of
    `HYBRID_BLOCK` into a scratch buffer the caller reuses.

    Every run header, bit width and length is validated exactly as
    `decode_hybrid_into` validates it, with the same errors.
    """

    var data: Span[UInt8, Self.origin]
    var width: Int
    var pos: Int
    var left: Int
    """Values still owed to the caller."""
    var run_left: Int
    """Values still to come out of the run in progress."""
    var run_value: UInt64
    var run_packed: Bool

    def __init__(
        out self, data: Span[UInt8, Self.origin], width: Int, count: Int
    ):
        self.data = data
        self.width = width
        self.pos = 0
        self.left = count
        self.run_left = 0
        self.run_value = 0
        self.run_packed = False

    def __init__(out self, *, copy: Self):
        self.data = copy.data
        self.width = copy.width
        self.pos = copy.pos
        self.left = copy.left
        self.run_left = copy.run_left
        self.run_value = copy.run_value
        self.run_packed = copy.run_packed

    def __init__(out self, *, deinit move: Self):
        self.data = move.data
        self.width = move.width
        self.pos = move.pos
        self.left = move.left
        self.run_left = move.run_left
        self.run_value = move.run_value
        self.run_packed = move.run_packed

    def _load_run(mut self) raises:
        if self.pos >= len(self.data):
            raise Error("parquet.rle: ran out of runs before all values")
        var head = read_uleb128(self.data, self.pos)
        self.pos = head[1]
        var indicator = head[0]
        if (indicator & 1) == 1:
            var n = Int(indicator >> 1) * 8
            if n <= 0:
                raise Error(String("parquet.rle: bit-packed run of ", n))
            self.run_left = n
            self.run_packed = True
            return
        var n = Int(indicator >> 1)
        if n <= 0:
            raise Error(String("parquet.rle: RLE run of ", n, " values"))
        var nbytes = (self.width + 7) // 8
        if self.pos + nbytes > len(self.data):
            raise Error(
                String("parquet.rle: truncated RLE run value at ", self.pos)
            )
        var v: UInt64 = 0
        for k in range(nbytes):
            v |= UInt64(self.data[self.pos + k]) << UInt64(8 * k)
        self.pos += nbytes
        self.run_value = v
        self.run_left = n
        self.run_packed = False

    def next_block(
        mut self, mut scratch: List[UInt32]
    ) raises -> Tuple[Int, Bool]:
        """Decode the next values into `scratch[0 : n]`; return `(n, uniform)`.

        `n` is zero once every value asked for has been handed out. `uniform`
        says the whole block came out of one repeated run, so a consumer that
        has to validate the values can do it with one comparison instead of a
        scan.

        `scratch` must hold at least `HYBRID_BLOCK` values, and a block is that
        many — a multiple of eight, which is what lets a bit-packed run be
        taken a block at a time. Eight values of `width` bits are exactly
        `width` bytes, so a block boundary inside a run is still a byte
        boundary; only the very last block of the stream may be shorter, and
        nothing is read after it.
        """
        if len(scratch) < HYBRID_BLOCK:
            raise Error("parquet.rle: hybrid block scratch is too small")
        if self.left == 0:
            return (0, False)
        while self.run_left == 0:
            self._load_run()
        var n = self.run_left
        if n > HYBRID_BLOCK:
            n = HYBRID_BLOCK
        if n > self.left:
            n = self.left
        if self.run_packed:
            _ = unpack_lsb_into[DType.uint32](
                self.data, self.pos, self.width, n, scratch, 0
            )
            self.pos += (n * self.width) // 8
        else:
            var v = self.run_value.cast[DType.uint32]()
            var dst = scratch.unsafe_ptr()
            for i in range(n):
                dst.unsafe_store(i, v)
        self.run_left -= n
        self.left -= n
        return (n, not self.run_packed)


def decode_levels_into(
    data: Span[UInt8, _],
    width: Int,
    count: Int,
    max_level: Int,
    mut out: List[UInt16],
) raises -> Int:
    """Append `count` hybrid-encoded levels to `out`; return how many of them
    equal `max_level`.

    Runs are handled whole: a repeated run is a fill, a bit-packed run is one
    `unpack_lsb_into` call. Nothing goes through a per-value decoder.
    """
    var base = len(out)
    out.resize(base + count, 0)
    if count == 0:
        return 0
    var pos = 0
    var done = 0
    var non_null = 0
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
            pos = unpack_lsb_into[DType.uint16](
                data, pos, width, n, out, base + done
            )
            var ptr = out.unsafe_ptr()
            for i in range(base + done, base + done + n):
                var v = Int(ptr.unsafe_load(i))
                if v > max_level:
                    raise Error(
                        String(
                            "parquet.page: level ",
                            v,
                            " exceeds the column maximum of ",
                            max_level,
                        )
                    )
                if v == max_level:
                    non_null += 1
            done += n
        else:
            var n = Int(indicator >> 1)
            if n <= 0:
                raise Error(String("parquet.rle: RLE run of ", n, " values"))
            if n > count - done:
                n = count - done
            var nbytes = (width + 7) // 8
            if pos + nbytes > len(data):
                raise Error(
                    String("parquet.rle: truncated RLE run value at ", pos)
                )
            var v: UInt64 = 0
            for k in range(nbytes):
                v |= UInt64(data[pos + k]) << UInt64(8 * k)
            pos += nbytes
            if Int(v) > max_level:
                raise Error(
                    String(
                        "parquet.page: level ",
                        v,
                        " exceeds the column maximum of ",
                        max_level,
                    )
                )
            var lv = UInt16(v)
            var ptr = out.unsafe_ptr()
            for i in range(base + done, base + done + n):
                ptr.unsafe_store(i, lv)
            if Int(v) == max_level:
                non_null += n
            done += n
    return non_null


def unpack_msb(
    data: Span[UInt8, _],
    start_byte: Int,
    width: Int,
    count: Int,
    mut out: List[UInt64],
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
            self.pos = unpack_lsb(
                self.data, self.pos, self.width, count, self.buffer
            )
            self.buf_pos = 0
            self.remaining = count
            self.is_packed = True
        else:
            var count = Int(indicator >> 1)
            if count <= 0:
                raise Error(
                    String("parquet.rle: RLE run of ", count, " values")
                )
            var nbytes = (self.width + 7) // 8
            if self.pos + nbytes > len(self.data):
                raise Error(
                    String("parquet.rle: truncated RLE run value at ", self.pos)
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
