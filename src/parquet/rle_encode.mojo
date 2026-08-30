"""Encoders for the two bit-level formats a Parquet writer needs."""


struct BitPacker(Copyable, Defaultable, Movable):
    """Packs values of a fixed bit width, least-significant bit first."""

    var out: List[UInt8]
    var bit: Int

    def __init__(out self):
        self.out = List[UInt8]()
        self.bit = 0

    def __init__(out self, *, copy: Self):
        self.out = copy.out.copy()
        self.bit = copy.bit

    def __init__(out self, *, deinit move: Self):
        self.out = move.out^
        self.bit = move.bit

    def put(mut self, value: UInt64, width: Int):
        var written = 0
        while written < width:
            var byte = self.bit // 8
            while len(self.out) <= byte:
                self.out.append(0)
            var off = self.bit % 8
            var room = 8 - off
            var take = width - written
            if take > room:
                take = room
            var mask = (UInt64(1) << UInt64(take)) - 1
            var chunk = (value >> UInt64(written)) & mask
            self.out[byte] |= UInt8(chunk << UInt64(off))
            written += take
            self.bit += take

    def align(mut self):
        self.bit = ((self.bit + 7) // 8) * 8


def write_uleb128(mut out: List[UInt8], value: UInt64):
    var v = value
    while True:
        var b = UInt8(v & 0x7F)
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def encode_hybrid(values: List[UInt16], width: Int) raises -> List[UInt8]:
    """The RLE / bit-packed hybrid: repeated runs of eight or more values, and
    bit-packed groups for everything else."""
    var out = List[UInt8]()
    if len(values) == 0:
        return out^
    if width == 0:
        # Every value is zero and carries no bits, but the run header still
        # has to be there — arrow-cpp's decoder reads one before it will hand
        # out a single value.
        write_uleb128(out, UInt64(len(values)) << 1)
        return out^
    var n = len(values)
    var nbytes = (width + 7) // 8
    var i = 0
    while i < n:
        var j = i + 1
        while j < n and values[j] == values[i]:
            j += 1
        if j - i >= 8:
            write_uleb128(out, UInt64(j - i) << 1)
            var v = UInt64(values[i])
            for k in range(nbytes):
                out.append(UInt8((v >> UInt64(8 * k)) & 0xFF))
            i = j
            continue
        # Collect values until a run of eight shows up, then bit-pack them.
        var k = i
        while k < n:
            var m = k + 1
            while m < n and values[m] == values[k]:
                m += 1
            if m - k >= 8:
                break
            k = m
        var count = k - i
        var groups = (count + 7) // 8
        write_uleb128(out, (UInt64(groups) << 1) | 1)
        var packer = BitPacker()
        for g in range(groups * 8):
            var idx = i + g
            packer.put(UInt64(values[idx]) if idx < n else 0, width)
        packer.align()
        out.extend(Span(packer.out))
        i += groups * 8
    return out^


def encode_levels(values: List[UInt16], width: Int) raises -> List[UInt8]:
    """Levels as a v1 data page wants them: a 4-byte length, then the hybrid."""
    var body = encode_hybrid(values, width)
    var out = List[UInt8](capacity=len(body) + 4)
    var n = UInt32(len(body))
    for k in range(4):
        out.append(UInt8((n >> UInt32(8 * k)) & 0xFF))
    out.extend(Span(body))
    return out^
