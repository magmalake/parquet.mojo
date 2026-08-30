"""Split-block bloom filters — `might_contain` for equality predicates.

Parquet's bloom filter is the "split block" design: the bit set is a sequence
of 32-byte blocks of eight 32-bit words, a value picks one block from the top
32 bits of its XXH64 hash, and eight salts turn the bottom 32 bits into one bit
per word. A value is definitely absent if any of those eight bits is clear.

```mojo
var bf = read_bloom_filter(Span(file_bytes), cm)
if bf and not bf.value().might_contain_string("alice"):
    ...  # this row group cannot contain it
```
"""

from hashes import xxh64
from parquet.stats import (
    SV_BOOL,
    SV_BYTES,
    SV_FLOAT,
    SV_INT,
    SV_UINT,
    ScalarValue,
)
from std.memory import bitcast
from thrift import BloomFilterHeader, ColumnMetaData, TCompactProtocolReader

comptime BLOOM_BLOCK_BYTES = 32

comptime SALT0: UInt32 = 0x47B6137B
comptime SALT1: UInt32 = 0x44974D91
comptime SALT2: UInt32 = 0x8824AD5B
comptime SALT3: UInt32 = 0xA2B7289D
comptime SALT4: UInt32 = 0x705495C7
comptime SALT5: UInt32 = 0x2DF1424B
comptime SALT6: UInt32 = 0x9EFC4947
comptime SALT7: UInt32 = 0x5C6BFB31


def _salt(i: Int) -> UInt32:
    if i == 0:
        return SALT0
    if i == 1:
        return SALT1
    if i == 2:
        return SALT2
    if i == 3:
        return SALT3
    if i == 4:
        return SALT4
    if i == 5:
        return SALT5
    if i == 6:
        return SALT6
    return SALT7


struct BloomFilter(Copyable, Movable, Defaultable):
    """One column chunk's split-block bloom filter bit set."""

    var bits: List[UInt8]

    def __init__(out self):
        self.bits = List[UInt8]()

    def __init__(out self, var bits: List[UInt8]):
        self.bits = bits^

    def __init__(out self, *, copy: Self):
        self.bits = copy.bits.copy()

    def __init__(out self, *, deinit move: Self):
        self.bits = move.bits^

    def num_blocks(self) -> Int:
        return len(self.bits) // BLOOM_BLOCK_BYTES

    def _word(self, block: Int, i: Int) -> UInt32:
        var base = block * BLOOM_BLOCK_BYTES + i * 4
        var v: UInt32 = 0
        for k in range(4):
            v |= UInt32(self.bits[base + k]) << UInt32(8 * k)
        return v

    def might_contain_hash(self, hash: UInt64) -> Bool:
        var blocks = self.num_blocks()
        if blocks == 0:
            return True
        var block = Int(((hash >> 32) * UInt64(blocks)) >> 32)
        var key = UInt32(hash & 0xFFFFFFFF)
        for i in range(8):
            var bit = (key * _salt(i)) >> 27
            var mask = UInt32(1) << bit
            if (self._word(block, i) & mask) == 0:
                return False
        return True

    def might_contain_bytes(self, data: Span[UInt8, _]) raises -> Bool:
        return self.might_contain_hash(xxh64(data))

    def might_contain_string(self, text: StringSlice) raises -> Bool:
        return self.might_contain_bytes(text.as_bytes())

    def might_contain_i32(self, v: Int32) raises -> Bool:
        return self.might_contain_bytes(Span(_le(UInt64(bitcast[DType.uint32](v)), 4)))

    def might_contain_i64(self, v: Int64) raises -> Bool:
        return self.might_contain_bytes(Span(_le(bitcast[DType.uint64](v), 8)))

    def might_contain_f32(self, v: Float32) raises -> Bool:
        return self.might_contain_bytes(
            Span(_le(UInt64(bitcast[DType.uint32](v)), 4))
        )

    def might_contain_f64(self, v: Float64) raises -> Bool:
        return self.might_contain_bytes(Span(_le(bitcast[DType.uint64](v), 8)))

    def might_contain(self, value: ScalarValue) raises -> Bool:
        """Hash a `ScalarValue` the way Parquet's PLAIN encoding would."""
        if value.kind == SV_BYTES:
            return self.might_contain_bytes(Span(value.b))
        if value.kind == SV_INT:
            return self.might_contain_i64(value.i)
        if value.kind == SV_UINT:
            return self.might_contain_i64(bitcast[DType.int64](value.u))
        if value.kind == SV_FLOAT:
            return self.might_contain_f64(value.f)
        if value.kind == SV_BOOL:
            var one = List[UInt8]()
            one.append(UInt8(1) if value.i else UInt8(0))
            return self.might_contain_bytes(Span(one))
        return True


def _le(v: UInt64, n: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=n)
    for k in range(n):
        out.append(UInt8((v >> UInt64(8 * k)) & 0xFF))
    return out^


def read_bloom_filter(file: Span[UInt8, _], cm: ColumnMetaData) raises -> Optional[BloomFilter]:
    """Read the bloom filter of one column chunk, or `None` if it has none."""
    if not cm.bloom_filter_offset:
        return None
    var off = Int(cm.bloom_filter_offset.value())
    if off <= 0 or off >= len(file):
        raise Error(
            String("parquet.bloom: bloom filter offset ", off, " is outside the file")
        )
    var p = TCompactProtocolReader(file, off)
    var hdr = BloomFilterHeader()
    hdr.read(p)
    var start = p.offset()
    var n = Int(hdr.numBytes)
    if n < 0 or start + n > len(file):
        raise Error(
            String(
                "parquet.bloom: bit set of ",
                n,
                " byte(s) at ",
                start,
                " runs past the file",
            )
        )
    if not hdr.algorithm.BLOCK:
        raise Error("parquet.bloom: only the split-block algorithm is defined")
    if not hdr.hash.XXHASH:
        raise Error("parquet.bloom: only the XXH64 hash is defined")
    if not hdr.compression.UNCOMPRESSED:
        raise Error("parquet.bloom: compressed bloom filters are not supported")
    if n % BLOOM_BLOCK_BYTES != 0:
        raise Error(
            String("parquet.bloom: bit set of ", n, " bytes is not a whole number of blocks")
        )
    var bits = List[UInt8](capacity=n)
    bits.extend(file[start : start + n])
    return BloomFilter(bits^)
