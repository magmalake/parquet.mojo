"""A byte-exact fingerprint of a decoded table, for the worker-count tests.

`parity.check_table` compares a table against pyarrow. This compares a table
against *itself*, read a second way: it walks every node of every batch's arena
and folds the Arrow buffers — values, validity, offsets and large offsets —
together with the lengths, null counts, types, names and field ids into one
CRC32. Two reads that differ in a single bit of a buffer, in one null count or
in one offset produce different fingerprints, which is what
`test_num_workers_is_bit_identical` needs and what a row-count or a value-level
comparison would miss.
"""

from hashes import crc32
from parquet import ArrayArena, CodecSet, ParquetReader, Table
from std.memory import bitcast


def _put_u64(mut buf: List[UInt8], v: UInt64):
    for k in range(8):
        buf.append(UInt8((v >> UInt64(8 * k)) & 0xFF))


def _fold_tree(arena: ArrayArena, root: Int, mut h: UInt32) raises:
    """Fold an array and everything under it into `h`, in pre-order.

    An explicit stack rather than recursion: children are pushed in reverse so
    they pop back in schema order, which is what makes the fingerprint depend
    on the tree's shape and not on how it was walked.
    """
    var stack: List[Int] = [root]
    while len(stack):
        var node = stack.pop()
        _fold_node(arena, node, h)
        ref kids = arena.nodes[node].children
        for k in range(len(kids) - 1, -1, -1):
            stack.append(kids[k])


def _fold_node(arena: ArrayArena, node: Int, mut h: UInt32) raises:
    """Fold one array's own buffers and counts into `h`."""
    ref a = arena.nodes[node]
    var buf = List[UInt8]()
    _put_u64(buf, UInt64(a.length))
    _put_u64(buf, UInt64(a.null_count))
    _put_u64(buf, UInt64(1) if a.nullable else UInt64(0))
    _put_u64(buf, UInt64(bitcast[DType.uint32](a.field_id)))
    _put_u64(buf, UInt64(len(a.validity)))
    _put_u64(buf, UInt64(len(a.offsets)))
    _put_u64(buf, UInt64(len(a.large_offsets)))
    _put_u64(buf, UInt64(len(a.values)))
    _put_u64(buf, UInt64(len(a.children)))
    for k in range(len(a.offsets)):
        _put_u64(buf, UInt64(bitcast[DType.uint32](a.offsets[k])))
    for k in range(len(a.large_offsets)):
        _put_u64(buf, bitcast[DType.uint64](a.large_offsets[k]))
    h = crc32(a.name.as_bytes(), h)
    h = crc32(String(a.type).as_bytes(), h)
    h = crc32(Span(buf), h)
    h = crc32(Span(a.validity), h)
    h = crc32(Span(a.values), h)


def table_fingerprint(t: Table) raises -> UInt32:
    """One CRC32 over everything a `Table` holds, batch boundaries included."""
    var h: UInt32 = 0
    var buf = List[UInt8]()
    _put_u64(buf, UInt64(t.num_rows))
    _put_u64(buf, UInt64(len(t.batches)))
    h = crc32(Span(buf), h)
    for b in range(len(t.batches)):
        ref batch = t.batches[b]
        var head = List[UInt8]()
        _put_u64(head, UInt64(batch.num_rows))
        _put_u64(head, UInt64(len(batch.roots)))
        h = crc32(Span(head), h)
        for c in range(len(batch.roots)):
            _fold_tree(batch.arena, batch.roots[c], h)
    return h


def read_fingerprint[
    Codecs: CodecSet
](path: StringSlice, workers: Int, batch_size: Int) raises -> UInt32:
    """Read `path` with `workers` workers and fingerprint what came out."""
    var r = ParquetReader[Codecs].open(path)
    r.num_workers = workers
    r.batch_size = batch_size
    return table_fingerprint(r.read_table())


def read_error[Codecs: CodecSet](data: Span[UInt8, _], workers: Int) -> String:
    """The message a read of `data` fails with, or "" if it succeeded."""
    try:
        var r = ParquetReader[Codecs].from_span(data)
        r.num_workers = workers
        _ = r.read_table()
        return String()
    except e:
        return String(e)
