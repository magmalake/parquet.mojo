"""A byte-exact fingerprint of a decoded table, for the worker-count tests.

`parity.check_table` compares a table against pyarrow. This compares a table
against *itself*, read a second way: it walks every node of every batch's arena
and folds the Arrow buffers — values, validity, offsets and large offsets —
together with the lengths, null counts, types, names and field ids into one
CRC32. Two reads that differ in a single bit of a buffer, in one null count or
in one offset produce different fingerprints, which is what
`test_num_workers_is_bit_identical` needs and what a row-count or a value-level
comparison would miss.

**Two halves, because values are not the whole answer.** `ArrayData` refers to
its children by *arena index*, so a batch has a shape the values alone do not
pin down: build the top-level fields in a different order and every index
shifts, giving a structurally different arena out of identical Arrow values.
A tree walk cannot see that — it starts at the roots and follows the indices
wherever they lead, so an arena permuted consistently folds to exactly the same
number. So the fingerprint is the sum of two folds:

* `_fold_tree` — every array's buffers and counts, in pre-order from each root.
  This is the values half, and it is what the file has always folded.
* `_fold_layout` — the arena as a flat list: how many nodes, which root indices,
  and every node's own index alongside its children's. This is the layout half,
  and it is what makes assembling fields in the wrong order a test failure
  rather than a silent structural change.

`permuted_arenas` is the negative control for the second half: it renumbers a
real table's arenas without touching a single value, and
`test_the_fingerprint_catches_a_permuted_arena` asserts that the values half
misses it and the whole fingerprint does not.
"""

from hashes import crc32
from parquet import ArrayArena, CodecSet, ParquetReader, RecordBatch, Table
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


def _fold_layout(arena: ArrayArena, roots: List[Int], mut h: UInt32) raises:
    """Fold the arena's *shape* into `h`: node order and every child index.

    A flat pass in index order, not a walk from the roots, which is the point:
    it is the one thing a tree walk cannot see. An arena whose nodes were
    appended in a different order — because two top-level fields were built on
    two threads and stitched back the wrong way round — folds differently here
    even though every array in it holds the same bytes.
    """
    var buf = List[UInt8]()
    _put_u64(buf, UInt64(len(arena.nodes)))
    _put_u64(buf, UInt64(len(roots)))
    for k in range(len(roots)):
        _put_u64(buf, UInt64(roots[k]))
    for i in range(len(arena.nodes)):
        ref a = arena.nodes[i]
        _put_u64(buf, UInt64(i))
        _put_u64(buf, UInt64(len(a.children)))
        for k in range(len(a.children)):
            _put_u64(buf, UInt64(a.children[k]))
    h = crc32(Span(buf), h)


def table_values_fingerprint(t: Table) raises -> UInt32:
    """The values half alone: every array's buffers and counts, from the roots.

    Exposed only so the negative control can show what it misses. Tests want
    `table_fingerprint`.
    """
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


def table_fingerprint(t: Table) raises -> UInt32:
    """One CRC32 over everything a `Table` holds, batch boundaries included.

    Values *and* arena layout — see the module docstring for why the second
    half is not redundant.
    """
    var h = table_values_fingerprint(t)
    for b in range(len(t.batches)):
        ref batch = t.batches[b]
        _fold_layout(batch.arena, batch.roots, h)
    return h


def permuted_arenas(t: Table) raises -> Table:
    """`t` with every batch's arena renumbered, and not one value changed.

    The negative control for the layout half. Each arena is reversed — node `i`
    becomes node `n - 1 - i` — and every child index and every root index is
    remapped to match, so the trees are the ones that went in and the arrays
    hold the bytes they held. This is exactly the damage that building the
    top-level fields on several threads and stitching them back in the wrong
    order would do, and nothing else: if `table_fingerprint` cannot tell this
    apart from `t`, it cannot tell a broken assembly order apart either.
    """
    var out = Table()
    out.num_rows = t.num_rows
    for b in range(len(t.batches)):
        ref batch = t.batches[b]
        var n = len(batch.arena.nodes)
        var moved = RecordBatch()
        moved.num_rows = batch.num_rows
        for i in range(n):
            var a = batch.arena.nodes[n - 1 - i].copy()
            for c in range(len(a.children)):
                a.children[c] = n - 1 - a.children[c]
            _ = moved.arena.add(a^)
        for c in range(len(batch.roots)):
            moved.roots.append(n - 1 - batch.roots[c])
        out.batches.append(moved^)
    return out^


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
