"""The *same* bodies, driven on the CPU — the other half of the shared source.

Without this file the claim "one source, two targets" would be an assertion.
`shared_gather_into` binds `TARGET_CPU` on `gather_fixed_body`,
`var_lengths_body` and `var_copy_body` — the exact functions
`gather_fixed_kernel` and friends bind `TARGET_GPU` on — over ordinary host
`List` memory, and the parity gate checks its output against
`parquet.encoding.gather_into` on every page it checks the GPU against. So the
comptime branch is instantiated both ways in one binary and both are proven
correct.

It is deliberately **not** a proposed replacement for `gather_into`. Two
reasons, and they are the substance of the spike's first answer:

* the shared body is scalar, where `gather_into` moves 4- and 8-byte entries
  with one widened load and store, so this is the slower of the two;
* everything outside the element body still forks. Look at the `BYTE_ARRAY`
  path below: the running sum in `_prefix_sum_host` is four lines, and its GPU
  counterpart is three kernels (`scan_block_kernel`, `scan_sums_kernel`,
  `scan_add_kernel`) plus a copy back to the host to size the output. Nothing
  is shared there, and nothing could be.
"""

from layout import TileTensor, row_major

from parquet.encoding import PK_BOOL, PK_VAR, PhysBuffer

from gather_shared import (
    TARGET_CPU,
    gather_fixed_body,
    var_copy_body,
    var_lengths_body,
)


def _out_of_range(index: UInt32, entries: Int) -> Error:
    """The out-of-range error, worded exactly as `parquet.encoding` words it.

    Args:
        index: The offending dictionary index.
        entries: How many entries the dictionary has.

    Returns:
        The error to raise.
    """
    return Error(
        String(
            "parquet.encoding: dictionary index ",
            index,
            " out of range (dictionary has ",
            entries,
            " entries)",
        )
    )


def _prefix_sum_host(mut lens: List[UInt32]) -> Int:
    """Turn per-value lengths into an inclusive prefix sum, in place.

    The CPU half of the fork. Its GPU counterpart is `scan_block_kernel`,
    `scan_sums_kernel` and `scan_add_kernel`, plus the copy back that lets the
    host size the output buffer.

    Args:
        lens: Per-value byte lengths in, inclusive prefix sum out.

    Returns:
        The total.
    """
    var total = 0
    var p = lens.unsafe_ptr()
    for i in range(len(lens)):
        total += Int(p.unsafe_load(i))
        p.unsafe_store(i, UInt32(total))
    return total


def shared_gather_into(
    mut out: PhysBuffer, dict: PhysBuffer, indices: List[UInt32]
) raises:
    """`gather_into`, driven on the CPU from the bodies the kernels share.

    Args:
        out: The output buffer, appended to.
        dict: The dictionary page.
        indices: The decoded dictionary indices.

    Raises:
        If the buffers disagree in shape or an index is out of range.
    """
    var n = len(indices)
    if n == 0:
        return
    if out.kind != dict.kind:
        raise Error("parquet.encoding: cannot concatenate unlike pages")
    if dict.kind == PK_BOOL:
        raise Error(
            "experiments.gpu_gather: a BOOLEAN dictionary keeps the CPU path"
        )

    var entries = dict.count
    for i in range(n):
        if Int(indices[i]) >= entries:
            raise _out_of_range(indices[i], entries)

    # `TileTensor` wants a mutable origin, so the read-only inputs are copied
    # into locals this function owns. A shipping implementation would not; the
    # point here is that the bodies bind and run, not that they are fast.
    var idx = indices.copy()
    var dbytes = dict.bytes.copy()
    var li = row_major(n)
    var ls = row_major(len(dbytes) if len(dbytes) > 0 else 1)

    if dict.kind == PK_VAR:
        var doff = List[UInt32](length=len(dict.offsets), fill=0)
        for i in range(len(dict.offsets)):
            doff[i] = UInt32(Int(dict.offsets[i]))
        var lens = List[UInt32](length=n, fill=0)
        var ln = row_major(n)
        var lo = row_major(len(doff))
        var_lengths_body[TARGET_CPU](
            TileTensor(lens.unsafe_ptr(), ln),
            TileTensor(doff.unsafe_ptr(), lo),
            TileTensor(idx.unsafe_ptr(), li),
            n,
        )
        var total = _prefix_sum_host(lens)

        var vbase = len(out.bytes)
        var obase = len(out.offsets)
        out.offsets.resize(obase + n, 0)
        for i in range(n):
            out.offsets[obase + i] = Int32(vbase + Int(lens[i]))
        out.bytes.resize(vbase + total, 0)
        var ld = row_major(total if total > 0 else 1)
        var_copy_body[TARGET_CPU](
            TileTensor(out.bytes.unsafe_ptr().unsafe_offset(vbase), ld),
            TileTensor(dbytes.unsafe_ptr(), ls),
            TileTensor(doff.unsafe_ptr(), lo),
            TileTensor(idx.unsafe_ptr(), li),
            TileTensor(lens.unsafe_ptr(), ln),
            n,
        )
        out.count += n
        return

    var w = dict.width
    if out.count == 0 and len(out.bytes) == 0:
        out.width = w
    if out.width != w:
        raise Error("parquet.encoding: cannot concatenate unlike pages")
    var vbase = len(out.bytes)
    out.bytes.resize(vbase + n * w, 0)
    var ld = row_major(n * w if n * w > 0 else 1)
    gather_fixed_body[TARGET_CPU](
        TileTensor(out.bytes.unsafe_ptr().unsafe_offset(vbase), ld),
        TileTensor(dbytes.unsafe_ptr(), ls),
        TileTensor(idx.unsafe_ptr(), li),
        w,
        n,
    )
    out.count += n
