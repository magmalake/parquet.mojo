"""The dictionary gather written once, for two targets.

This file is the answer to the spike's first question — *can the gather logic
be genuinely shared between CPU and GPU, comptime-specialised over the target,
or does it fork into two implementations that merely agree?*

The shape it settles on:

* `_copy_entry` and `_entry_len` are the **element bodies**. They are target
  agnostic — indexing and byte moves over `TileTensor`, no `global_idx`, no
  `barrier`, no allocation. Both targets run this exact code.
* `gather_fixed_body[target]` and the two `..._var_..._body[target]` functions
  are the **drivers**. One `comptime if` picks between "my thread does element
  `global_idx.x`" and "this thread does every element in a loop". Everything
  under the branch is the shared body.
* `gather_fixed_kernel` and the other `..._kernel` functions are one-line
  entry points that bind `target = TARGET_GPU`. `host_driver.mojo` binds
  `TARGET_CPU` on the same three bodies, and the parity gate checks *that*
  against the shipping oracle too — so both instantiations of the branch are
  in one binary and both are proven correct.

So the *body* is shared and the *driver* is a two-line comptime branch. What is
**not** shared, and cannot be, is above that line: allocation, the bounds check
(a GPU kernel cannot raise), and — for `BYTE_ARRAY` — the offset prefix sum,
which is a running total on the CPU and a two-level block scan on the GPU.
`scan_block_kernel` below has no CPU counterpart at all. See the README.

Correctness note: the element bodies here are *scalar*. `parquet.encoding`'s
`gather_into` is not — it moves 4- and 8-byte entries with a single widened
load and store, and byte arrays eight bytes at a time. That vectorisation is
the CPU driver's, not the body's, so a shared body means the CPU side of the
share is the slow one. The GPU is therefore measured against the real
`gather_into`, never against `host_driver.shared_gather_into`.
"""

from max.gpu import global_idx, thread_idx, block_idx
from std.math import ceildiv
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout

comptime TARGET_CPU = 0
"""Run the body once per element, on the calling thread."""
comptime TARGET_GPU = 1
"""Run the body for the element this GPU thread owns."""

comptime U8 = DType.uint8
comptime U32 = DType.uint32

comptime BLOCK = 256
"""Threads per block for the gather kernels."""
comptime SCAN_BLOCK = 512
"""Elements (and threads) per block for `scan_block_kernel`."""
comptime SCAN_SUMS = 1024
"""Threads in the single-block scan of the per-block sums. The offset scan
therefore tops out at `SCAN_BLOCK * SCAN_SUMS` = 524288 values per call;
`gpu_gather.mojo` raises rather than silently truncating past that."""


@always_inline
def gpu_grid[sabotage: Bool](n: Int) -> Int:
    """Blocks of `BLOCK` threads needed to cover `n` elements.

    `sabotage` is the negative control: it launches one block short, so every
    page past a single block loses its tail. The parity gate runs itself a
    second time with `sabotage = True` and fails unless *that* run fails.

    One block short is the classic GPU bug, and it is deliberately a *partial*
    one: pages of `BLOCK` values or fewer still come out right. A gate that
    only ever saw a 256-value page would not notice — which is exactly what
    the control is there to disprove.

    Parameters:
        sabotage: Whether to launch one block short.

    Args:
        n: How many elements the launch must cover.

    Returns:
        The grid dimension.
    """
    var g = ceildiv(n, BLOCK)
    comptime if sabotage:
        return g - 1 if g > 1 else g
    else:
        return g


# ── the element bodies — target agnostic, run verbatim by both drivers ──────


@always_inline
def _copy_entry[
    LD: TensorLayout, LS: TensorLayout
](
    dst: TileTensor[U8, LD, MutAnyOrigin],
    src: TileTensor[U8, LS, MutAnyOrigin],
    dst_at: Int,
    src_at: Int,
    n: Int,
):
    """Move `n` bytes of one dictionary entry into one output slot.

    Args:
        dst: Output bytes.
        src: Dictionary bytes.
        dst_at: Byte offset in `dst` to write at.
        src_at: Byte offset in `src` to read from.
        n: How many bytes this entry is.
    """
    comptime assert dst.flat_rank == 1, "gather buffers are flat"
    comptime assert src.flat_rank == 1, "gather buffers are flat"
    for b in range(n):
        dst[dst_at + b] = rebind[dst.ElementType](src[src_at + b])


@always_inline
def _entry_len[
    LO: TensorLayout
](offsets: TileTensor[U32, LO, MutAnyOrigin], k: Int) -> Int:
    """How many bytes dictionary entry `k` holds, from the offsets array.

    Args:
        offsets: The dictionary's `count + 1` byte offsets.
        k: Which entry.

    Returns:
        The entry's byte length.
    """
    return Int(rebind[Scalar[U32]](offsets[k + 1])) - Int(
        rebind[Scalar[U32]](offsets[k])
    )


@always_inline
def _index_at[
    LI: TensorLayout
](idx: TileTensor[U32, LI, MutAnyOrigin], i: Int) -> Int:
    """The dictionary index for output slot `i`.

    Args:
        idx: Decoded dictionary indices.
        i: Which output slot.

    Returns:
        The dictionary entry the slot names.
    """
    return Int(rebind[Scalar[U32]](idx[i]))


# ── the drivers — one comptime branch, then the shared body ─────────────────


@always_inline
def gather_fixed_body[
    LD: TensorLayout, LS: TensorLayout, LI: TensorLayout, //, target: Int
](
    dst: TileTensor[U8, LD, MutAnyOrigin],
    src: TileTensor[U8, LS, MutAnyOrigin],
    idx: TileTensor[U32, LI, MutAnyOrigin],
    w: Int,
    n: Int,
):
    """Fixed-width gather: `dst[i] = src[idx[i]]`, `w` bytes each.

    Parameters:
        target: `TARGET_CPU` or `TARGET_GPU`.
        LD: Layout of `dst`.
        LS: Layout of `src`.
        LI: Layout of `idx`.

    Args:
        dst: Output bytes, `n * w` of them.
        src: Dictionary bytes.
        idx: Decoded dictionary indices, `n` of them.
        w: Bytes per entry.
        n: How many values to gather.
    """
    comptime if target == TARGET_GPU:
        var i = Int(global_idx.x)
        if i < n:
            _copy_entry(dst, src, i * w, _index_at(idx, i) * w, w)
    else:
        for i in range(n):
            _copy_entry(dst, src, i * w, _index_at(idx, i) * w, w)


@always_inline
def var_lengths_body[
    LL: TensorLayout, LO: TensorLayout, LI: TensorLayout, //, target: Int
](
    lens: TileTensor[U32, LL, MutAnyOrigin],
    doff: TileTensor[U32, LO, MutAnyOrigin],
    idx: TileTensor[U32, LI, MutAnyOrigin],
    n: Int,
):
    """`BYTE_ARRAY` pass 1: the byte length each output slot will hold.

    Parameters:
        target: `TARGET_CPU` or `TARGET_GPU`.
        LL: Layout of `lens`.
        LO: Layout of `doff`.
        LI: Layout of `idx`.

    Args:
        lens: Output lengths, `n` of them.
        doff: The dictionary's byte offsets.
        idx: Decoded dictionary indices.
        n: How many values.
    """
    comptime assert lens.flat_rank == 1, "gather buffers are flat"
    comptime if target == TARGET_GPU:
        var i = Int(global_idx.x)
        if i < n:
            lens[i] = rebind[lens.ElementType](
                UInt32(_entry_len(doff, _index_at(idx, i)))
            )
    else:
        for i in range(n):
            lens[i] = rebind[lens.ElementType](
                UInt32(_entry_len(doff, _index_at(idx, i)))
            )


@always_inline
def var_copy_body[
    LD: TensorLayout,
    LS: TensorLayout,
    LO: TensorLayout,
    LI: TensorLayout,
    LP: TensorLayout,
    //,
    target: Int,
](
    dst: TileTensor[U8, LD, MutAnyOrigin],
    src: TileTensor[U8, LS, MutAnyOrigin],
    doff: TileTensor[U32, LO, MutAnyOrigin],
    idx: TileTensor[U32, LI, MutAnyOrigin],
    incl: TileTensor[U32, LP, MutAnyOrigin],
    n: Int,
):
    """`BYTE_ARRAY` pass 3: move each value's bytes to where the scan put it.

    Parameters:
        target: `TARGET_CPU` or `TARGET_GPU`.
        LD: Layout of `dst`.
        LS: Layout of `src`.
        LO: Layout of `doff`.
        LI: Layout of `idx`.
        LP: Layout of `incl`.

    Args:
        dst: Output bytes.
        src: Dictionary bytes.
        doff: The dictionary's byte offsets.
        idx: Decoded dictionary indices.
        incl: Inclusive prefix sum of the lengths.
        n: How many values.
    """
    comptime if target == TARGET_GPU:
        var i = Int(global_idx.x)
        if i < n:
            var k = _index_at(idx, i)
            var ln = _entry_len(doff, k)
            var end = Int(rebind[Scalar[U32]](incl[i]))
            _copy_entry(
                dst, src, end - ln, Int(rebind[Scalar[U32]](doff[k])), ln
            )
    else:
        for i in range(n):
            var k = _index_at(idx, i)
            var ln = _entry_len(doff, k)
            var end = Int(rebind[Scalar[U32]](incl[i]))
            _copy_entry(
                dst, src, end - ln, Int(rebind[Scalar[U32]](doff[k])), ln
            )


# ── GPU entry points — bind target = TARGET_GPU and nothing else ────────────


def gather_fixed_kernel[
    LD: TensorLayout, LS: TensorLayout, LI: TensorLayout
](
    dst: TileTensor[U8, LD, MutAnyOrigin],
    src: TileTensor[U8, LS, MutAnyOrigin],
    idx: TileTensor[U32, LI, MutAnyOrigin],
    w_arg: Int32,
    n_arg: Int32,
):
    """One thread per output value; the body is `gather_fixed_body`.

    Parameters:
        LD: Layout of `dst`.
        LS: Layout of `src`.
        LI: Layout of `idx`.

    Args:
        dst: Output bytes.
        src: Dictionary bytes.
        idx: Decoded dictionary indices.
        w_arg: Bytes per entry.
        n_arg: How many values.
    """
    gather_fixed_body[TARGET_GPU](dst, src, idx, Int(w_arg), Int(n_arg))


def var_lengths_kernel[
    LL: TensorLayout, LO: TensorLayout, LI: TensorLayout
](
    lens: TileTensor[U32, LL, MutAnyOrigin],
    doff: TileTensor[U32, LO, MutAnyOrigin],
    idx: TileTensor[U32, LI, MutAnyOrigin],
    n_arg: Int32,
):
    """One thread per output value; the body is `var_lengths_body`.

    Parameters:
        LL: Layout of `lens`.
        LO: Layout of `doff`.
        LI: Layout of `idx`.

    Args:
        lens: Output lengths.
        doff: The dictionary's byte offsets.
        idx: Decoded dictionary indices.
        n_arg: How many values.
    """
    var_lengths_body[TARGET_GPU](lens, doff, idx, Int(n_arg))


def var_copy_kernel[
    LD: TensorLayout,
    LS: TensorLayout,
    LO: TensorLayout,
    LI: TensorLayout,
    LP: TensorLayout,
](
    dst: TileTensor[U8, LD, MutAnyOrigin],
    src: TileTensor[U8, LS, MutAnyOrigin],
    doff: TileTensor[U32, LO, MutAnyOrigin],
    idx: TileTensor[U32, LI, MutAnyOrigin],
    incl: TileTensor[U32, LP, MutAnyOrigin],
    n_arg: Int32,
):
    """One thread per output value; the body is `var_copy_body`.

    Parameters:
        LD: Layout of `dst`.
        LS: Layout of `src`.
        LO: Layout of `doff`.
        LI: Layout of `idx`.
        LP: Layout of `incl`.

    Args:
        dst: Output bytes.
        src: Dictionary bytes.
        doff: The dictionary's byte offsets.
        idx: Decoded dictionary indices.
        incl: Inclusive prefix sum of the lengths.
        n_arg: How many values.
    """
    var_copy_body[TARGET_GPU](dst, src, doff, idx, incl, Int(n_arg))


# ── GPU-only: the bounds check and the prefix sum ───────────────────────────
#
# Neither has a shared body. The CPU bounds check raises on the first offending
# index; a kernel cannot raise, so this one max-reduces per block and the host
# decides. The CPU prefix sum is `total += ln`; this one is a two-level block
# scan. These are the parts of the answer to question 1 that say "no".


def index_max_kernel[
    LI: TensorLayout, LM: TensorLayout
](
    idx: TileTensor[U32, LI, MutAnyOrigin],
    bmax: TileTensor[U32, LM, MutAnyOrigin],
    n_arg: Int32,
):
    """Per-block maximum of the dictionary indices, for the bounds check.

    The host reduces `bmax` (one entry per block) and, only if the maximum is
    out of range, scans the indices itself to find the *first* offender — so
    the error it raises is character-for-character the one `gather_into`
    raises. `_check_dict_block` in `parquet.encoding` takes the same
    max-then-locate shape for the same reason.

    Parameters:
        LI: Layout of `idx`.
        LM: Layout of `bmax`.

    Args:
        idx: Decoded dictionary indices.
        bmax: One `UInt32` per block, receiving that block's maximum.
        n_arg: How many indices.
    """
    comptime assert bmax.flat_rank == 1, "gather buffers are flat"
    var n = Int(n_arg)
    var sh = stack_allocation[BLOCK, U32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var i = Int(block_idx.x) * BLOCK + t
    sh.unsafe_store(t, rebind[Scalar[U32]](idx[i]) if i < n else UInt32(0))
    barrier()
    var off = BLOCK // 2
    while off > 0:
        if t < off:
            var a = sh.unsafe_load(t)
            var b = sh.unsafe_load(t + off)
            sh.unsafe_store(t, b if b > a else a)
        barrier()
        off //= 2
    if t == 0:
        bmax[Int(block_idx.x)] = rebind[bmax.ElementType](sh.unsafe_load(0))


def scan_block_kernel[
    LV: TensorLayout, LP: TensorLayout, LB: TensorLayout
](
    vals: TileTensor[U32, LV, MutAnyOrigin],
    part: TileTensor[U32, LP, MutAnyOrigin],
    bsums: TileTensor[U32, LB, MutAnyOrigin],
    n_arg: Int32,
):
    """Hillis–Steele inclusive scan within each block of `SCAN_BLOCK` values.

    Parameters:
        LV: Layout of `vals`.
        LP: Layout of `part`.
        LB: Layout of `bsums`.

    Args:
        vals: The per-value byte lengths.
        part: Receives the block-local inclusive scan.
        bsums: Receives each block's total.
        n_arg: How many values.
    """
    comptime assert part.flat_rank == 1, "gather buffers are flat"
    comptime assert bsums.flat_rank == 1, "gather buffers are flat"
    var n = Int(n_arg)
    var sh = stack_allocation[
        SCAN_BLOCK, U32, address_space=AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    var i = Int(block_idx.x) * SCAN_BLOCK + t
    sh.unsafe_store(t, rebind[Scalar[U32]](vals[i]) if i < n else UInt32(0))
    barrier()
    var off = 1
    while off < SCAN_BLOCK:
        var add = UInt32(0)
        if t >= off:
            add = sh.unsafe_load(t - off)
        barrier()
        if t >= off:
            sh.unsafe_store(t, sh.unsafe_load(t) + add)
        barrier()
        off *= 2
    if i < n:
        part[i] = rebind[part.ElementType](sh.unsafe_load(t))
    if t == SCAN_BLOCK - 1:
        bsums[Int(block_idx.x)] = rebind[bsums.ElementType](sh.unsafe_load(t))


def scan_sums_kernel[
    LB: TensorLayout
](bsums: TileTensor[U32, LB, MutAnyOrigin], nb_arg: Int32):
    """Turn the per-block totals into per-block *exclusive* offsets, in place.

    One block of `SCAN_SUMS` threads, so `nb` must be at most `SCAN_SUMS`.

    Parameters:
        LB: Layout of `bsums`.

    Args:
        bsums: Per-block totals in, per-block exclusive offsets out.
        nb_arg: How many blocks.
    """
    comptime assert bsums.flat_rank == 1, "gather buffers are flat"
    var nb = Int(nb_arg)
    var sh = stack_allocation[
        SCAN_SUMS, U32, address_space=AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    var mine = rebind[Scalar[U32]](bsums[t]) if t < nb else UInt32(0)
    sh.unsafe_store(t, mine)
    barrier()
    var off = 1
    while off < SCAN_SUMS:
        var add = UInt32(0)
        if t >= off:
            add = sh.unsafe_load(t - off)
        barrier()
        if t >= off:
            sh.unsafe_store(t, sh.unsafe_load(t) + add)
        barrier()
        off *= 2
    # inclusive -> exclusive
    var excl = sh.unsafe_load(t) - mine
    barrier()
    if t < nb:
        bsums[t] = rebind[bsums.ElementType](excl)


def scan_add_kernel[
    LP: TensorLayout, LB: TensorLayout
](
    part: TileTensor[U32, LP, MutAnyOrigin],
    bofs: TileTensor[U32, LB, MutAnyOrigin],
    base_arg: Int32,
    n_arg: Int32,
):
    """Add each block's offset (and the output's existing byte count) to the
    block-local scan, making `part` the whole page's inclusive prefix sum.

    Parameters:
        LP: Layout of `part`.
        LB: Layout of `bofs`.

    Args:
        part: Block-local inclusive scan in, global inclusive scan out.
        bofs: Per-block exclusive offsets.
        base_arg: Bytes already in the output buffer.
        n_arg: How many values.
    """
    comptime assert part.flat_rank == 1, "gather buffers are flat"
    var i = Int(global_idx.x)
    if i >= Int(n_arg):
        return
    part[i] = rebind[part.ElementType](
        rebind[Scalar[U32]](part[i])
        + rebind[Scalar[U32]](bofs[i // SCAN_BLOCK])
        + UInt32(base_arg)
    )
