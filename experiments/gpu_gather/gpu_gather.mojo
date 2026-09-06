"""The host side of the GPU dictionary gather: buffers, launches, download.

`gpu_gather_into` has the same signature and the same contract as
`parquet.encoding.gather_into` — same `PhysBuffer` in, same `PhysBuffer` out,
same error text for an out-of-range index — so the parity gate can hand both
the same real dictionary page and compare the results byte for byte.

Everything here is host-side, and none of it is shared with the CPU path:
allocation, staging, the bounds check a kernel cannot raise from, the prefix
sum's three launches, and the download. `gather_shared.mojo` holds the part
that *is* shared.

**Transfers go through `enqueue_create_host_buffer` + `enqueue_copy`, not
`map_to_host`.** On this machine `map_to_host` costs about 550 microseconds per
call whatever the size, and it carries its own device synchronization, so a
gather written with a map to upload and a map to download starts at 1.3
milliseconds before it has done anything. Staged copies are enqueued and cost
almost nothing on their own; one `ctx.synchronize()` at the end covers the
whole page. `pixi run -e gpu gpu-memory` has the numbers. The host staging
buffers are deliberately kept alive in the same scope as that synchronize —
an `enqueue_copy` reads them after the call returns.

The fixed-width path therefore needs exactly **one** synchronize per page. The
`BYTE_ARRAY` path needs **two**: the offset scan has to come back to the host
before the output buffer can be sized, and only then can the byte copy launch.
That is a property of the operation, not of this code.

Two entry points, because the honest number depends on which you mean:

* `gpu_gather_into` uploads the dictionary on every call — one page in
  isolation, every overhead included.
* `DeviceDict` + `gpu_gather_into_resident` uploads the dictionary once and
  gathers many pages against it, which is what a real column chunk does.
"""

from std.math import ceildiv
from std.memory import unsafe_memcpy
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from layout import TileTensor, row_major

from parquet.encoding import PK_BOOL, PK_VAR, PhysBuffer

from gather_shared import (
    BLOCK,
    SCAN_BLOCK,
    SCAN_SUMS,
    U32,
    U8,
    gather_fixed_kernel,
    gpu_grid,
    index_max_kernel,
    scan_add_kernel,
    scan_block_kernel,
    scan_sums_kernel,
    var_copy_kernel,
    var_lengths_kernel,
)
from kernel_cache import cached_enqueue

comptime U8Buf = DeviceBuffer[U8]
comptime U32Buf = DeviceBuffer[U32]
comptime U8Host = HostBuffer[U8]
comptime U32Host = HostBuffer[U32]

comptime MAX_VAR_BYTES = (1 << 31) - 1
"""Matches `parquet.encoding._MAX_VAR_BYTES`: the largest byte count an Arrow
32-bit offset can address."""

comptime MAX_SCAN = SCAN_BLOCK * SCAN_SUMS
"""Largest page the two-level offset scan handles in one call."""


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


def _var_bytes_overflow(total: Int) -> Error:
    """The 2 GiB overflow error, worded as `parquet.encoding` words it.

    Args:
        total: The byte count that overflowed.

    Returns:
        The error to raise.
    """
    return Error(
        String(
            "parquet.encoding: column chunk holds ",
            total,
            (
                " bytes of BYTE_ARRAY data, past the 2 GiB an Arrow 32-bit"
                " offset can address — this file needs 64-bit offsets"
                " (large_binary) or the column split across record batches,"
                " neither of which parquet.mojo supports yet"
            ),
        )
    )


struct Staged8(Movable):
    """A staged upload of bytes: the host copy and the device buffer."""

    var host: U8Host
    var dev: U8Buf

    def __init__(out self, var host: U8Host, var dev: U8Buf):
        """Hold both halves of one staged upload.

        Args:
            host: The staging buffer the copy reads.
            dev: The device buffer the copy writes.
        """
        self.host = host^
        self.dev = dev^

    def __init__(out self, *, deinit move: Self):
        """Move.

        Args:
            move: The value being moved from.
        """
        self.host = move.host^
        self.dev = move.dev^

    def take(deinit self) -> U8Buf:
        """Consume this upload and keep only the device buffer.

        Only call it after the `synchronize` that retires the copy — the
        staging buffer is dropped here.

        Returns:
            The device buffer.
        """
        return self.dev^


struct Staged32(Movable):
    """A staged upload of words: the host copy and the device buffer."""

    var host: U32Host
    var dev: U32Buf

    def __init__(out self, var host: U32Host, var dev: U32Buf):
        """Hold both halves of one staged upload.

        Args:
            host: The staging buffer the copy reads.
            dev: The device buffer the copy writes.
        """
        self.host = host^
        self.dev = dev^

    def __init__(out self, *, deinit move: Self):
        """Move.

        Args:
            move: The value being moved from.
        """
        self.host = move.host^
        self.dev = move.dev^

    def take(deinit self) -> U32Buf:
        """Consume this upload and keep only the device buffer.

        Only call it after the `synchronize` that retires the copy — the
        staging buffer is dropped here.

        Returns:
            The device buffer.
        """
        return self.dev^


def stage_u8(ctx: DeviceContext, host: Span[UInt8, _]) raises -> Staged8:
    """Copy `host` into a staging buffer and enqueue the upload.

    The returned host buffer must outlive the `synchronize` that retires the
    copy; that is why it comes back rather than being dropped here.

    Args:
        ctx: The device context.
        host: The bytes to upload.

    Returns:
        The staged upload.
    """
    var n = len(host)
    var size = n if n > 0 else 1
    var h = ctx.enqueue_create_host_buffer[U8](size)
    var d = ctx.enqueue_create_buffer[U8](size)
    if n > 0:
        unsafe_memcpy(dest=h.unsafe_ptr(), src=host.unsafe_ptr(), count=n)
    ctx.enqueue_copy(d, h)
    return Staged8(h^, d^)


def stage_u32(ctx: DeviceContext, host: Span[UInt32, _]) raises -> Staged32:
    """Copy `host` into a staging buffer and enqueue the upload.

    Args:
        ctx: The device context.
        host: The words to upload.

    Returns:
        The staged upload.
    """
    var n = len(host)
    var size = n if n > 0 else 1
    var h = ctx.enqueue_create_host_buffer[U32](size)
    var d = ctx.enqueue_create_buffer[U32](size)
    if n > 0:
        unsafe_memcpy(dest=h.unsafe_ptr(), src=host.unsafe_ptr(), count=n)
    ctx.enqueue_copy(d, h)
    return Staged32(h^, d^)


def upload_u32(ctx: DeviceContext, host: Span[UInt32, _]) raises -> U32Buf:
    """Upload `host` and wait for it — for callers outside a batched page.

    Args:
        ctx: The device context.
        host: The words to upload.

    Returns:
        The device buffer.
    """
    var staged = stage_u32(ctx, host)
    ctx.synchronize()
    return staged^.take()


def _offsets_as_u32(dict: PhysBuffer) -> List[UInt32]:
    """The dictionary's `Int32` offsets widened to `UInt32` for the kernels.

    Parquet offsets are non-negative by construction, and `decode_plain`
    already rejects a byte array longer than 2 GiB, so the reinterpretation is
    lossless.

    Args:
        dict: The dictionary page.

    Returns:
        The offsets as `UInt32`.
    """
    var out = List[UInt32](length=len(dict.offsets), fill=0)
    var p = out.unsafe_ptr()
    var q = dict.offsets.unsafe_ptr()
    for i in range(len(dict.offsets)):
        p.unsafe_store(i, UInt32(Int(q.unsafe_load(i))))
    return out^


struct DeviceDict(Movable):
    """A dictionary page resident on the device, gathered against many times.

    A column chunk has one dictionary and many data pages; uploading it once is
    what the real reader would do, so the benchmark reports both this and the
    upload-every-time number.
    """

    var bytes: U8Buf
    var offsets: U32Buf
    var kind: Int
    var width: Int
    var count: Int
    var nbytes: Int
    """How many dictionary bytes are on the device — the kernels need it to
    bind a layout over `bytes`."""

    def __init__(out self, ctx: DeviceContext, dict: PhysBuffer) raises:
        """Upload `dict`.

        Args:
            ctx: The device context.
            dict: The dictionary page to upload.
        """
        if dict.kind == PK_BOOL:
            raise Error(
                "experiments.gpu_gather: a BOOLEAN dictionary keeps the CPU"
                " path — see the README"
            )
        var sb = stage_u8(ctx, Span(dict.bytes))
        var off = _offsets_as_u32(dict)
        var so = stage_u32(ctx, Span(off))
        ctx.synchronize()
        self.bytes = sb^.take()
        self.offsets = so^.take()
        self.kind = dict.kind
        self.width = dict.width
        self.count = dict.count
        self.nbytes = len(dict.bytes)

    def __init__(out self, *, deinit move: Self):
        """Move.

        Args:
            move: The value being moved from.
        """
        self.bytes = move.bytes^
        self.offsets = move.offsets^
        self.kind = move.kind
        self.width = move.width
        self.count = move.count
        self.nbytes = move.nbytes


def _host_bounds_check(indices: Span[UInt32, _], entries: Int) raises:
    """The bounds check on the calling thread.

    Offered because the kernel version costs a launch and a staged copy back to
    save a scan the CPU does in microseconds. The benchmark reports both, so
    the verdict is not resting on a straw-man GPU path.

    Args:
        indices: The decoded dictionary indices.
        entries: How many entries the dictionary has.

    Raises:
        If any index is at or past `entries`.
    """
    var p = indices.unsafe_ptr()
    for i in range(len(indices)):
        if Int(p.unsafe_load(i)) >= entries:
            raise _out_of_range(p.unsafe_load(i), entries)


def _launch_index_max(
    ctx: DeviceContext,
    mut idx_dev: U32Buf,
    mut bmax: U32Buf,
    n: Int,
    nb: Int,
    cached: Bool,
) raises:
    """Enqueue the per-block maximum of the dictionary indices.

    Deliberately launched with the true grid rather than `gpu_grid`: the
    negative control breaks the gather, not the bounds check, so a sabotaged
    run fails on wrong values and not on an unwritten `bmax` slot.

    Args:
        ctx: The device context.
        idx_dev: The indices, on the device.
        bmax: One `UInt32` per block, receiving that block's maximum.
        n: How many indices.
        nb: How many blocks.
        cached: Whether to dispatch through the compiled-kernel cache.

    Raises:
        If the launch fails.
    """
    var li = row_major(n)
    var lm = row_major(nb)
    comptime k = index_max_kernel[type_of(li), type_of(lm)]
    if cached:
        cached_enqueue[k](
            ctx,
            TileTensor(idx_dev, li),
            TileTensor(bmax, lm),
            Int32(n),
            grid_dim=nb,
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[k](
            TileTensor(idx_dev, li),
            TileTensor(bmax, lm),
            Int32(n),
            grid_dim=nb,
            block_dim=BLOCK,
        )


def _settle_bounds(
    hmax: U32Host, nb: Int, indices: Span[UInt32, _], entries: Int
) raises:
    """Reduce the per-block maxima and raise `gather_into`'s exact error.

    Only when the maximum is out of range are the indices scanned for the first
    offender, so the happy path is `nb` comparisons and a corrupt page raises
    the same message, naming the same index, as the CPU.

    Args:
        hmax: The per-block maxima, copied back.
        nb: How many blocks there were.
        indices: The indices on the host, for locating the offender.
        entries: How many entries the dictionary has.

    Raises:
        If any index is at or past `entries`.
    """
    var top = UInt32(0)
    var p = hmax.unsafe_ptr()
    for b in range(nb):
        var v = p.unsafe_load(b)
        if v > top:
            top = v
    if Int(top) < entries:
        return
    var q = indices.unsafe_ptr()
    for i in range(len(indices)):
        if Int(q.unsafe_load(i)) >= entries:
            raise _out_of_range(q.unsafe_load(i), entries)
    # The reduction said there was an offender and the scan disagrees: that is
    # a wrong kernel, not a corrupt file, and it must not pass silently.
    raise Error(
        String(
            "experiments.gpu_gather: index_max_kernel reported a maximum of ",
            top,
            " but no host index is at or past ",
            entries,
        )
    )


def _gather_fixed[
    sabotage: Bool
](
    ctx: DeviceContext,
    mut out: PhysBuffer,
    ddict: DeviceDict,
    indices: List[UInt32],
    cached: Bool,
    host_check: Bool,
) raises:
    """The fixed-width gather: everything enqueued, then one synchronize.

    Parameters:
        sabotage: Run the negative control's deliberately broken launch.

    Args:
        ctx: The device context.
        out: The output buffer, appended to.
        ddict: The dictionary, resident on the device.
        indices: The decoded dictionary indices.
        cached: Whether to dispatch through the compiled-kernel cache.
        host_check: Bounds-check on the calling thread instead of with a
            kernel.

    Raises:
        If `out` and the dictionary disagree on entry width, or an index is out
        of range.
    """
    var n = len(indices)
    var w = ddict.width
    if out.count == 0 and len(out.bytes) == 0:
        out.width = w
    if out.width != w:
        raise Error("parquet.encoding: cannot concatenate unlike pages")
    var nbytes = n * w
    var nb = ceildiv(n, BLOCK)

    var sidx = stage_u32(ctx, Span(indices))
    ref idx_dev = sidx.dev
    var bmax = ctx.enqueue_create_buffer[U32](nb)
    var hmax = ctx.enqueue_create_host_buffer[U32](nb)
    var dout = ctx.enqueue_create_buffer[U8](nbytes if nbytes > 0 else 1)
    var hout = ctx.enqueue_create_host_buffer[U8](nbytes if nbytes > 0 else 1)

    if not host_check:
        _launch_index_max(ctx, idx_dev, bmax, n, nb, cached)

    var ld = row_major(nbytes if nbytes > 0 else 1)
    var ls = row_major(ddict.nbytes if ddict.nbytes > 0 else 1)
    var li = row_major(n)
    comptime k = gather_fixed_kernel[type_of(ld), type_of(ls), type_of(li)]
    if cached:
        cached_enqueue[k](
            ctx,
            TileTensor(dout, ld),
            TileTensor(ddict.bytes, ls),
            TileTensor(idx_dev, li),
            Int32(w),
            Int32(n),
            grid_dim=gpu_grid[sabotage](n),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[k](
            TileTensor(dout, ld),
            TileTensor(ddict.bytes, ls),
            TileTensor(idx_dev, li),
            Int32(w),
            Int32(n),
            grid_dim=gpu_grid[sabotage](n),
            block_dim=BLOCK,
        )

    if not host_check:
        ctx.enqueue_copy(hmax, bmax)
    ctx.enqueue_copy(hout, dout)
    ctx.synchronize()

    if host_check:
        _host_bounds_check(Span(indices), ddict.count)
    else:
        _settle_bounds(hmax, nb, Span(indices), ddict.count)

    var vbase = len(out.bytes)
    out.bytes.resize(vbase + nbytes, 0)
    if nbytes > 0:
        unsafe_memcpy(
            dest=out.bytes.unsafe_ptr().unsafe_offset(vbase),
            src=hout.unsafe_ptr(),
            count=nbytes,
        )
    out.count += n
    _ = sidx^
    _ = hmax^
    _ = hout^


def _gather_var[
    sabotage: Bool
](
    ctx: DeviceContext,
    mut out: PhysBuffer,
    ddict: DeviceDict,
    indices: List[UInt32],
    cached: Bool,
    host_check: Bool,
) raises:
    """The `BYTE_ARRAY` gather: lengths, a two-level scan, then the byte copy.

    Two synchronizes, not one: the scan's total decides how many bytes the
    output needs, and that number has to reach the host before the byte-copy
    kernel can be given somewhere to write.

    Parameters:
        sabotage: Run the negative control's deliberately broken launch.

    Args:
        ctx: The device context.
        out: The output buffer, appended to.
        ddict: The dictionary, resident on the device.
        indices: The decoded dictionary indices.
        cached: Whether to dispatch through the compiled-kernel cache.
        host_check: Bounds-check on the calling thread instead of with a
            kernel.

    Raises:
        If the page is larger than the scan handles, an index is out of range,
        or the output passes 2 GiB.
    """
    var n = len(indices)
    if n > MAX_SCAN:
        raise Error(
            String(
                "experiments.gpu_gather: page of ",
                n,
                " values is past the ",
                MAX_SCAN,
                (
                    " the two-level offset scan handles in one call — see the"
                    " README"
                ),
            )
        )
    var nb = ceildiv(n, BLOCK)
    var nsb = ceildiv(n, SCAN_BLOCK)

    var sidx = stage_u32(ctx, Span(indices))
    ref idx_dev = sidx.dev
    var lens = ctx.enqueue_create_buffer[U32](n)
    var part = ctx.enqueue_create_buffer[U32](n)
    var bsums = ctx.enqueue_create_buffer[U32](nsb)
    var bmax = ctx.enqueue_create_buffer[U32](nb)
    var hmax = ctx.enqueue_create_host_buffer[U32](nb)
    var hpart = ctx.enqueue_create_host_buffer[U32](n)

    var ln = row_major(n)
    var lo = row_major(ddict.count + 1)
    var li = row_major(n)
    var lb = row_major(nsb)
    comptime k1 = var_lengths_kernel[type_of(ln), type_of(lo), type_of(li)]
    comptime k2 = scan_block_kernel[type_of(ln), type_of(ln), type_of(lb)]
    comptime k3 = scan_sums_kernel[type_of(lb)]
    comptime k4 = scan_add_kernel[type_of(ln), type_of(lb)]

    if not host_check:
        _launch_index_max(ctx, idx_dev, bmax, n, nb, cached)

    if cached:
        cached_enqueue[k1](
            ctx,
            TileTensor(lens, ln),
            TileTensor(ddict.offsets, lo),
            TileTensor(idx_dev, li),
            Int32(n),
            grid_dim=gpu_grid[sabotage](n),
            block_dim=BLOCK,
        )
        cached_enqueue[k2](
            ctx,
            TileTensor(lens, ln),
            TileTensor(part, ln),
            TileTensor(bsums, lb),
            Int32(n),
            grid_dim=nsb,
            block_dim=SCAN_BLOCK,
        )
        cached_enqueue[k3](
            ctx,
            TileTensor(bsums, lb),
            Int32(nsb),
            grid_dim=1,
            block_dim=SCAN_SUMS,
        )
        cached_enqueue[k4](
            ctx,
            TileTensor(part, ln),
            TileTensor(bsums, lb),
            Int32(0),
            Int32(n),
            grid_dim=gpu_grid[sabotage](n),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[k1](
            TileTensor(lens, ln),
            TileTensor(ddict.offsets, lo),
            TileTensor(idx_dev, li),
            Int32(n),
            grid_dim=gpu_grid[sabotage](n),
            block_dim=BLOCK,
        )
        ctx.enqueue_function[k2](
            TileTensor(lens, ln),
            TileTensor(part, ln),
            TileTensor(bsums, lb),
            Int32(n),
            grid_dim=nsb,
            block_dim=SCAN_BLOCK,
        )
        ctx.enqueue_function[k3](
            TileTensor(bsums, lb),
            Int32(nsb),
            grid_dim=1,
            block_dim=SCAN_SUMS,
        )
        ctx.enqueue_function[k4](
            TileTensor(part, ln),
            TileTensor(bsums, lb),
            Int32(0),
            Int32(n),
            grid_dim=gpu_grid[sabotage](n),
            block_dim=BLOCK,
        )

    if not host_check:
        ctx.enqueue_copy(hmax, bmax)
    ctx.enqueue_copy(hpart, part)
    ctx.synchronize()

    if host_check:
        _host_bounds_check(Span(indices), ddict.count)
    else:
        _settle_bounds(hmax, nb, Span(indices), ddict.count)

    # The scan's last entry is the page's byte total; the offsets go out with
    # the output's existing byte count folded in, as `gather_into` writes them.
    var vbase = len(out.bytes)
    var obase = len(out.offsets)
    out.offsets.resize(obase + n, 0)
    var hp = hpart.unsafe_ptr()
    var page_bytes = Int(hp.unsafe_load(n - 1))
    if vbase + page_bytes > MAX_VAR_BYTES:
        raise _var_bytes_overflow(vbase + page_bytes)
    var o = out.offsets.unsafe_ptr()
    for i in range(n):
        o.unsafe_store(obase + i, Int32(vbase + Int(hp.unsafe_load(i))))

    var size = page_bytes if page_bytes > 0 else 1
    var dout = ctx.enqueue_create_buffer[U8](size)
    var hout = ctx.enqueue_create_host_buffer[U8](size)
    var ld = row_major(size)
    var lsrc = row_major(ddict.nbytes if ddict.nbytes > 0 else 1)
    comptime k5 = var_copy_kernel[
        type_of(ld),
        type_of(lsrc),
        type_of(lo),
        type_of(li),
        type_of(ln),
    ]
    if cached:
        cached_enqueue[k5](
            ctx,
            TileTensor(dout, ld),
            TileTensor(ddict.bytes, lsrc),
            TileTensor(ddict.offsets, lo),
            TileTensor(idx_dev, li),
            TileTensor(part, ln),
            Int32(n),
            grid_dim=gpu_grid[sabotage](n),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[k5](
            TileTensor(dout, ld),
            TileTensor(ddict.bytes, lsrc),
            TileTensor(ddict.offsets, lo),
            TileTensor(idx_dev, li),
            TileTensor(part, ln),
            Int32(n),
            grid_dim=gpu_grid[sabotage](n),
            block_dim=BLOCK,
        )
    ctx.enqueue_copy(hout, dout)
    ctx.synchronize()

    out.bytes.resize(vbase + page_bytes, 0)
    if page_bytes > 0:
        unsafe_memcpy(
            dest=out.bytes.unsafe_ptr().unsafe_offset(vbase),
            src=hout.unsafe_ptr(),
            count=page_bytes,
        )
    out.count += n
    _ = sidx^
    _ = hmax^
    _ = hpart^
    _ = hout^


def gpu_gather_into_resident[
    sabotage: Bool = False
](
    ctx: DeviceContext,
    mut out: PhysBuffer,
    ddict: DeviceDict,
    indices: List[UInt32],
    cached: Bool = True,
    host_check: Bool = False,
) raises:
    """Gather `indices` out of an already-uploaded dictionary onto `out`.

    Parameters:
        sabotage: Run the negative control's deliberately broken launch.

    Args:
        ctx: The device context.
        out: The output buffer, appended to.
        ddict: The dictionary, resident on the device.
        indices: The decoded dictionary indices.
        cached: Whether to dispatch through the compiled-kernel cache.
        host_check: Bounds-check on the calling thread instead of with a
            kernel, saving a launch and a copy back.

    Raises:
        If the buffers disagree in shape, an index is out of range, or the page
        is larger than the offset scan handles.
    """
    var n = len(indices)
    if n == 0:
        return
    if out.kind != ddict.kind:
        raise Error("parquet.encoding: cannot concatenate unlike pages")
    if ddict.kind == PK_VAR:
        _gather_var[sabotage](ctx, out, ddict, indices, cached, host_check)
        return
    _gather_fixed[sabotage](ctx, out, ddict, indices, cached, host_check)


def gpu_gather_into[
    sabotage: Bool = False
](
    ctx: DeviceContext,
    mut out: PhysBuffer,
    dict: PhysBuffer,
    indices: List[UInt32],
    cached: Bool = True,
) raises:
    """`parquet.encoding.gather_into`, on the GPU, dictionary upload included.

    Parameters:
        sabotage: Run the negative control's deliberately broken launch.

    Args:
        ctx: The device context.
        out: The output buffer, appended to.
        dict: The dictionary page.
        indices: The decoded dictionary indices.
        cached: Whether to dispatch through the compiled-kernel cache.

    Raises:
        Whatever `gather_into` would raise, with the same message.
    """
    if len(indices) == 0:
        return
    if out.kind != dict.kind:
        raise Error("parquet.encoding: cannot concatenate unlike pages")
    var ddict = DeviceDict(ctx, dict)
    gpu_gather_into_resident[sabotage](ctx, out, ddict, indices, cached)


def gpu_gather(
    ctx: DeviceContext, dict: PhysBuffer, indices: List[UInt32]
) raises -> PhysBuffer:
    """`parquet.encoding.gather`, on the GPU.

    Args:
        ctx: The device context.
        dict: The dictionary page.
        indices: The decoded dictionary indices.

    Returns:
        A fresh buffer holding the gathered values.
    """
    var out = PhysBuffer(dict.kind, dict.width)
    gpu_gather_into(ctx, out, dict, indices)
    return out^
