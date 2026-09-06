"""Where the GPU path's time actually goes, and whether unified memory helps.

Apple Silicon has unified memory, so the host-to-device transfer that sinks
discrete-GPU Parquet might not apply. Three probes settle it.

**Can a kernel read host memory directly?** `probe_zero_copy` fills a buffer
from `enqueue_create_host_buffer` and hands it straight to a kernel, then does
the same with a device buffer as a control. If the host buffer works, unified
memory removes the copy; if it does not, every value has to be staged.

**What do the host-side primitives cost?** `probe_primitives` times
`map_to_host`, `synchronize` and `enqueue_create_buffer` in batches, at several
buffer sizes. These are the per-call constants the benchmark's fixed cost is
made of.

**And what does the first dispatch cost?** `probe_dispatch` times a cold first
dispatch, a warm one, and the same pair through `kernel_cache.cached_enqueue`,
which is `millfolio/engine`'s answer to `enqueue_function` recompiling on every
call.

`DeviceContext()` creation is timed too, because a per-call context would
dominate everything else.
"""

from std.math import ceildiv
from std.memory import unsafe_memcpy
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major

from gather_shared import BLOCK, U32, U8, gather_fixed_kernel
from gpu_gather import upload_u32
from kernel_cache import cached_enqueue

comptime REPS = 200
"""Iterations inside one timed batch — `perf_counter_ns` ticks at about a
microsecond here, so a single call is below the clock's resolution."""


def sizes() -> List[Int]:
    """Buffer sizes to probe, 1 KiB to 64 MiB.

    Returns:
        The sizes, ascending.
    """
    return [1024, 65536, 1048576, 16777216, 67108864]


def bump_kernel[
    LA: TensorLayout, LB: TensorLayout
](
    a: TileTensor[U32, LA, MutAnyOrigin],
    b: TileTensor[U32, LB, MutAnyOrigin],
    n_arg: Int32,
):
    """`b[i] = a[i] + 1` — the smallest kernel that proves what it read.

    Parameters:
        LA: Layout of `a`.
        LB: Layout of `b`.

    Args:
        a: Input.
        b: Output.
        n_arg: How many elements.
    """
    comptime assert b.flat_rank == 1, "flat buffers"
    var i = Int(global_idx.x)
    if i < Int(n_arg):
        b[i] = rebind[b.ElementType](rebind[Scalar[U32]](a[i]) + 1)


def probe_zero_copy(ctx: DeviceContext) raises:
    """Can a kernel read and write a host-allocated buffer?

    Args:
        ctx: The device context.

    Raises:
        If a device operation fails.
    """
    var n = 8
    var lay = row_major(n)
    comptime k = bump_kernel[type_of(lay), type_of(lay)]

    var hin = ctx.enqueue_create_host_buffer[U32](n)
    var hout = ctx.enqueue_create_host_buffer[U32](n)
    ctx.synchronize()
    for i in range(n):
        hin.unsafe_ptr().unsafe_store(i, UInt32(100 + i))
        hout.unsafe_ptr().unsafe_store(i, UInt32(0))

    print("can a kernel touch host memory directly?")
    print("   input is 100..107, so a correct run prints 101..108")

    ctx.enqueue_function[k](
        TileTensor(hin, lay),
        TileTensor(hout, lay),
        Int32(n),
        grid_dim=1,
        block_dim=BLOCK,
    )
    ctx.synchronize()
    var a = String("   host buffer in, host buffer out:     ")
    for i in range(n):
        a += String(hout.unsafe_ptr().unsafe_load(i), " ")
    print(a)

    var dout = ctx.enqueue_create_buffer[U32](n)
    ctx.enqueue_function[k](
        TileTensor(hin, lay),
        TileTensor(dout, lay),
        Int32(n),
        grid_dim=1,
        block_dim=BLOCK,
    )
    ctx.synchronize()
    var b = String("   host buffer in, device buffer out:   ")
    with dout.map_to_host() as m:
        for i in range(n):
            b += String(m.unsafe_ptr().unsafe_load(i), " ")
    print(b)

    var din = ctx.enqueue_create_buffer[U32](n)
    with din.map_to_host() as m:
        for i in range(n):
            m.unsafe_ptr().unsafe_store(i, UInt32(100 + i))
    var dout2 = ctx.enqueue_create_buffer[U32](n)
    ctx.enqueue_function[k](
        TileTensor(din, lay),
        TileTensor(dout2, lay),
        Int32(n),
        grid_dim=1,
        block_dim=BLOCK,
    )
    ctx.synchronize()
    var c = String("   device in, device out (the control): ")
    with dout2.map_to_host() as m:
        for i in range(n):
            c += String(m.unsafe_ptr().unsafe_load(i), " ")
    print(c)
    _ = hin^
    _ = hout^
    _ = dout^
    _ = din^
    _ = dout2^


def probe_primitives(ctx: DeviceContext) raises:
    """Batched cost of `map_to_host`, `synchronize` and buffer creation.

    Args:
        ctx: The device context.

    Raises:
        If a device operation fails.
    """
    print()
    print("host-side primitives, ns per call (batches of", REPS, "):")
    print("       bytes    map_to_host    synchronize    create_buffer")
    var all = sizes()
    for si in range(len(all)):
        var n = all[si]
        var buf = ctx.enqueue_create_buffer[U8](n)
        ctx.synchronize()
        for _ in range(10):
            with buf.map_to_host() as m:
                _ = m.unsafe_ptr()

        var t0 = perf_counter_ns()
        for _ in range(REPS):
            with buf.map_to_host() as m:
                _ = m.unsafe_ptr()
        var t1 = perf_counter_ns()

        var s0 = perf_counter_ns()
        for _ in range(REPS):
            ctx.synchronize()
        var s1 = perf_counter_ns()

        var c0 = perf_counter_ns()
        for _ in range(REPS):
            var b = ctx.enqueue_create_buffer[U8](n)
            _ = b^
        var c1 = perf_counter_ns()

        print(
            "   ",
            n,
            "   ",
            Int(t1 - t0) // REPS,
            "   ",
            Int(s1 - s0) // REPS,
            "   ",
            Int(c1 - c0) // REPS,
            sep="   ",
        )
        _ = buf^


def probe_copy_bandwidth(ctx: DeviceContext) raises:
    """`map_to_host` + `memcpy` against a plain host `memcpy` of the same bytes.

    Args:
        ctx: The device context.

    Raises:
        If a device operation fails.
    """
    print()
    print("moving N bytes, ns (p50 of 9):")
    print("       bytes    host memcpy    map+memcpy    hostbuf+enqueue_copy")
    var all = sizes()
    for si in range(len(all)):
        var n = all[si]
        var src = List[UInt8](length=n, fill=7)
        var sink = List[UInt8](length=n, fill=0)
        var dev = ctx.enqueue_create_buffer[U8](n)
        var host = ctx.enqueue_create_host_buffer[U8](n)
        ctx.synchronize()

        var hh = List[Int]()
        var mm = List[Int]()
        var ec = List[Int]()
        for _ in range(9):
            var a0 = perf_counter_ns()
            unsafe_memcpy(dest=sink.unsafe_ptr(), src=src.unsafe_ptr(), count=n)
            var a1 = perf_counter_ns()
            hh.append(Int(a1 - a0))

            var b0 = perf_counter_ns()
            with dev.map_to_host() as m:
                unsafe_memcpy(
                    dest=m.unsafe_ptr(), src=src.unsafe_ptr(), count=n
                )
            var b1 = perf_counter_ns()
            mm.append(Int(b1 - b0))

            var c0 = perf_counter_ns()
            unsafe_memcpy(dest=host.unsafe_ptr(), src=src.unsafe_ptr(), count=n)
            ctx.enqueue_copy(dev, host)
            ctx.synchronize()
            var c1 = perf_counter_ns()
            ec.append(Int(c1 - c0))

        print(
            "   ",
            n,
            "   ",
            _median(hh^),
            "   ",
            _median(mm^),
            "   ",
            _median(ec^),
            sep="   ",
        )
        _ = src^
        _ = sink^
        _ = dev^
        _ = host^


def _median(var samples: List[Int]) -> Int:
    """The median of `samples`.

    Args:
        samples: The measurements.

    Returns:
        The median.
    """
    var s = samples^
    for i in range(1, len(s)):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    return s[len(s) // 2]


def probe_dispatch(ctx: DeviceContext) raises:
    """Cold versus warm dispatch, plain and through the kernel cache.

    Args:
        ctx: The device context.

    Raises:
        If a device operation fails.
    """
    var n = 65536
    var w = 8
    var idx = List[UInt32](length=n, fill=0)
    for i in range(n):
        idx[i] = UInt32(i % 512)
    var src = ctx.enqueue_create_buffer[U8](512 * w)
    var dst = ctx.enqueue_create_buffer[U8](n * w)
    var idx_dev = upload_u32(ctx, Span(idx))
    ctx.synchronize()
    var ld = row_major(n * w)
    var ls = row_major(512 * w)
    var li = row_major(n)
    comptime k = gather_fixed_kernel[type_of(ld), type_of(ls), type_of(li)]

    print()
    print("dispatch of one 65536-value gather, ns:")

    var t0 = perf_counter_ns()
    ctx.enqueue_function[k](
        TileTensor(dst, ld),
        TileTensor(src, ls),
        TileTensor(idx_dev, li),
        Int32(w),
        Int32(n),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("   enqueue_function, first call:  ", Int(t1 - t0))

    var warm = List[Int]()
    for _ in range(21):
        var a = perf_counter_ns()
        ctx.enqueue_function[k](
            TileTensor(dst, ld),
            TileTensor(src, ls),
            TileTensor(idx_dev, li),
            Int32(w),
            Int32(n),
            grid_dim=ceildiv(n, BLOCK),
            block_dim=BLOCK,
        )
        ctx.synchronize()
        var b = perf_counter_ns()
        warm.append(Int(b - a))
    print("   enqueue_function, later calls: ", _median(warm^))

    var c0 = perf_counter_ns()
    cached_enqueue[k](
        ctx,
        TileTensor(dst, ld),
        TileTensor(src, ls),
        TileTensor(idx_dev, li),
        Int32(w),
        Int32(n),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )
    ctx.synchronize()
    var c1 = perf_counter_ns()
    print("   cached_enqueue, first call:    ", Int(c1 - c0))

    var cw = List[Int]()
    for _ in range(21):
        var a = perf_counter_ns()
        cached_enqueue[k](
            ctx,
            TileTensor(dst, ld),
            TileTensor(src, ls),
            TileTensor(idx_dev, li),
            Int32(w),
            Int32(n),
            grid_dim=ceildiv(n, BLOCK),
            block_dim=BLOCK,
        )
        ctx.synchronize()
        var b = perf_counter_ns()
        cw.append(Int(b - a))
    print("   cached_enqueue, later calls:   ", _median(cw^))
    _ = src^
    _ = dst^
    _ = idx_dev^


def main() raises:
    """Run every probe and print the numbers.

    Raises:
        If there is no accelerator, or a device operation fails.
    """
    if not has_accelerator():
        raise Error("experiments.gpu_gather: no GPU — this probe needs Metal")

    var ctxs = List[Int]()
    for _ in range(5):
        var t0 = perf_counter_ns()
        var c = DeviceContext()
        var t1 = perf_counter_ns()
        ctxs.append(Int(t1 - t0))
        _ = c^
    print("DeviceContext() creation, p50 ns:", _median(ctxs^))

    var ctx = DeviceContext()
    print("device:", ctx.name(), "/", ctx.api())
    print()
    probe_zero_copy(ctx)
    probe_primitives(ctx)
    probe_copy_bandwidth(ctx)
    probe_dispatch(ctx)
