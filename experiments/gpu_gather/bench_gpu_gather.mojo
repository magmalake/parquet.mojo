"""CPU versus GPU for the dictionary gather, with every overhead included.

Run the parity gate first. A number from a wrong kernel is worse than no
number.

Built ahead of time (`pixi run -e gpu gpu-bench`), never `mojo run` — a JIT
configuration is not comparable to an AOT one, and this file is only ever
compared against the AOT CPU path in the same binary.

**What is timed.** Three implementations of the same operation, over the same
real dictionary pages:

* `gather_into` — `parquet.encoding`'s CPU oracle, taking an already-decoded
  `List[UInt32]` of indices. The apples-to-apples baseline, because it is what
  the GPU path is handed too.
* `gpu resident` — the GPU path with the `DeviceContext` and the uploaded
  dictionary created *outside* the timer, because a column chunk has one
  dictionary and many pages. Still inside the timer: the index upload, the
  bounds-check kernel, the gather kernel(s), the synchronize, the output
  allocation and the download.
* `gpu fresh` — the same, but uploading the dictionary on every call. This is
  one page in complete isolation, with nothing amortised.

`DeviceContext()` creation is *not* in any of them; `gpu-memory` reports it
separately (it is a once-per-process cost, and putting it in a per-page number
would be dishonest in the other direction).

**And the number that actually matters.** `gather_dict_into` — the fused
production path, which decodes the RLE index stream and gathers a block at a
time without ever materialising the index list — is timed on the real pages
too. That is what the GPU would have to beat to be worth shipping, and it is
faster than `gather_into` by the margin the fusion bought.

Percentiles are p50 and p90, nearest rank.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from parquet import DefaultCodecs, ParquetReader
from parquet.encoding import (
    PK_VAR,
    PhysBuffer,
    gather_dict_into,
    gather_into,
)
from thrift import read_parquet_file

from dictpages import DictChunk, find_dict_chunks
from gather_shared import BLOCK, U32, U8, gather_fixed_kernel
from gpu_gather import (
    DeviceDict,
    gpu_gather_into,
    gpu_gather_into_resident,
    upload_u32,
)
from kernel_cache import cached_enqueue

comptime WIDE = "build/bench-wide.parquet"
comptime BIG = "tests/fixtures/big.parquet"

comptime BUDGET_NS = 300000000
"""Roughly how long to spend on each measurement, before the repeat floor."""
comptime MIN_REPS = 7
comptime MAX_REPS = 401


def _sorted(var samples: List[Int]) -> List[Int]:
    """Insertion-sort a small sample list.

    Args:
        samples: The measurements.

    Returns:
        The same measurements, ascending.
    """
    var s = samples^
    for i in range(1, len(s)):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    return s^


def _pcts(var samples: List[Int]) -> Tuple[Int, Int]:
    """p50 and p90 of `samples`, nearest rank.

    Args:
        samples: The measurements.

    Returns:
        The median and the 90th percentile.
    """
    var s = _sorted(samples^)
    var n = len(s)
    var i90 = (9 * n) // 10
    if i90 >= n:
        i90 = n - 1
    return (s[n // 2], s[i90])


def _reps(n: Int, est_ns: Int) -> Int:
    """How many repeats to run, from a one-shot estimate.

    Args:
        n: How many values the measurement gathers.
        est_ns: What one call took, in nanoseconds.

    Returns:
        The repeat count.
    """
    _ = n
    if est_ns <= 0:
        return MAX_REPS
    var want = BUDGET_NS // est_ns
    if want < MIN_REPS:
        return MIN_REPS
    if want > MAX_REPS:
        return MAX_REPS
    return Int(want)


struct Row(Movable):
    """One measured configuration."""

    var label: String
    var n: Int
    var p50: Int
    var p90: Int

    def __init__(out self, var label: String, n: Int, p50: Int, p90: Int):
        """Record a measurement.

        Args:
            label: What was measured.
            n: How many values it gathered.
            p50: Median nanoseconds.
            p90: 90th-percentile nanoseconds.
        """
        self.label = label^
        self.n = n
        self.p50 = p50
        self.p90 = p90

    def __init__(out self, *, deinit move: Self):
        """Move.

        Args:
            move: The value being moved from.
        """
        self.label = move.label^
        self.n = move.n
        self.p50 = move.p50
        self.p90 = move.p90


def _ns_per_value(p50: Int, n: Int) -> String:
    """`p50` nanoseconds spread over `n` values, to two decimals.

    Args:
        p50: Median nanoseconds.
        n: How many values.

    Returns:
        The per-value cost as text.
    """
    if n == 0:
        return String("-")
    var hundredths = (p50 * 100) // n
    return String(
        hundredths // 100, ".", (hundredths % 100) // 10, hundredths % 10
    )


def _pad(var s: String, width: Int) -> String:
    """Right-align `s` in `width` columns.

    Args:
        s: The text.
        width: The column width.

    Returns:
        The padded text.
    """
    var out = String()
    for _ in range(width - s.byte_length()):
        out += " "
    out += s
    return out^


def report(rows: List[Row]) raises:
    """Print a table of measurements.

    Args:
        rows: What was measured.

    Raises:
        Never; `raises` only so callers can stay uniform.
    """
    print(
        "  ",
        _pad(String("values"), 9),
        _pad(String("p50 ns"), 12),
        _pad(String("p90 ns"), 12),
        _pad(String("ns/value"), 10),
        "  what",
    )
    for i in range(len(rows)):
        ref r = rows[i]
        print(
            "  ",
            _pad(String(r.n), 9),
            _pad(String(r.p50), 12),
            _pad(String(r.p90), 12),
            _pad(_ns_per_value(r.p50, r.n), 10),
            "  ",
            r.label,
        )


def bench_cpu(
    dict: PhysBuffer, indices: List[UInt32], var label: String
) raises -> Row:
    """Time `gather_into` on one index list.

    Args:
        dict: The dictionary page.
        indices: The decoded dictionary indices.
        label: What to call this row.

    Returns:
        The measurement.

    Raises:
        If the gather raises.
    """
    var n = len(indices)
    var t0 = perf_counter_ns()
    var warm = PhysBuffer(dict.kind, dict.width)
    gather_into(warm, dict, indices)
    var t1 = perf_counter_ns()
    var reps = _reps(n, Int(t1 - t0))
    var samples = List[Int]()
    for _ in range(reps):
        var a = perf_counter_ns()
        var out = PhysBuffer(dict.kind, dict.width)
        gather_into(out, dict, indices)
        var b = perf_counter_ns()
        samples.append(Int(b - a))
        _ = out^
    var p = _pcts(samples^)
    _ = warm^
    return Row(label^, n, p[0], p[1])


def bench_gpu_resident(
    ctx: DeviceContext,
    dict: PhysBuffer,
    indices: List[UInt32],
    var label: String,
) raises -> Row:
    """Time the GPU path with the dictionary already on the device.

    Args:
        ctx: The device context.
        dict: The dictionary page.
        indices: The decoded dictionary indices.
        label: What to call this row.

    Returns:
        The measurement.

    Raises:
        If the gather raises.
    """
    var n = len(indices)
    var ddict = DeviceDict(ctx, dict)
    var t0 = perf_counter_ns()
    var warm = PhysBuffer(dict.kind, dict.width)
    gpu_gather_into_resident(ctx, warm, ddict, indices)
    var t1 = perf_counter_ns()
    var reps = _reps(n, Int(t1 - t0))
    var samples = List[Int]()
    for _ in range(reps):
        var a = perf_counter_ns()
        var out = PhysBuffer(dict.kind, dict.width)
        gpu_gather_into_resident(ctx, out, ddict, indices)
        var b = perf_counter_ns()
        samples.append(Int(b - a))
        _ = out^
    var p = _pcts(samples^)
    _ = warm^
    _ = ddict^
    return Row(label^, n, p[0], p[1])


def bench_gpu_hostcheck(
    ctx: DeviceContext,
    dict: PhysBuffer,
    indices: List[UInt32],
    var label: String,
) raises -> Row:
    """The GPU path with the bounds check moved back onto the calling thread.

    Args:
        ctx: The device context.
        dict: The dictionary page.
        indices: The decoded dictionary indices.
        label: What to call this row.

    Returns:
        The measurement.

    Raises:
        If the gather raises.
    """
    var n = len(indices)
    var ddict = DeviceDict(ctx, dict)
    var t0 = perf_counter_ns()
    var warm = PhysBuffer(dict.kind, dict.width)
    gpu_gather_into_resident(ctx, warm, ddict, indices, True, True)
    var t1 = perf_counter_ns()
    var reps = _reps(n, Int(t1 - t0))
    var samples = List[Int]()
    for _ in range(reps):
        var a = perf_counter_ns()
        var out = PhysBuffer(dict.kind, dict.width)
        gpu_gather_into_resident(ctx, out, ddict, indices, True, True)
        var b = perf_counter_ns()
        samples.append(Int(b - a))
        _ = out^
    var p = _pcts(samples^)
    _ = warm^
    _ = ddict^
    return Row(label^, n, p[0], p[1])


def bench_kernel_only(
    ctx: DeviceContext,
    dict: PhysBuffer,
    indices: List[UInt32],
    var label: String,
) raises -> Row:
    """The fixed-width gather kernel alone: no upload, no download, no check.

    Every buffer is created and filled before the timer starts and the output
    is left on the device, so this is the *floor* — what the GPU path would
    cost if the host API were free. It is not a usable implementation; it is
    there to separate "the kernel is slow" from "the plumbing is slow".

    Args:
        ctx: The device context.
        dict: The dictionary page.
        indices: The decoded dictionary indices.
        label: What to call this row.

    Returns:
        The measurement.

    Raises:
        If the launch fails.
    """
    var n = len(indices)
    var w = dict.width
    var ddict = DeviceDict(ctx, dict)
    var idx_dev = upload_u32(ctx, Span(indices))
    var dst = ctx.enqueue_create_buffer[U8](n * w)
    ctx.synchronize()
    var ld = row_major(n * w)
    var ls = row_major(ddict.nbytes)
    var li = row_major(n)
    comptime k = gather_fixed_kernel[type_of(ld), type_of(ls), type_of(li)]
    for _ in range(3):
        cached_enqueue[k](
            ctx,
            TileTensor(dst, ld),
            TileTensor(ddict.bytes, ls),
            TileTensor(idx_dev, li),
            Int32(w),
            Int32(n),
            grid_dim=ceildiv(n, BLOCK),
            block_dim=BLOCK,
        )
    ctx.synchronize()
    var samples = List[Int]()
    for _ in range(MAX_REPS):
        var a = perf_counter_ns()
        cached_enqueue[k](
            ctx,
            TileTensor(dst, ld),
            TileTensor(ddict.bytes, ls),
            TileTensor(idx_dev, li),
            Int32(w),
            Int32(n),
            grid_dim=ceildiv(n, BLOCK),
            block_dim=BLOCK,
        )
        ctx.synchronize()
        var b = perf_counter_ns()
        samples.append(Int(b - a))
    var p = _pcts(samples^)
    _ = dst^
    _ = idx_dev^
    _ = ddict^
    return Row(label^, n, p[0], p[1])


def bench_gpu_fresh(
    ctx: DeviceContext,
    dict: PhysBuffer,
    indices: List[UInt32],
    var label: String,
) raises -> Row:
    """Time the GPU path with the dictionary uploaded on every call.

    Args:
        ctx: The device context.
        dict: The dictionary page.
        indices: The decoded dictionary indices.
        label: What to call this row.

    Returns:
        The measurement.

    Raises:
        If the gather raises.
    """
    var n = len(indices)
    var t0 = perf_counter_ns()
    var warm = PhysBuffer(dict.kind, dict.width)
    gpu_gather_into(ctx, warm, dict, indices)
    var t1 = perf_counter_ns()
    var reps = _reps(n, Int(t1 - t0))
    var samples = List[Int]()
    for _ in range(reps):
        var a = perf_counter_ns()
        var out = PhysBuffer(dict.kind, dict.width)
        gpu_gather_into(ctx, out, dict, indices)
        var b = perf_counter_ns()
        samples.append(Int(b - a))
        _ = out^
    var p = _pcts(samples^)
    _ = warm^
    return Row(label^, n, p[0], p[1])


def take(chunk: DictChunk, n: Int) raises -> List[UInt32]:
    """`n` indices, cycled out of the chunk's real ones.

    Cycling keeps the value distribution — how often each dictionary entry is
    hit, and how long the runs are — the distribution the file actually has,
    rather than a synthetic uniform one.

    Args:
        chunk: The column chunk to draw from.
        n: How many indices to produce.

    Returns:
        The indices.

    Raises:
        If the chunk has no pages.
    """
    var all = chunk.all_indices()
    if len(all) == 0:
        raise Error("experiments.gpu_gather: chunk has no indices")
    var out = List[UInt32](length=n, fill=0)
    var p = out.unsafe_ptr()
    var q = all.unsafe_ptr()
    var m = len(all)
    for i in range(n):
        p.unsafe_store(i, q.unsafe_load(i % m))
    return out^


def sweep(
    ctx: DeviceContext, chunk: DictChunk, var name: String, whole: Int
) raises:
    """CPU against GPU across page sizes, for one column chunk's dictionary.

    Args:
        ctx: The device context.
        chunk: The column chunk to draw the dictionary and indices from.
        name: What to call this sweep.
        whole: The chunk's real total value count, added to the sweep.

    Raises:
        If a gather raises.
    """
    print()
    print(
        "── ",
        name,
        " — ",
        chunk.dict.count,
        " dictionary entries, ",
        len(chunk.dict.bytes),
        " dictionary bytes, ",
        "BYTE_ARRAY" if chunk.dict.kind
        == PK_VAR else String(chunk.dict.width, "-byte fixed"),
        sep="",
    )
    # Ascending and ending at the chunk's real size, so the two-point fit
    # below really does span the smallest and the largest measurement.
    var ladder = [64, 256, 1024, 4096, 16384, 65536]
    var sizes = List[Int]()
    for i in range(len(ladder)):
        if ladder[i] < whole:
            sizes.append(ladder[i])
    sizes.append(whole)
    var stride = 5 if chunk.dict.kind != PK_VAR else 4
    var rows = List[Row]()
    for si in range(len(sizes)):
        var n = sizes[si]
        var idx = take(chunk, n)
        rows.append(bench_cpu(chunk.dict, idx, String("cpu  gather_into")))
        rows.append(
            bench_gpu_resident(
                ctx, chunk.dict, idx, String("gpu  resident dictionary")
            )
        )
        rows.append(
            bench_gpu_hostcheck(
                ctx,
                chunk.dict,
                idx,
                String("gpu  resident + host bounds check"),
            )
        )
        rows.append(
            bench_gpu_fresh(
                ctx, chunk.dict, idx, String("gpu  dictionary uploaded too")
            )
        )
        if chunk.dict.kind != PK_VAR:
            rows.append(
                bench_kernel_only(
                    ctx,
                    chunk.dict,
                    idx,
                    String("gpu  kernel + synchronize only (floor)"),
                )
            )
    report(rows)

    # A two-point fit — smallest and largest n — gives each variant a fixed
    # cost and a marginal cost per value, and the crossover falls out of them.
    # Where the GPU's marginal cost is the larger of the two there is no
    # crossover at any size: the gap widens with n rather than closing.
    print()
    print("   fixed cost and marginal cost per value (two-point fit):")
    var lo = 0
    var hi = len(rows) - stride
    var dn = rows[hi].n - rows[lo].n
    var cpu_slope_x1000 = ((rows[hi].p50 - rows[lo].p50) * 1000) // dn
    for j in range(stride):
        var f = rows[lo + j].p50
        var slope_x1000 = ((rows[hi + j].p50 - rows[lo + j].p50) * 1000) // dn
        var note: String
        if j == 0:
            note = String(" (the baseline)")
        elif slope_x1000 >= cpu_slope_x1000:
            note = String(" — marginal cost >= the CPU's, so no crossover ever")
        else:
            var gap = cpu_slope_x1000 - slope_x1000
            var cross = ((f - rows[lo].p50) * 1000) // gap if gap > 0 else 0
            note = String(" — would cross the CPU at ~", cross, " values")
        print(
            "     ",
            _pad(String(f), 10),
            " ns fixed  +  ",
            _pad(String(slope_x1000), 6),
            "/1000 ns per value   ",
            rows[lo + j].label,
            note,
            sep="",
        )

    print()
    print("   crossover (best GPU variant vs cpu gather_into):")
    var found = False
    for i in range(0, len(rows), stride):
        var cpu = rows[i].p50
        var gpu = rows[i + 2].p50
        if rows[i + 1].p50 < gpu:
            gpu = rows[i + 1].p50
        if stride == 5 and rows[i + 4].p50 < gpu:
            gpu = rows[i + 4].p50
        if gpu <= cpu and not found:
            print(
                "     GPU first wins at ",
                rows[i].n,
                " values (",
                gpu,
                " ns vs ",
                cpu,
                " ns)",
                sep="",
            )
            found = True
    if not found:
        var last = len(rows) - stride
        var best = rows[last + 2].p50
        if rows[last + 1].p50 < best:
            best = rows[last + 1].p50
        if stride == 5 and rows[last + 4].p50 < best:
            best = rows[last + 4].p50
        print(
            "     never — at the largest size measured (",
            rows[last].n,
            " values) the best GPU variant still takes ",
            (best * 100) // rows[last].p50,
            "% of the CPU's time",
            sep="",
        )


def bench_real_chunk(
    ctx: DeviceContext, chunk: DictChunk, var name: String
) raises:
    """The whole column chunk, page by page, as the reader would decode it.

    Three paths: the fused `gather_dict_into` the reader actually ships, the
    unfused `decode_dict_indices` + `gather_into` it replaced, and the GPU with
    the dictionary resident.

    Args:
        ctx: The device context.
        chunk: The column chunk.
        name: What to call this measurement.

    Raises:
        If a gather raises.
    """
    var total = chunk.total_values()
    print()
    print(
        "── ",
        name,
        " — the real chunk, page by page (",
        len(chunk.pages),
        " pages, ",
        total,
        " values)",
        sep="",
    )

    var fused = List[Int]()
    for _ in range(MIN_REPS):
        var a = perf_counter_ns()
        var out = PhysBuffer(chunk.dict.kind, chunk.dict.width)
        for p in range(len(chunk.raw)):
            gather_dict_into(
                out, chunk.dict, Span(chunk.raw[p]), chunk.counts[p]
            )
        var b = perf_counter_ns()
        fused.append(Int(b - a))
        _ = out^

    var unfused = List[Int]()
    for _ in range(MIN_REPS):
        var a = perf_counter_ns()
        var out = PhysBuffer(chunk.dict.kind, chunk.dict.width)
        for p in range(len(chunk.pages)):
            gather_into(out, chunk.dict, chunk.pages[p])
        var b = perf_counter_ns()
        unfused.append(Int(b - a))
        _ = out^

    var ddict = DeviceDict(ctx, chunk.dict)
    var gpu = List[Int]()
    for _ in range(MIN_REPS):
        var a = perf_counter_ns()
        var out = PhysBuffer(chunk.dict.kind, chunk.dict.width)
        for p in range(len(chunk.pages)):
            gpu_gather_into_resident(ctx, out, ddict, chunk.pages[p])
        var b = perf_counter_ns()
        gpu.append(Int(b - a))
        _ = out^

    var rows = List[Row]()
    var pf = _pcts(fused^)
    var pu = _pcts(unfused^)
    var pg = _pcts(gpu^)
    rows.append(
        Row(
            String("cpu  gather_dict_into (fused, shipping)"),
            total,
            pf[0],
            pf[1],
        )
    )
    rows.append(
        Row(String("cpu  gather_into (unfused oracle)"), total, pu[0], pu[1])
    )
    rows.append(Row(String("gpu  resident dictionary"), total, pg[0], pg[1]))
    report(rows)
    _ = ddict^


def load(path: String) raises -> List[DictChunk]:
    """Every dictionary-encoded column chunk of one file.

    Args:
        path: The Parquet file.

    Returns:
        The chunks.

    Raises:
        If the file cannot be read.
    """
    var data = read_parquet_file(path)
    var r = ParquetReader[DefaultCodecs].from_span(Span(data))
    var chunks = find_dict_chunks[DefaultCodecs](
        Span(data), r.meta, r.schema, False, False
    )
    _ = data^
    return chunks^


def pick(
    chunks: List[DictChunk], want_var: Bool, min_entries: Int
) raises -> Int:
    """The first chunk of the requested shape.

    Args:
        chunks: The candidates.
        want_var: Whether to want a `BYTE_ARRAY` dictionary.
        min_entries: The smallest acceptable dictionary.

    Returns:
        The index of the chosen chunk.

    Raises:
        If nothing matches.
    """
    for i in range(len(chunks)):
        var is_var = chunks[i].dict.kind == PK_VAR
        if is_var == want_var and chunks[i].dict.count >= min_entries:
            return i
    raise Error("experiments.gpu_gather: no chunk of the requested shape")


def main() raises:
    """Run the sweeps and print the tables.

    Raises:
        If there is no accelerator, or a gather raises.
    """
    if not has_accelerator():
        raise Error("experiments.gpu_gather: no GPU — this bench needs Metal")
    var ctx = DeviceContext()
    print("device:", ctx.name(), "/", ctx.api())
    print(
        "note: DeviceContext creation is outside every timer; see"
        " `gpu-memory` for what it costs."
    )

    var wide = load(String(WIDE))
    var big = load(String(BIG))

    var wf = pick(wide, False, 1000)
    var wv = pick(wide, True, 900)
    var bf = pick(big, False, 20000)

    sweep(ctx, wide[wf], String("wide `c` — int64, small dictionary"), 250000)
    sweep(ctx, wide[wv], String("wide `d` — string, small dictionary"), 250000)
    sweep(
        ctx,
        big[bf],
        String("big `i` — int64, 24k-entry dictionary"),
        big[bf].total_values(),
    )

    bench_real_chunk(ctx, wide[wf], String("wide `c`"))
    bench_real_chunk(ctx, wide[wv], String("wide `d`"))
