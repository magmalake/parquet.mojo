"""The parity gate: CPU and GPU gathers of real dictionary pages, byte for byte.

This runs *before* any benchmark, for the reason the brief gives — the sibling
project recorded MAX's own fused Metal kernel returning garbage on this
hardware while its CPU path stayed coherent. For a library whose whole claim is
cell-exact agreement, a fast wrong answer is worth nothing.

What it compares: `parquet.encoding.gather_into` — the shipping CPU oracle,
vectorised loads and stores and all — against `gpu_gather.gpu_gather_into`, on
every dictionary-encoded column chunk of the fixture corpus plus the generated
wide benchmark file. Every field of the resulting `PhysBuffer` is compared:
`kind`, `width`, `count`, every value byte, and every offset. The out-of-range
error is compared as a string, not just as "both raised".

It also compares the oracle against `host_driver.shared_gather_into` — the
*same* element bodies the kernels run, with the `comptime` branch bound to
`TARGET_CPU` instead. Both instantiations are therefore in this one binary and
both are checked, which is what turns "one source serves two targets" from a
claim into a result.

**Non-vacuity.** A gate that passes because both paths were handed nothing is
the failure mode this project has hit before, so the run asserts, at the end:
at least one fixed-width and one `BYTE_ARRAY` dictionary were exercised; at
least `MIN_VALUES` values were actually gathered; at least `MIN_BYTES` value
bytes came out; at least one page was larger than a single GPU block; and the
gathers touched at least `MIN_DISTINCT` distinct dictionary entries. Any of
those failing is a failure.

**The negative control.** The same gate, in the same binary, run a second time
with `sabotage = True` — which launches every gather one block short, so any
page past 256 values loses its tail. `main` fails unless *that* run fails, and
prints which comparison caught it. See `gpu_grid` in `gather_shared.mojo`.
"""

from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from parquet import DefaultCodecs, ParquetReader
from parquet.encoding import PK_VAR, PhysBuffer, gather_into
from thrift import read_parquet_file

from dictpages import DictChunk, find_dict_chunks
from gather_shared import BLOCK
from host_driver import shared_gather_into
from gpu_gather import DeviceDict, gpu_gather_into, gpu_gather_into_resident

comptime FIXTURES = "tests/fixtures/"
comptime WIDE = "build/bench-wide.parquet"

comptime MIN_VALUES = 1000000
"""Floor on values actually gathered across the whole run."""
comptime MIN_BYTES = 4000000
"""Floor on value bytes actually produced across the whole run."""
comptime MIN_DISTINCT = 900
"""Floor on distinct dictionary entries the gathers landed on."""


struct Tally(Movable):
    """What the run actually exercised, so the gate can refuse to be vacuous."""

    var values: Int
    var bytes: Int
    var fixed_chunks: Int
    var var_chunks: Int
    var multi_block_pages: Int
    var distinct: Int
    var comparisons: Int

    def __init__(out self):
        """Nothing counted yet."""
        self.values = 0
        self.bytes = 0
        self.fixed_chunks = 0
        self.var_chunks = 0
        self.multi_block_pages = 0
        self.distinct = 0
        self.comparisons = 0

    def __init__(out self, *, deinit move: Self):
        """Move.

        Args:
            move: The value being moved from.
        """
        self.values = move.values
        self.bytes = move.bytes
        self.fixed_chunks = move.fixed_chunks
        self.var_chunks = move.var_chunks
        self.multi_block_pages = move.multi_block_pages
        self.distinct = move.distinct
        self.comparisons = move.comparisons


def _distinct_count(indices: List[UInt32], entries: Int) -> Int:
    """How many distinct dictionary entries `indices` names.

    Args:
        indices: The decoded dictionary indices.
        entries: How many entries the dictionary has.

    Returns:
        The number of distinct in-range entries touched.
    """
    var seen = List[Bool](length=entries, fill=False)
    var n = 0
    for i in range(len(indices)):
        var k = Int(indices[i])
        if k < entries and not seen[k]:
            seen[k] = True
            n += 1
    return n


def assert_same(cpu: PhysBuffer, gpu: PhysBuffer, what: String) raises:
    """Fail unless the two buffers are identical in every field.

    Args:
        cpu: What `gather_into` produced.
        gpu: What the GPU path produced.
        what: A label for the error message.

    Raises:
        If anything differs.
    """
    if cpu.kind != gpu.kind:
        raise Error(
            String(what, ": kind ", cpu.kind, " (cpu) vs ", gpu.kind, " (gpu)")
        )
    if cpu.width != gpu.width:
        raise Error(
            String(
                what, ": width ", cpu.width, " (cpu) vs ", gpu.width, " (gpu)"
            )
        )
    if cpu.count != gpu.count:
        raise Error(
            String(
                what, ": count ", cpu.count, " (cpu) vs ", gpu.count, " (gpu)"
            )
        )
    if len(cpu.bytes) != len(gpu.bytes):
        raise Error(
            String(
                what,
                ": ",
                len(cpu.bytes),
                " value bytes (cpu) vs ",
                len(gpu.bytes),
                " (gpu)",
            )
        )
    var cb = cpu.bytes.unsafe_ptr()
    var gb = gpu.bytes.unsafe_ptr()
    for i in range(len(cpu.bytes)):
        if cb.unsafe_load(i) != gb.unsafe_load(i):
            raise Error(
                String(
                    what,
                    ": value byte ",
                    i,
                    " of ",
                    len(cpu.bytes),
                    " is ",
                    cb.unsafe_load(i),
                    " (cpu) but ",
                    gb.unsafe_load(i),
                    " (gpu)",
                )
            )
    if len(cpu.offsets) != len(gpu.offsets):
        raise Error(
            String(
                what,
                ": ",
                len(cpu.offsets),
                " offsets (cpu) vs ",
                len(gpu.offsets),
                " (gpu)",
            )
        )
    var co = cpu.offsets.unsafe_ptr()
    var go = gpu.offsets.unsafe_ptr()
    for i in range(len(cpu.offsets)):
        if co.unsafe_load(i) != go.unsafe_load(i):
            raise Error(
                String(
                    what,
                    ": offset ",
                    i,
                    " of ",
                    len(cpu.offsets),
                    " is ",
                    co.unsafe_load(i),
                    " (cpu) but ",
                    go.unsafe_load(i),
                    " (gpu)",
                )
            )


def check_one[
    sabotage: Bool
](
    ctx: DeviceContext,
    dict: PhysBuffer,
    indices: List[UInt32],
    what: String,
    mut tally: Tally,
) raises:
    """Gather `indices` both ways into a fresh buffer and compare.

    Parameters:
        sabotage: Run the negative control's deliberately broken launch.

    Args:
        ctx: The device context.
        dict: The dictionary page.
        indices: The decoded dictionary indices.
        what: A label for the error message.
        tally: Updated with what this comparison exercised.

    Raises:
        If the two paths disagree anywhere.
    """
    var cpu = PhysBuffer(dict.kind, dict.width)
    gather_into(cpu, dict, indices)
    var gpu = PhysBuffer(dict.kind, dict.width)
    gpu_gather_into[sabotage](ctx, gpu, dict, indices)
    assert_same(cpu, gpu, what)
    # The same element bodies the kernels run, driven on the CPU. This is what
    # makes "one source, two targets" evidence rather than an assertion: both
    # instantiations of the `comptime` branch are in this binary and both are
    # checked against the shipping oracle.
    var shared = PhysBuffer(dict.kind, dict.width)
    shared_gather_into(shared, dict, indices)
    assert_same(cpu, shared, String(what, " [shared body, CPU driver]"))
    tally.comparisons += 2
    tally.values += cpu.count
    tally.bytes += len(cpu.bytes)
    if len(indices) > BLOCK:
        tally.multi_block_pages += 1


def check_chunk[
    sabotage: Bool
](ctx: DeviceContext, chunk: DictChunk, label: String, mut tally: Tally) raises:
    """Every page of a chunk on its own, then the whole chunk accumulated.

    The accumulated pass is the one that catches an offsets bug: `gather_into`
    appends onto a buffer that already holds values, and the `BYTE_ARRAY`
    offsets it writes are absolute, not page-relative.

    Parameters:
        sabotage: Run the negative control's deliberately broken launch.

    Args:
        ctx: The device context.
        chunk: The column chunk's dictionary and index streams.
        label: A label for error messages.
        tally: Updated with what this chunk exercised.

    Raises:
        If the two paths disagree anywhere.
    """
    if chunk.dict.kind == PK_VAR:
        tally.var_chunks += 1
    else:
        tally.fixed_chunks += 1

    for p in range(len(chunk.pages)):
        check_one[sabotage](
            ctx,
            chunk.dict,
            chunk.pages[p],
            String(label, " page ", p),
            tally,
        )

    # The whole chunk, page after page, into one growing buffer — and on the
    # GPU with the dictionary uploaded once, which is the path the benchmark
    # times.
    var cpu = PhysBuffer(chunk.dict.kind, chunk.dict.width)
    var gpu = PhysBuffer(chunk.dict.kind, chunk.dict.width)
    var ddict = DeviceDict(ctx, chunk.dict)
    for p in range(len(chunk.pages)):
        gather_into(cpu, chunk.dict, chunk.pages[p])
        gpu_gather_into_resident[sabotage](ctx, gpu, ddict, chunk.pages[p])
    assert_same(cpu, gpu, String(label, " accumulated"))
    tally.comparisons += 1

    var all = chunk.all_indices()
    tally.distinct += _distinct_count(all, chunk.dict.count)
    print(
        "  ",
        label,
        ": ",
        len(chunk.pages),
        " page(s), ",
        chunk.dict.count,
        " dictionary entries, ",
        cpu.count,
        " values, ",
        len(cpu.bytes),
        " bytes — identical",
        sep="",
    )


def check_edges[
    sabotage: Bool
](ctx: DeviceContext, dict: PhysBuffer, label: String, mut tally: Tally) raises:
    """The awkward pages: empty, one entry, and every index the same.

    Parameters:
        sabotage: Run the negative control's deliberately broken launch.

    Args:
        ctx: The device context.
        dict: A real dictionary page to build the cases on.
        label: A label for error messages.
        tally: Updated with what these cases exercised.

    Raises:
        If the two paths disagree anywhere.
    """
    # An empty page. Both paths must leave `out` untouched — and the gate must
    # not count this as coverage, which is what the tally floors enforce.
    check_one[sabotage](
        ctx, dict, List[UInt32](), String(label, " empty page"), tally
    )

    # A page whose indices are all the same — the run-length case the RLE
    # decoder produces constantly.
    var same = List[UInt32](length=4096, fill=UInt32(dict.count - 1))
    check_one[sabotage](ctx, dict, same, String(label, " uniform page"), tally)

    # A single-entry dictionary, gathered many times.
    var one = PhysBuffer(dict.kind, dict.width)
    gather_into(one, dict, List[UInt32](length=1, fill=UInt32(0)))
    var zeros = List[UInt32](length=1000, fill=UInt32(0))
    check_one[sabotage](
        ctx, one, zeros, String(label, " single-entry dictionary"), tally
    )

    # A page of exactly one block, and one of one block plus one, so the tail
    # handling is exercised from both sides.
    var b0 = List[UInt32](length=BLOCK, fill=UInt32(0))
    var b1 = List[UInt32](length=BLOCK + 1, fill=UInt32(0))
    for i in range(BLOCK):
        b0[i] = UInt32(i % dict.count)
        b1[i] = UInt32(i % dict.count)
    b1[BLOCK] = UInt32(dict.count - 1)
    check_one[sabotage](
        ctx, dict, b0, String(label, " exactly one block"), tally
    )
    check_one[sabotage](
        ctx, dict, b1, String(label, " one block plus one"), tally
    )


def check_out_of_range[
    sabotage: Bool
](ctx: DeviceContext, dict: PhysBuffer) raises:
    """A corrupt index must raise the same error from both paths.

    Parameters:
        sabotage: Run the negative control's deliberately broken launch.

    Args:
        ctx: The device context.
        dict: A real dictionary page.

    Raises:
        If either path accepts the index, or if the messages differ.
    """
    var bad = List[UInt32](length=1000, fill=UInt32(0))
    bad[700] = UInt32(dict.count)  # one past the end
    bad[900] = UInt32(dict.count + 5)  # a later, larger offender

    var cpu_msg: String
    try:
        var cpu = PhysBuffer(dict.kind, dict.width)
        gather_into(cpu, dict, bad)
        raise Error("out-of-range: the CPU path accepted a bad index")
    except e:
        cpu_msg = String(e)

    var gpu_msg: String
    try:
        var gpu = PhysBuffer(dict.kind, dict.width)
        gpu_gather_into[sabotage](ctx, gpu, dict, bad)
        raise Error("out-of-range: the GPU path accepted a bad index")
    except e:
        gpu_msg = String(e)

    if cpu_msg != gpu_msg:
        raise Error(
            String(
                "out-of-range: the two paths raise different errors\n  cpu: ",
                cpu_msg,
                "\n  gpu: ",
                gpu_msg,
            )
        )
    print("   out-of-range index: both raise `", cpu_msg, "`", sep="")


def check_file[
    sabotage: Bool
](
    ctx: DeviceContext, path: String, mut tally: Tally, mut edges_done: Bool
) raises:
    """Every dictionary-encoded column chunk of one file.

    Parameters:
        sabotage: Run the negative control's deliberately broken launch.

    Args:
        ctx: The device context.
        path: The Parquet file to read.
        tally: Updated with what the file exercised.
        edges_done: Set once the edge and error cases have run on a real
            fixed-width and a real `BYTE_ARRAY` dictionary.

    Raises:
        If the two paths disagree anywhere.
    """
    var data = read_parquet_file(path)
    var r = ParquetReader[DefaultCodecs].from_span(Span(data))
    var chunks = find_dict_chunks[DefaultCodecs](
        Span(data), r.meta, r.schema, False, False
    )
    if len(chunks) == 0:
        print("  ", path, ": no dictionary-encoded chunk, skipped", sep="")
        return
    print(path, ":", sep="")
    for i in range(len(chunks)):
        check_chunk[sabotage](
            ctx,
            chunks[i],
            String(path, " [", chunks[i].path, " #", i, "]"),
            tally,
        )
    if not edges_done:
        var have_fixed = -1
        var have_var = -1
        for i in range(len(chunks)):
            if chunks[i].dict.kind == PK_VAR:
                if have_var < 0 and chunks[i].dict.count > 1:
                    have_var = i
            elif have_fixed < 0 and chunks[i].dict.count > 1:
                have_fixed = i
        if have_fixed >= 0 and have_var >= 0:
            check_edges[sabotage](ctx, chunks[have_fixed].dict, "fixed", tally)
            check_edges[sabotage](
                ctx, chunks[have_var].dict, "byte_array", tally
            )
            check_out_of_range[sabotage](ctx, chunks[have_fixed].dict)
            check_out_of_range[sabotage](ctx, chunks[have_var].dict)
            edges_done = True
    _ = data^


def run_gate[sabotage: Bool](ctx: DeviceContext) raises -> Tally:
    """Compare both paths over the whole corpus and return what it exercised.

    Parameters:
        sabotage: Run the negative control's deliberately broken launch.

    Args:
        ctx: The device context.

    Returns:
        What the run exercised, for the non-vacuity assertions.

    Raises:
        If the two paths disagree anywhere.
    """
    var tally = Tally()
    var edges_done = False

    # The wide benchmark file first: it holds the biggest real dictionary
    # column chunks available (250k values a chunk, an int64 dictionary and a
    # string one), and it is where the edge cases get built.
    check_file[sabotage](ctx, String(WIDE), tally, edges_done)
    var corpus = [
        "encodings.parquet",
        "big.parquet",
        "manypages.parquet",
        "prune.parquet",
        "primitives.parquet",
        "v2pages.parquet",
        "v1pages.parquet",
        "logical.parquet",
        "pageindex.parquet",
        "bloom.parquet",
    ]
    for i in range(len(corpus)):
        check_file[sabotage](
            ctx, String(FIXTURES, corpus[i]), tally, edges_done
        )

    if not edges_done:
        raise Error(
            "vacuous: the edge cases never ran — no file offered both a"
            " fixed-width and a BYTE_ARRAY dictionary"
        )
    return tally^


def assert_not_vacuous(tally: Tally) raises:
    """Refuse a run that agreed because it was handed almost nothing.

    Args:
        tally: What the run exercised.

    Raises:
        If any floor was not reached.
    """
    if tally.fixed_chunks == 0:
        raise Error("vacuous: no fixed-width dictionary was exercised")
    if tally.var_chunks == 0:
        raise Error("vacuous: no BYTE_ARRAY dictionary was exercised")
    if tally.multi_block_pages == 0:
        raise Error(
            String(
                "vacuous: every page fitted in one ",
                BLOCK,
                "-thread block, so the tail was never tested",
            )
        )
    if tally.values < MIN_VALUES:
        raise Error(
            String(
                "vacuous: only ",
                tally.values,
                " values gathered, wanted at least ",
                MIN_VALUES,
            )
        )
    if tally.bytes < MIN_BYTES:
        raise Error(
            String(
                "vacuous: only ",
                tally.bytes,
                " value bytes produced, wanted at least ",
                MIN_BYTES,
            )
        )
    if tally.distinct < MIN_DISTINCT:
        raise Error(
            String(
                "vacuous: the gathers touched only ",
                tally.distinct,
                " distinct dictionary entries, wanted at least ",
                MIN_DISTINCT,
            )
        )


def main() raises:
    """Run the gate, then the negative control, and exit non-zero on either.

    Raises:
        If the two paths disagree, if the run was vacuous, or if the
        deliberately broken build passed the gate anyway.
    """
    if not has_accelerator():
        raise Error("experiments.gpu_gather: no GPU — this gate needs Metal")
    var ctx = DeviceContext()
    print("device:", ctx.name(), "/", ctx.api())
    print()

    var tally = run_gate[False](ctx)

    print()
    print("comparisons:        ", tally.comparisons)
    print("values gathered:    ", tally.values)
    print("value bytes:        ", tally.bytes)
    print("fixed-width chunks: ", tally.fixed_chunks)
    print("byte-array chunks:  ", tally.var_chunks)
    print("multi-block pages:  ", tally.multi_block_pages)
    print("distinct entries:   ", tally.distinct)
    assert_not_vacuous(tally)
    print()
    print("gpu-gather parity gate: PASS")

    # ── the negative control ────────────────────────────────────────────────
    # The same gate against a GPU path launched one block short. If this run
    # also passes, the gate is not testing anything and the PASS above is
    # worthless.
    print()
    print("negative control: the same gate, GPU launched one block short")
    var caught = String()
    try:
        var bad = run_gate[True](ctx)
        assert_not_vacuous(bad)
    except e:
        caught = String(e)
    if not caught:
        raise Error(
            "NEGATIVE CONTROL FAILED: the sabotaged GPU path passed the parity"
            " gate, so the gate proves nothing"
        )
    print("  caught: ", caught, sep="")
    print()
    print("negative control: PASS (the gate fails when the kernel is wrong)")
