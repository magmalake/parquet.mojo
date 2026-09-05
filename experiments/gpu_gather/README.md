# The dictionary gather on Metal — a spike, and its verdict

**Question.** Can one Mojo source serve both a CPU and a GPU decode path, with
the two proven to agree cell for cell? And is the GPU path worth having?

**Answer.** The inner loop can be shared, and the two paths do agree — 554
byte-for-byte comparisons over 2.31 million values of the real fixture corpus,
with a negative control proving the gate can fail. The GPU path is
**6× to 31× slower** than the CPU path it would replace, and it does not get
better with size: its cost *per value* is higher than the CPU's, so the gap
widens rather than closing. **Do not pursue this.**

Nothing here ships. It lives outside `src/`, in its own pixi environment, and
the published tin does not gain a `max` dependency — `pixi build` emits a
bit-identical `parquet.mojopkg` with and without this directory, and all five
pre-existing environments are unchanged in `pixi.lock`.

## Running it

    pixi run -e gpu gpu-probe     # is there a Metal device at all
    pixi run -e gpu gpu-parity    # the gate + its negative control
    pixi run -e gpu gpu-bench     # CPU vs GPU, every overhead included
    pixi run -e gpu gpu-memory    # where the time goes; the unified-memory question
    pixi run -e gpu lint-gpu      # mojolint over experiments/

`gpu-bench` and `gpu-parity` need `build/bench-wide.parquet`
(`python tools/bench_pyarrow.py --make`). Everything is built ahead of time and
run as a binary; nothing here is `mojo run`, so the GPU and CPU numbers come
out of the same AOT binary.

## 1. Can the logic be shared?

Partly, and the boundary is sharp. `gather_shared.mojo` has three layers.

**The element bodies are genuinely shared.** `_copy_entry`, `_entry_len` and
`_index_at` are `TileTensor` indexing and byte moves, with no `global_idx`, no
`barrier` and no allocation. Both targets run this exact code.

**The driver is a two-line `comptime` branch.** This is the code that decides
the question:

```mojo
comptime if target == TARGET_GPU:
    var i = Int(global_idx.x)
    if i < n:
        _copy_entry(dst, src, i * w, _index_at(idx, i) * w, w)
else:
    for i in range(n):
        _copy_entry(dst, src, i * w, _index_at(idx, i) * w, w)
```

`gather_fixed_kernel` binds `TARGET_GPU`; `host_driver.shared_gather_into`
binds `TARGET_CPU` on the same three bodies. Both instantiations are in the
gate's binary and the gate checks *both* against `parquet.encoding.gather_into`
— so "one source, two targets" is a result here, not a claim.

**Everything above that line forks completely, and has to.**

* *Allocation.* The CPU grows a `List[UInt8]` in place. The GPU sizes a device
  buffer up front, which for `BYTE_ARRAY` means it cannot know the size until
  the scan has run.
* *The bounds check.* A kernel cannot raise. The GPU version max-reduces per
  block and hands the host a small array to reduce and act on
  (`index_max_kernel` + `_settle_bounds`); only on a corrupt page does it scan
  for the first offender, so the message and the index it names are identical
  to `gather_into`'s. That is three pieces of code to replace one `if`.
* *The offset prefix sum.* `total += ln` on the CPU; a two-level block scan on
  the GPU (`scan_block_kernel`, `scan_sums_kernel`, `scan_add_kernel`). There
  is no shared body here at all, and there cannot be — the algorithms are
  different, not the same algorithm on different hardware.

And the sharing is not free even where it works. `parquet.encoding`'s
`gather_into` moves 4- and 8-byte entries with one widened load and store and
byte arrays eight bytes at a time. The shared body is scalar. A CPU driver
built on the shared body would be slower than the CPU code this repository
already has, so the shared body buys uniformity by giving up the CPU's
vectorisation. The parity gate therefore compares the GPU against the *real*
`gather_into`, not against the shared-body CPU driver.

**So: the answer is "the inner loop, yes; the surrounding operation, no".**
About 40 lines are genuinely shared and about 300 are not.

## 2. Does it agree? (the gate, and the control)

`pixi run -e gpu gpu-parity`. Real dictionary pages from
`build/bench-wide.parquet` and ten fixture files, gathered by
`parquet.encoding.gather_into` and by `gpu_gather.gpu_gather_into`, compared on
`kind`, `width`, `count`, every value byte and every offset.

```
comparisons:         554
values gathered:     2314527
value bytes:         14356457
fixed-width chunks:  70
byte-array chunks:   30
multi-block pages:   135
distinct entries:    231995

gpu-gather parity gate: PASS
```

Covered: fixed-width and `BYTE_ARRAY` dictionaries; an empty page; a
single-entry dictionary gathered a thousand times; a page whose indices are all
the same; a page of exactly one GPU block and one of a block plus one; each
page on its own and then the whole chunk accumulated into one growing buffer
(which is what catches an absolute-vs-page-relative offsets bug); and an
out-of-range index, where the two paths are required to raise the *same string*:

```
parquet.encoding: dictionary index 1000 out of range (dictionary has 1000 entries)
```

**Non-vacuity is asserted, not hoped for.** The run fails unless it exercised
both dictionary kinds, at least one page larger than a single block, a million
values, four million value bytes and nine hundred distinct dictionary entries.
Handing both paths nothing is a failure, not a pass.

**The negative control is in the same binary.** `gpu_grid[sabotage]` launches
one block short — the classic GPU bug, and deliberately a partial one, so pages
of 256 values or fewer still come out right. The gate runs a second time with
`sabotage = True` and `main` fails unless *that* run fails:

```
negative control: the same gate, GPU launched one block short
  caught: build/bench-wide.parquet [c #0] page 0: value byte 159744 of 160000 is 200 (cpu) but 0 (gpu)
negative control: PASS (the gate fails when the kernel is wrong)
```

It is caught at byte 159744 of 160000 — the tail of the last block, which is
exactly what a gate that only ever saw a small page would miss.

## 3. Does unified memory remove the copy?

**No.** Three findings, all from `pixi run -e gpu gpu-memory`.

**A kernel cannot touch host memory at all, and fails silently when it tries.**
A buffer from `enqueue_create_host_buffer` handed straight to a kernel reads as
zeros and swallows writes. Input 100..107, so a correct run prints 101..108:

```
   host buffer in, host buffer out:     0 0 0 0 0 0 0 0
   host buffer in, device buffer out:   1 1 1 1 1 1 1 1
   device in, device out (the control): 101 102 103 104 105 106 107 108
```

No error, no warning — the kernel read zeros and the run looks fine. That is
the same class of failure the sibling project recorded for MAX's own fused
Metal kernel on this hardware, and it is the reason the parity gate exists.

**`map_to_host` is expensive and it is not a mapping.** Per call, batched over
200 iterations:

| buffer | `map_to_host` | `synchronize` | `enqueue_create_buffer` |
|---|---:|---:|---:|
| 1 KiB | 580 µs | 157 µs | 1.4 µs |
| 64 KiB | 518 µs | 145 µs | 3.3 µs |
| 1 MiB | 572 µs | 143 µs | 3.2 µs |
| 16 MiB | 1853 µs | 166 µs | 3.0 µs |
| 64 MiB | 5215 µs | 156 µs | 6.8 µs |

Half a millisecond, flat, before any bytes move — and a size term of roughly
78 µs/MiB on top. A gather written the obvious way (map to upload, map to
download) starts at about 1.3 ms per page before doing anything.

**Staged copies are far cheaper, so that is what the driver uses.**
`enqueue_create_host_buffer` + `enqueue_copy` costs 201 µs at 1 MiB against
640 µs for `map_to_host` + `memcpy` — and almost all of that 201 µs is the
`synchronize` the measurement includes. Copies can be *enqueued* for nearly
nothing; one `ctx.synchronize()` at the end of the page retires all of them.
Rewriting the driver that way took the per-page fixed cost from ~1.3 ms to
~0.34 ms, and is why the numbers below are as good as they are.

So: the copy is required, it is not free, and the only thing unified memory
buys is that `enqueue_copy` runs at memory speed rather than over a bus.

**Kernel compilation is real but the cache fixes it.** `enqueue_function`
compiles the pipeline on every call: 649 µs cold, 194 µs warm. `cached_enqueue`
— lifted from `millfolio/engine`'s `kernel_cache.mojo` — is 211 µs cold and
193 µs warm. Worth having, and cheap; but the warm numbers are the same, so it
is not where the time goes. `DeviceContext()` creation is 187 µs, once per
process.

## 4. The numbers, and the crossover

`pixi run -e gpu gpu-bench`. Apple M4 (10 cores, 24 GB), macOS 26.6.2, Mojo
nightly `modular` 26.6, AOT-built, p50/p90 over a ~300 ms budget per
measurement.

**The machine was not idle.** Load average was 2.48 going in and 2.60 coming
out, on ten cores — other work was running in the same session. That is worth
knowing, and it does not touch the conclusion: the p90/p50 spread is under 10%
throughout, the numbers reproduce across three separate runs to within a few
per cent, and the gaps are 6× to 31×.

`wide c`, int64, 1000-entry dictionary:

| values | CPU `gather_into` | GPU resident dict | GPU + host bounds check | GPU dict uploaded too | GPU kernel + sync only |
|---:|---:|---:|---:|---:|---:|
| 64 | <1 µs | 337 µs | 301 µs | 583 µs | 186 µs |
| 1 024 | 1 µs | 328 µs | 300 µs | 590 µs | 187 µs |
| 16 384 | 8 µs | 352 µs | 319 µs | 588 µs | 190 µs |
| 65 536 | 34 µs | 396 µs | 378 µs | 637 µs | 204 µs |
| 250 000 | **128 µs** | **787 µs** | 811 µs | 1051 µs | 279 µs |

`wide d`, `BYTE_ARRAY`, 997-entry dictionary:

| values | CPU `gather_into` | GPU resident dict | GPU dict uploaded too |
|---:|---:|---:|---:|
| 1 024 | 1 µs | 635 µs | 894 µs |
| 16 384 | 22 µs | 908 µs | 1145 µs |
| 65 536 | 91 µs | 1845 µs | 2069 µs |
| 250 000 | **350 µs** | **5867 µs** | 6078 µs |

**The crossover.** A two-point fit over each sweep gives a fixed cost and a
marginal cost per value:

| path | fixed | per value | crossover |
|---|---:|---:|---|
| CPU `gather_into` (int64) | 0 | 0.51 ns | baseline |
| GPU resident dict (int64) | 337 µs | 1.80 ns | **never** |
| GPU + host bounds check (int64) | 301 µs | 2.04 ns | **never** |
| GPU kernel + sync only (int64) | 186 µs | 0.37 ns | ~1 330 000 values |
| CPU `gather_into` (`BYTE_ARRAY`) | 0 | 1.40 ns | baseline |
| GPU resident dict (`BYTE_ARRAY`) | 627 µs | 20.97 ns | **never** |

Read that carefully: **there is no crossover for any usable GPU path, at any
size.** The GPU's cost per value is 1.5× to 15× the CPU's, so the curves
diverge. Size does not save it.

The only line that ever crosses is "kernel + synchronize only" — indices
already on the device, output left stranded on the device, no bounds check, no
transfer. That is not an implementation of anything; it is the floor. And it
crosses at **1.33 million values in a single page**. Real pages in this corpus
are about 19 000 values; the *entire* 250 000-value column chunk is 5.3× short
of that floor. One Parquet page would have to be seventy times larger than
anything pyarrow writes before the GPU's compute alone caught up — and it would
still lose the moment you asked for the answer back.

**And the baseline above is not even the one that ships.** The reader uses
`gather_dict_into`, which fuses RLE decoding into the gather and never
materialises the index list. Over the real chunk, page by page:

| | `gather_dict_into` (shipping) | `gather_into` (oracle) | GPU, dict resident |
|---|---:|---:|---:|
| wide `c` — 13 pages, 250k int64 | 207 µs | 168 µs | 4 487 µs |
| wide `d` — 13 pages, 250k strings | 397 µs | 363 µs | 12 439 µs |

**22× and 31× slower** on the operation as the reader actually performs it,
because a 13-page chunk pays the per-page fixed cost thirteen times over.

Why the `BYTE_ARRAY` path is so much worse is worth naming: the offset prefix
sum forces a *second* synchronize per page (the scan's total decides how big
the output buffer must be, and that number has to reach the host before the
byte-copy kernel can be given somewhere to write), the block scan is
Hillis–Steele and does O(n log n) work where the CPU does O(n), and the byte
copy is one thread per value walking a serial byte loop over strings averaging
four bytes — so there is essentially no parallelism inside a value and a great
deal of divergence between them.

## 5. What about the harder stages?

**Dremel assembly (segmented scan).** The scan machinery is already here, and
it is the part that went worst. The `BYTE_ARRAY` offsets scan is the same shape
as a Dremel segmented scan and it costs 20.97 ns/value against the CPU's 1.40 —
before the extra synchronize the variable-size output forces. Assembly also has
to produce *nested* offsets, so the round trip to the host to size each level's
buffer happens once per level, not once per page. It would be worse than this,
and this is already 15× off.

**RLE level decoding.** Structurally hostile. The hybrid RLE/bit-packed stream
is serial by construction — you cannot know where run *k* begins without
decoding runs 0..k-1 — so the GPU would need a separate pass to find run
boundaries before it could do anything in parallel, and the input is a few
kilobytes per page where the fixed cost is 340 µs. `_check_dict_block` in
`parquet.encoding` already notes that the shape of this problem is a widening
fold, and the CPU does it in L1.

Neither is worth attempting while the per-page fixed cost is a third of a
millisecond and the CPU decodes a whole column chunk in a fifth of one.

## 6. Verdict

**No. Do not pursue this.**

Not because the GPU is the wrong tool for a gather — it is a textbook gather,
and the kernel itself is fine: 0.37 ns/value against the CPU's 0.51. The
problem is that a Parquet page is far too small an object to hand to this
stack. Every page costs a ~190 µs synchronize floor plus the staging, and the
CPU has finished the whole chunk in less time than one round trip.

The three things that would have to change first, none of them ours:

1. **A per-dispatch cost measured in microseconds, not hundreds of them.** As
   it stands, `ctx.synchronize()` alone is 145–166 µs on an empty queue.
2. **A real zero-copy path.** Unified memory exists in the hardware and is
   inaccessible from this API — and, worse, silently wrong when you try.
3. **A unit of work far larger than a page.** The only shape where this could
   pay is offloading a whole row group, or many column chunks at once, so that
   one synchronize covers millions of values. That is a different design from
   "GPU-accelerate the gather", and it would need the reader restructured
   around device-resident columns end to end. Nothing in this spike suggests
   that is worth doing either, but it is the only version of the idea that is
   not arithmetically dead on arrival.

What the spike is worth keeping for: the answer to question 1 (the inner loop
shares, the operation does not), the `map_to_host` and `synchronize` numbers,
and the zero-copy result, which is a silent-wrong-answer trap anyone else
walking into `max.gpu` on Metal should know about.

## Layout

| file | what |
|---|---|
| `gather_shared.mojo` | the shared element bodies, the `comptime` target branch, and the GPU-only bounds-check and scan kernels |
| `host_driver.mojo` | the same bodies with `TARGET_CPU` bound — the half that makes the sharing checkable |
| `gpu_gather.mojo` | the host side: staging, launches, download, and `gather_into`'s exact error contract |
| `kernel_cache.mojo` | `cached_enqueue`, from `millfolio/engine` (Apache-2.0, same author) |
| `dictpages.mojo` | real dictionary pages and index streams pulled out of the fixture corpus |
| `parity.mojo` | the gate, the non-vacuity assertions, and the negative control |
| `bench_gpu_gather.mojo` | the measurements and the crossover fit |
| `memprobe.mojo` | zero-copy, `map_to_host`, `synchronize`, kernel compilation |

### Known limits of the spike code

* `BOOLEAN` dictionaries are not implemented on the GPU (legal, pointless,
  nothing in the wild writes one; `gather_dict_into` keeps them on the CPU
  too).
* The two-level offset scan handles at most `SCAN_BLOCK * SCAN_SUMS` = 524 288
  values per call, and raises rather than truncating past that.
* `dictpages.mojo` reads v1 data pages of non-repeated leaves only. That is
  enough for 100 column chunks across the corpus; v2 pages and repeated leaves
  are skipped rather than half-decoded.
