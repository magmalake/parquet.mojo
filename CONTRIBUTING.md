# Contributing

1. Fork, branch off `main`.
2. `pixi run test` on both toolchains (`pixi run -e stable test` for 1.1.0).
   The codec tests are `pixi run -e codecs-stable test-codecs`.
3. `pixi run -e fmt format-check` before committing.
4. If you change the public API, update `README.md`.
5. Open a PR. Small focused PRs over big ones.

A change made for speed should say how much faster, measured as below, in the
PR description.

## Measuring performance

Four tools, from "did it get faster" to "where does the time go":

| command | answers |
|---|---|
| `pixi run -e bench bench` | how long each read and write takes, with p50/p90 — the before/after number |
| `pixi run profile` | which stage of a read or a write the time goes to |
| `pixi run -e codecs-stable profile-scan` | the same for a column projection over a big file |
| `pixi run bench-pyarrow` | pyarrow's time on the same files, for comparison |

The profilers build against the sibling repositories' sources
(`-I ../thrift.mojo/src` and so on), so check those out next to this one:
`arrow-mlake.mojo`, `threads.mojo`, `thrift.mojo`, `hashes.mojo`,
`snappy.mojo`, `deflate.mojo`, and for `profile-scan` also `zstd.mojo`,
`lz4.mojo` and `brotli.mojo`. The benchmark does not need them; it takes the
siblings as pinned packages.

### One-time setup

The two 1M-row benchmarks read `build/bench-wide.parquet`, which is generated
rather than committed. It needs pyarrow, which no pixi environment has:

```sh
uv run --no-project --with 'pyarrow>=21,<26' python tools/bench_pyarrow.py --make
```

Without it the benchmark stops before running anything. To skip those two
instead, add `--skip bench_read_wide bench_write_wide` to every `bench` command
below.

### Before and after

Take a baseline on `main` *before* changing anything, then the same run on
your branch:

```sh
git switch main
pixi run -e bench bench -- --out build/before.json --strict

git switch my-branch
pixi run -e bench bench -- --out build/after.json --strict
```

The benchmark compiles this repository from `src/`, so it measures whatever is
checked out. The sibling tins come in as pinned packages and are identical in
both runs; the only difference between them is your change.

`--strict` exits non-zero when the harness cannot vouch for the run — the
machine's own speed drifted during it, or the load average was high. When that
happens, close what else is running and run it again rather than keeping the
numbers.

Then compare the medians:

```sh
jq -rn --slurpfile a build/before.json --slurpfile b build/after.json '
  ($a[0].results | map({(.name): .median_ns}) | add) as $old |
  $b[0].results[] | select($old[.name]) |
  "\(.name)\t\($old[.name]/1e6|.*1000|round/1000) ms → \(.median_ns/1e6|.*1000|round/1000) ms\t\((.median_ns/$old[.name]-1)*100|round)%"' \
  | column -t -s $'\t'
```

```
bench_read_big   2.417 ms → 2.431 ms    1%
bench_write_big  11.025 ms → 11.142 ms  1%
```

That example is the *same* code measured twice: about 1% is the noise floor on
an idle M4. Treat a change inside a few percent as no change. Each JSON result
also carries `mean_ns`, `p90_ns` and the per-repetition `runs_ns`, and the file
records the machine it ran on under `host`.

While iterating, run only the benchmarks you are working on, which is much
faster; take the full before/after at the end:

```sh
pixi run -e bench bench -- --only bench_read_big bench_write_big
```

`pixi run -e bench ./build/bench-parquet --list` prints every benchmark name.
The `_w1`/`_w2`/`_w4`/… variants time a read with that many worker threads;
everything else is one thread.

Keep the conditions the same for both runs: same machine, plugged in, nothing
heavy running.

### Where the time goes

```sh
pixi run profile
```

Splits one read and one write into their stages — footer, page headers,
decompression, levels, values, dictionary gather, Arrow assembly on the read
side; shredding, dictionary, levels, values, statistics on the write side —
with each stage's share of the total. Aim an optimisation at the stage that
dominates. The docstring at the top of `bench/profile_parquet.mojo` says
exactly what each row covers.

For a projected read over a larger file — a few columns out of many, which is
what a query engine asks for:

```sh
pixi run -e codecs-stable profile-scan                     # builds build/profile-scan
pixi run -e codecs-stable ./build/profile-scan build/bench-wide.parquet --select a,b --repeat 5
```

Each stage row carries the bytes it moved and the bandwidth that implies. A
stage already running at memory bandwidth cannot be made faster, only skipped.

`profile-scan` needs `codecs-stable`, not `codecs`: the codec tins arrive
precompiled by the stable compiler, and the nightly one refuses them.

### Against pyarrow

```sh
pixi run bench-pyarrow
```

Needs [uv](https://docs.astral.sh/uv/). Times pyarrow on the same fixtures,
single-threaded and threaded. Compare its single-threaded numbers with this
repository's (non-`_w`) benchmarks; the README's tables were taken that way.

### What CI measures

Every push to `main` runs the benchmark on a GitHub runner and appends it to
[magmalake.github.io/parquet.mojo/benchmarks](https://magmalake.github.io/parquet.mojo/benchmarks/).
That runner is slower and noisier than a laptop and skips the two wide
benchmarks, so it shows trends over time, not the effect of one change.
Measure a change locally.
