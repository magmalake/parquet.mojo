# Contributing

1. Branch off `main`.
2. `pixi run test` and `pixi run -e stable test`. Codec tests:
   `pixi run -e codecs-stable test-codecs`.
3. `pixi run -e fmt format-check`.
4. Open a PR. A performance change should include before/after numbers,
   measured as below.

## Measuring performance

The profilers need the sibling repositories checked out next to this one:
`arrow-mlake.mojo`, `threads.mojo`, `thrift.mojo`, `hashes.mojo`,
`snappy.mojo`, `deflate.mojo`, and for `profile-scan` also `zstd.mojo`,
`lz4.mojo`, `brotli.mojo`. The benchmark does not.

### Setup, once

```sh
uv run --no-project --with 'pyarrow>=21,<26' python tools/bench_pyarrow.py --make
```

This writes `build/bench-wide.parquet`, which the wide benchmarks read.

### Before and after

Build both versions:

```sh
git switch main
pixi run -e bench bench -- --list > /dev/null && cp build/bench-parquet build/bench-before
git switch my-branch
pixi run -e bench bench -- --list > /dev/null && cp build/bench-parquet build/bench-after
```

Run them alternately, three times each (about four minutes):

```sh
for i in 1 2 3; do
  for v in before after; do
    pixi run -e bench ./build/bench-$v --strict --out build/$v-$i.json --only \
      bench_read_big bench_read_prune bench_read_encodings bench_read_v2pages \
      bench_read_wide bench_write_big bench_write_wide
  done
done
```

Compare:

```sh
python3 tools/bench_compare.py build/before-*.json -- build/after-*.json
```

```
bench_read_big            2.471 ms      2.500 ms    +1.2%
bench_read_prune          0.130 ms      0.132 ms    +1.5%
...
```

That output is the same code on both sides: differences under 2% are noise.

- If `--strict` fails a run, run it again; don't use its numbers.
- Close other applications and stay on AC power.
- Add the `_w2`…`_w10` benchmarks only when changing threading; they are
  noisier.
- `pixi run -e bench ./build/bench-after --list` prints every benchmark name.

### Where the time goes

```sh
pixi run profile
```

Per-stage times for one read and one write.

```sh
pixi run -e codecs-stable profile-scan
pixi run -e codecs-stable ./build/profile-scan build/bench-wide.parquet --select a,b
```

Per-stage times and bandwidth for reading some columns of a large file.

### Against pyarrow

```sh
pixi run bench-pyarrow
```

Needs [uv](https://docs.astral.sh/uv/).
