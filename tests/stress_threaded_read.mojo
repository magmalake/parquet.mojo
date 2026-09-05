"""The threaded read path, hammered — the body the sanitizers run.

    pixi run -e stable stress         # plain, no sanitizer
    pixi run -e stable stress-tsan    # macOS: mojo build --sanitize thread
    pixi run -e stable stress-asan    # Linux: mojo build --sanitize address

`test_num_workers_is_bit_identical` proves the *output* does not depend on the
worker count. That is not the same as proving there is no race: a race can be
benign on one run and wrong on the next, and a fingerprint comparison only ever
sees the run it got. This is the other half — a small, repetitive workload over
the same code, built under ThreadSanitizer, where the tool reports the race
whether or not it changed the answer this time.

Deliberately narrow: a handful of fixtures with different shapes (nulls,
nesting, dictionaries, many pages, several row groups), read over and over at
several worker counts, plus the error path, which is the one place a worker
writes a `String` rather than a decoded buffer. TSan slows a program down by
five to fifteen times, so the round count is small by default and
`STRESS_ROUNDS` raises it.

Environment:

- `STRESS_ROUNDS` — passes over the fixture list (default 8).
"""

from fingerprint import read_error, read_fingerprint
from parquet import DefaultCodecs, ParquetReader
from std.os import getenv
from std.testing import assert_equal, assert_true
from thrift import read_parquet_file


comptime FIXTURES = "tests/fixtures/"


def _stress_fixtures() -> List[String]:
    """Shapes that between them cover every branch of `_decode_leaf`."""
    return [
        String("big"),
        String("nested"),
        String("legacy_list"),
        String("allnull"),
        String("manypages"),
        String("encodings"),
        String("prune"),
    ]


def _rounds() raises -> Int:
    var v = getenv("STRESS_ROUNDS")
    if v == "":
        return 8
    return Int(v)


def _corrupt_big() raises -> List[UInt8]:
    """`big.parquet` with the first page of column 0 scribbled over."""
    var data = read_parquet_file(String(FIXTURES, "big.parquet"))
    var r = ParquetReader[DefaultCodecs].from_span(Span(data))
    ref cm = r.meta.row_groups[0].columns[0].meta_data.value()
    var at = Int(cm.data_page_offset)
    if cm.dictionary_page_offset:
        var d = Int(cm.dictionary_page_offset.value())
        if d > 0 and d < at:
            at = d
    for k in range(at, at + 48):
        data[k] = 0xA5
    return data^


def main() raises:
    var rounds = _rounds()
    var workers: List[Int] = [2, 4, 8, 10]
    var names = _stress_fixtures()
    print("stress: ", rounds, " round(s) over ", len(names), " fixture(s)")

    var reads = 0
    for f in names:
        var path = String(FIXTURES, f, ".parquet")
        # The same batch size on both sides: batching decides where batch
        # boundaries fall and so is part of the fingerprint, and this is a
        # test about worker counts and not about `batch_size`.
        var want = read_fingerprint[DefaultCodecs](path, 1, 4096)
        for _ in range(rounds):
            for w in range(len(workers)):
                assert_equal(
                    read_fingerprint[DefaultCodecs](path, workers[w], 4096),
                    want,
                    String(f, " at ", workers[w], " workers"),
                )
                reads += 1

    # The error path: several tasks racing to write into their own error slots
    # while the rest write decoded buffers.
    var bad = _corrupt_big()
    var expected = read_error[DefaultCodecs](Span(bad), 1)
    assert_true(expected != "", "the corrupt fixture read cleanly")
    for _ in range(rounds):
        for w in range(len(workers)):
            assert_equal(
                read_error[DefaultCodecs](Span(bad), workers[w]),
                expected,
                String("corrupt read at ", workers[w], " workers"),
            )
            reads += 1

    print("stress: ", reads, " threaded read(s), all identical")
