"""Run the apache/parquet-testing corpus through the reader.

`data/` holds files every implementation should read; `bad_data/` holds files
every implementation should *reject*. The corpus is data only — each project
writes its own runner, so this is ours.

```console
pixi run -e codecs conformance /path/to/parquet-testing
```

Prints one line per file and a summary. Exit status is non-zero if any file
in `data/` failed to read or any file in `bad_data/` was accepted, so this is
usable as a check.
"""

from std.sys import argv
from std.os.path import exists
from std.os import listdir
from std.builtin.sort import sort
from parquet import ParquetReader
from parquet.ext_full import AllCodecs


# Reading the footer is not enough: a file can have a valid schema and fail in
# page decode, which is where most of the interesting bugs live. Materialising
# the table forces every page of every column through the decoders.
def _read_fully(path: String) raises -> Int:
    var r = ParquetReader[AllCodecs].open(path)
    var t = r.read_table()
    var cells = 0
    for b in range(len(t.batches)):
        ref batch = t.batches[b]
        cells += batch.num_rows * batch.num_columns()
    return cells


def _expected_unreadable(name: String) -> String:
    """Why a `data/` file is not expected to read, or "" if it should.

    Two of these are deliberately corrupt and rejecting them is the correct
    answer; two need a feature we have not built. Naming them here rather than
    folding them into the pass count means a *new* failure stands out instead
    of blending into a known-bad list.
    """
    if name == "datapage_v1-corrupt-checksum.parquet":
        return "corrupt CRC, rejected on purpose"
    if name == "rle-dict-uncompressed-corrupt-checksum.parquet":
        return "corrupt CRC, rejected on purpose"
    if name == "alp_extended.zstd.parquet":
        return "ALP encoding not implemented"
    if name == "large_string_map.brotli.parquet":
        return "BROTLI codec not implemented"
    return String()


def _known_accepted(name: String) -> String:
    """Why we accept a `bad_data/` file Arrow rejects, or "" if we reject it.

    Being more permissive than Arrow is a real difference, and it belongs in
    the report rather than hidden in a pass.
    """
    if name == "ARROW-GH-43605.parquet":
        return String(
            "dictionary indices use bit width 0; every index decodes to 0,"
            " which is in range, and nothing distinguishes that from a page"
            " that only references the first dictionary entry"
        )
    return String()


def _parquet_files(dir: String) raises -> List[String]:
    var out = List[String]()
    if not exists(dir):
        return out^
    for name in listdir(dir):
        if name.endswith(".parquet"):
            out.append(name)
    # Stable order, so two runs are diffable.
    sort(out)
    return out^


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: conformance <path to parquet-testing checkout>")
        raise Error("missing corpus path")
    var root = String(args[1])

    var good_dir = root + "/data"
    var bad_dir = root + "/bad_data"

    var good_pass = 0
    var good_fail = 0
    var good_known = 0
    var bad_rejected = 0
    var bad_accepted = 0
    var bad_known = 0

    print("== data/ (must read) ==")
    for name in _parquet_files(good_dir):
        var note = _expected_unreadable(name)
        try:
            var cells = _read_fully(good_dir + "/" + name)
            if note:
                # It started working: the table is now the thing that is wrong.
                print("UNEXPECTED PASS ", name, " -- listed as: ", note)
                good_fail += 1
            else:
                print("PASS ", name, " (", cells, " cells)")
                good_pass += 1
        except e:
            if note:
                print("known   ", name, " :: ", note)
                good_known += 1
            else:
                print("FAIL ", name, " :: ", String(e))
                good_fail += 1

    print()
    print("== bad_data/ (must be rejected) ==")
    for name in _parquet_files(bad_dir):
        var note = _known_accepted(name)
        try:
            _ = _read_fully(bad_dir + "/" + name)
            if note:
                print("known   ", name, " accepted :: ", note)
                bad_known += 1
            else:
                print("ACCEPTED ", name, " -- should have been rejected")
                bad_accepted += 1
        except:
            if note:
                print("UNEXPECTED REJECT ", name, " -- listed as accepted")
                bad_accepted += 1
            else:
                print("rejected ", name)
                bad_rejected += 1

    print()
    print(
        "data/     :", good_pass, "read,", good_known,
        "known-unreadable,", good_fail, "failed",
    )
    print(
        "bad_data/ :", bad_rejected, "rejected,", bad_known,
        "known-accepted,", bad_accepted, "wrongly accepted",
    )

    if good_fail or bad_accepted:
        raise Error("conformance failures")
