#!/usr/bin/env python3
"""What encoding, and how many bytes, every column chunk we wrote came out as.

    python tools/encoding_report.py build/encodings           # one tree
    python tools/encoding_report.py build/encodings ref/dir   # two, compared

`tools/encoding_corpus.mojo` writes the corpus; this reads it back with pyarrow
and reports, per file and column, the encodings chosen, whether the chunks
carry a dictionary page, and the compressed and uncompressed bytes. With two
directories it prints the same table for both and marks every row that moved,
so a change to the writer's dictionary rule can be judged file by file instead
of by whether the tests still pass.

Exit status is 1 only when a directory is missing or unreadable — a difference
is what the tool is for, not a failure.
"""

import glob
import os
import sys

import pyarrow.parquet as pq


def summarise(directory):
    """{(file, column): (encodings, dict_chunks, chunks, compressed, uncompressed)}"""
    rows = {}
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    if not files:
        sys.exit(f"{directory}: no .parquet files")
    for path in files:
        name = os.path.basename(path)[: -len(".parquet")]
        md = pq.ParquetFile(path).metadata
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            for c in range(rg.num_columns):
                col = rg.column(c)
                key = (name, col.path_in_schema)
                enc, ndict, n, comp, uncomp = rows.get(
                    key, (set(), 0, 0, 0, 0)
                )
                enc.update(col.encodings)
                rows[key] = (
                    enc,
                    ndict + (1 if col.dictionary_page_offset else 0),
                    n + 1,
                    comp + col.total_compressed_size,
                    uncomp + col.total_uncompressed_size,
                )
    return rows


def encoding_of(entry):
    """The value encoding, dictionary or not, with RLE level encoding dropped."""
    enc = sorted(e for e in entry[0] if e != "RLE")
    if "RLE_DICTIONARY" in enc:
        return "RLE_DICTIONARY"
    return "+".join(enc) if enc else "-"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    mine = summarise(sys.argv[1])
    ref = summarise(sys.argv[2]) if len(sys.argv) > 2 else None

    if ref is None:
        print(f"{'file':<22} {'column':<20} {'encoding':<16} {'chunks':>7} "
              f"{'dict':>5} {'compressed':>12} {'uncompressed':>13}")
        for (f, c), e in sorted(mine.items()):
            print(f"{f:<22} {c:<20} {encoding_of(e):<16} {e[2]:>7} "
                  f"{e[1]:>5} {e[3]:>12,} {e[4]:>13,}")
        return

    print(f"{'file':<22} {'column':<20} {'ref encoding':<16} "
          f"{'new encoding':<16} {'ref bytes':>12} {'new bytes':>12} "
          f"{'delta':>10}")
    changed = 0
    total_ref = 0
    total_new = 0
    for key in sorted(set(mine) | set(ref)):
        a = ref.get(key)
        b = mine.get(key)
        ea = encoding_of(a) if a else "(absent)"
        eb = encoding_of(b) if b else "(absent)"
        ba = a[3] if a else 0
        bb = b[3] if b else 0
        total_ref += ba
        total_new += bb
        mark = ""
        if ea != eb:
            mark = "  ENCODING CHANGED"
            changed += 1
        elif ba != bb:
            mark = "  size changed"
            changed += 1
        print(f"{key[0]:<22} {key[1]:<20} {ea:<16} {eb:<16} "
              f"{ba:>12,} {bb:>12,} {bb - ba:>+10,}{mark}")
    print(f"\n{len(set(mine) | set(ref))} column(s), {changed} changed; "
          f"total compressed {total_ref:,} -> {total_new:,} "
          f"({total_new - total_ref:+,})")


if __name__ == "__main__":
    main()
