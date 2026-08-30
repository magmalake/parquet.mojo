"""The parity harness: read a fixture with our decoder, compare with pyarrow."""

from avro.json import JsonDoc
from hashes import crc32
from oracle import canon_value, load_oracle, match_value
from parquet import CodecSet, ParquetReader, Table
from std.testing import assert_equal, assert_true


def _column_index(t: Table, name: StringSlice) -> Int:
    for i in range(t.num_columns()):
        if t.name(i) == name:
            return i
    return -1


def check_fixture[
    Codecs: CodecSet
](name: StringSlice, columns: List[String], batch_size: Int) raises -> Int:
    """Read `<name>.parquet` and assert every value matches the oracle.

    Returns the number of values checked. `columns` empty means "all of them".
    """
    return check_path[Codecs](
        String("tests/fixtures/", name), name, columns, batch_size
    )


def check_path[
    Codecs: CodecSet
](
    stem: StringSlice, name: StringSlice, columns: List[String], batch_size: Int
) raises -> Int:
    """The same, for a fixture that is not directly under `tests/fixtures`."""
    var path = String(stem, ".parquet")
    var doc = load_oracle(String(path, ".oracle.json"))
    var r = ParquetReader[Codecs].open(path)
    r.batch_size = batch_size
    if len(columns):
        r.select_columns(columns)
    var t = r.read_table()
    return check_table(doc, t, name, columns)


def check_table(
    doc: JsonDoc, t: Table, name: StringSlice, columns: List[String]
) raises -> Int:
    """Assert a decoded table matches an oracle document, value by value."""
    var root = doc.root
    assert_equal(
        t.num_rows,
        Int(doc.as_int(doc.get(root, "num_rows"))),
        String(name, ": row count"),
    )

    var oracle_cols = doc.get(root, "columns")
    var checked = 0
    for c in range(doc.len_of(oracle_cols)):
        var oc = doc.child(oracle_cols, c)
        var cname = doc.as_string(doc.get(oc, "name"))
        if len(columns):
            var wanted = False
            for w in columns:
                if w == cname:
                    wanted = True
            if not wanted:
                continue
        var ci = _column_index(t, cname)
        if ci < 0:
            raise Error(String(name, ": we did not decode column '", cname, "'"))

        var values = doc.get(oc, "values")
        var explicit = doc.as_bool(doc.get(oc, "explicit"))
        var n_expect = Int(doc.as_int(doc.get(oc, "num_values")))
        assert_equal(n_expect, t.num_rows, String(name, ".", cname, ": value count"))

        # Every value the oracle spells out, one by one.
        var row = 0
        var limit = doc.len_of(values)
        for b in range(len(t.batches)):
            ref batch = t.batches[b]
            var node = batch.roots[ci]
            for i in range(batch.num_rows):
                if row >= limit:
                    break
                match_value(
                    doc,
                    doc.child(values, row),
                    batch.arena,
                    node,
                    i,
                    String(name, ".", cname, "[", row, "]"),
                )
                checked += 1
                row += 1

        # …and a CRC32 over the canonical text of *all* of them.
        var canon = String()
        for b in range(len(t.batches)):
            ref batch = t.batches[b]
            var node = batch.roots[ci]
            for i in range(batch.num_rows):
                canon_value(batch.arena, node, i, canon)
        var got = crc32(canon.as_bytes())
        var want_hex = doc.as_string(doc.get(oc, "digest"))
        var want: UInt32 = 0
        for k in range(want_hex.byte_length()):
            var ch = want_hex.as_bytes()[k]
            var d = (
                Int(ch) - 48 if ch <= 57 else Int(ch) - 87
            )
            want = (want << 4) | UInt32(d)
        assert_equal(
            got,
            want,
            String(
                name,
                ".",
                cname,
                ": CRC32 over every value (",
                "explicit" if explicit else "digest-only",
                ")",
            ),
        )
        checked += 1
    return checked
