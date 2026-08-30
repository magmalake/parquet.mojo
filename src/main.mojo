"""`parquet-mojo` — inspect a Parquet file from the command line.

```console
$ parquet-mojo schema tests/fixtures/nested.parquet
$ parquet-mojo meta tests/fixtures/pageindex.parquet
$ parquet-mojo cat tests/fixtures/primitives.parquet --columns i64,s --limit 5
$ parquet-mojo cat tests/fixtures/nested.parquet --json --limit 2
```
"""

from parquet import ParquetReader, ParquetSchema
from parquet.display import json_escape, value_json, value_text
from std.sys import argv
from thrift import CompressionCodec, Encoding, Type

comptime USAGE = """usage: parquet-mojo <command> [options] <file.parquet>

commands:
  schema   print the Parquet schema and the Arrow types it maps to
  meta     print file, row group and column chunk metadata
  cat      print rows

options:
  --columns a,b   only these top-level columns
  --field-ids 1,2 only the columns with these Parquet field ids
  --limit N       stop after N rows (cat only)
  --json          emit JSON
  --no-crc        do not verify page checksums
"""


def _split(text: StringSlice, sep: StringSlice) -> List[String]:
    var out = List[String]()
    for part in text.split(sep):
        if part:
            out.append(String(part))
    return out^


def _indent(n: Int) -> String:
    var s = String()
    for _ in range(n):
        s += "  "
    return s^


def print_schema_tree(s: ParquetSchema, root: Int, depth: Int) raises:
    """Print one Arrow field and everything under it, depth first."""
    var stack: List[Tuple[Int, Int]] = [(root, depth)]
    while len(stack):
        var top = stack.pop()
        ref f = s.fields[top[0]]
        var line = String(_indent(top[1] + 1), f.name, ": ", String(f.type))
        if not f.nullable:
            line += " not null"
        if f.field_id >= 0:
            line += String(" field_id=", f.field_id)
        if f.leaf >= 0:
            ref lc = s.leaves[f.leaf]
            line += String(
                " <- ",
                Type(lc.physical).name(),
                " def=",
                lc.max_def,
                " rep=",
                lc.max_rep,
            )
        print(line)
        for k in range(len(f.children)):
            stack.append((f.children[len(f.children) - 1 - k], top[1] + 1))


def cmd_schema(mut r: ParquetReader, as_json: Bool) raises:
    if as_json:
        var out = String('{"num_rows":', r.num_rows(), ',"columns":[')
        for i in range(len(r.schema.roots)):
            if i:
                out += ","
            ref f = r.schema.fields[r.schema.roots[i]]
            out += String(
                '{"name":',
                json_escape(f.name),
                ',"type":',
                json_escape(String(f.type)),
                ',"field_id":',
                f.field_id,
                "}",
            )
        out += "]}"
        print(out)
        return
    print("message schema {")
    for root in r.schema.roots:
        print_schema_tree(r.schema, root, 0)
    print("}")
    print()
    print("leaf columns:")
    for i in range(len(r.schema.leaves)):
        ref lc = r.schema.leaves[i]
        print(
            String(
                "  [",
                i,
                "] ",
                lc.dotted(),
                " ",
                Type(lc.physical).name(),
                " -> ",
                String(lc.arrow),
                " max_def=",
                lc.max_def,
                " max_rep=",
                lc.max_rep,
            )
        )


def cmd_meta(mut r: ParquetReader, as_json: Bool) raises:
    if as_json:
        var out = String(
            '{"num_rows":',
            r.num_rows(),
            ',"num_row_groups":',
            r.num_row_groups(),
            ',"created_by":',
            json_escape(r.created_by()),
            ',"split_offsets":[',
        )
        var so = r.split_offsets()
        for i in range(len(so)):
            if i:
                out += ","
            out += String(so[i])
        out += "]}"
        print(out)
        return
    print("rows:       ", r.num_rows())
    print("row groups: ", r.num_row_groups())
    print("columns:    ", len(r.schema.leaves))
    print("created by: ", r.created_by())
    var kv = r.key_value_metadata()
    if len(kv):
        print("key/value:")
        for e in kv:
            print(String("  ", e[0], " (", e[1].byte_length(), " bytes)"))
    var so = r.split_offsets()
    var line = String("split offsets:")
    for v in so:
        line += String(" ", v)
    print(line)
    for g in range(r.num_row_groups()):
        ref rg = r.meta.row_groups[g]
        print(
            String(
                "\nrow group ",
                g,
                ": ",
                rg.num_rows,
                " rows, ",
                rg.total_byte_size,
                " bytes uncompressed",
            )
        )
        for c in range(len(rg.columns)):
            if not rg.columns[c].meta_data:
                continue
            ref cm = rg.columns[c].meta_data.value()
            var encs = String()
            for e in cm.encodings:
                if encs:
                    encs += ","
                encs += e.name()
            var text = String(
                "  [",
                c,
                "] ",
                r.schema.leaves[c].dotted(),
                " ",
                cm.type_.name(),
                " ",
                cm.codec.name(),
                " ",
                encs,
                " values=",
                cm.num_values,
                " compressed=",
                cm.total_compressed_size,
            )
            var st = r.statistics(g, c)
            if st.has_min_max:
                text += String(" min=", String(st.min), " max=", String(st.max))
            if st.has_null_count:
                text += String(" nulls=", st.null_count)
            if cm.bloom_filter_offset:
                text += String(" bloom@", cm.bloom_filter_offset.value())
            print(text)


def cmd_cat(mut r: ParquetReader, limit: Int, as_json: Bool) raises:
    var printed = 0
    var names = List[String]()
    var t = r.read_table()
    for i in range(t.num_columns()):
        names.append(t.name(i))
    if not as_json:
        var head = String()
        for i in range(len(names)):
            if i:
                head += "\t"
            head += names[i]
        print(head)
    for b in range(len(t.batches)):
        ref batch = t.batches[b]
        for i in range(batch.num_rows):
            if limit >= 0 and printed >= limit:
                return
            var line = String("{") if as_json else String()
            for c in range(batch.num_columns()):
                if c:
                    line += "," if as_json else "\t"
                if as_json:
                    line += json_escape(names[c])
                    line += ":"
                    line += value_json(batch.arena, batch.roots[c], i)
                else:
                    line += value_text(batch.arena, batch.roots[c], i)
            if as_json:
                line += "}"
            print(line)
            printed += 1


def main() raises:
    var args = argv()
    if len(args) < 3:
        print(USAGE)
        return
    var command = String(args[1])
    var path = String()
    var columns = List[String]()
    var field_ids = List[Int32]()
    var limit = -1
    var as_json = False
    var crc = True
    var i = 2
    while i < len(args):
        var a = String(args[i])
        if a == "--columns" and i + 1 < len(args):
            columns = _split(String(args[i + 1]), ",")
            i += 1
        elif a == "--field-ids" and i + 1 < len(args):
            for part in _split(String(args[i + 1]), ","):
                field_ids.append(Int32(Int(part)))
            i += 1
        elif a == "--limit" and i + 1 < len(args):
            limit = Int(String(args[i + 1]))
            i += 1
        elif a == "--json":
            as_json = True
        elif a == "--no-crc":
            crc = False
        elif a == "--help" or a == "-h":
            print(USAGE)
            return
        else:
            path = a
        i += 1
    if not path:
        print(USAGE)
        return

    var r = ParquetReader.open(path)
    r.verify_crc = crc
    if len(columns):
        r.select_columns(columns)
    if len(field_ids):
        r.select_field_ids(field_ids)
    if command == "schema":
        cmd_schema(r, as_json)
    elif command == "meta":
        cmd_meta(r, as_json)
    elif command == "cat":
        cmd_cat(r, limit, as_json)
    else:
        print(String("parquet-mojo: unknown command '", command, "'"))
        print(USAGE)
