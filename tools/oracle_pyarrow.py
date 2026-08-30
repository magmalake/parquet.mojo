#!/usr/bin/env python3
"""Dump every value of every column of every fixture, as read by pyarrow.

pyarrow is the oracle: `tests/test_parquet.mojo` reproduces these files with
our own decoder and asserts equality value by value.

Scalars are rendered as JSON *strings* so nothing is lost to JSON's number
model — 64-bit unsigned values, 128-bit decimals and the exact bit pattern of
a float all survive. Nulls stay JSON `null`, lists become JSON arrays, structs
become JSON objects and maps become arrays of `[key, value]` pairs.

  bool            "true" / "false"
  int8..int64     decimal, signed
  uint8..uint64   decimal, unsigned
  float16/32/64   the IEEE-754 bits of the value widened to double, 16 hex digits
  utf8            the string itself
  binary / flba   lowercase hex
  decimal         the unscaled integer, decimal, signed
  date32          days since epoch
  time/timestamp  the integer in the column's own unit
"""

import decimal
import json
import os
import struct
import sys

import zlib
import uuid as _uuid

import pyarrow as pa
import pyarrow.parquet as pq

# Files bigger than this get a digest instead of every value spelled out, so
# the oracle JSON stays small enough to live in the repository.
EXPLICIT_ROWS = 2000
PREVIEW_ROWS = 200


def canon(v, out):
    """A canonical, unambiguous text for one oracle value (for the digest)."""
    if v is None:
        out.append("N")
    elif isinstance(v, str):
        out.append("S%d:%s" % (len(v.encode()), v))
    elif isinstance(v, list):
        out.append("L%d:" % len(v))
        for x in v:
            canon(x, out)
    elif isinstance(v, dict):
        out.append("O%d:" % len(v))
        for k, x in v.items():
            out.append("S%d:%s" % (len(k.encode()), k))
            canon(x, out)
    else:
        raise SystemExit("oracle: bad canon value %r" % (v,))


OUT = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"


def dbits(x):
    return struct.pack(">d", float(x)).hex()


def render(v, t):
    """Render one non-null pyarrow scalar value of arrow type `t`."""
    if pa.types.is_boolean(t):
        return "true" if v else "false"
    if pa.types.is_integer(t):
        return str(int(v))
    if pa.types.is_floating(t):
        return dbits(v)
    if pa.types.is_decimal(t):
        # scaleb() honours the *context* precision, which defaults to 28
        # significant digits and would silently round a 38-digit decimal.
        return str(int(v.scaleb(t.scale, decimal.Context(prec=80))))
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return str(v)
    if pa.types.is_binary(t) or pa.types.is_large_binary(t) or pa.types.is_fixed_size_binary(t):
        return bytes(v).hex()
    if pa.types.is_date32(t):
        return str(v.toordinal() - 719163) if hasattr(v, "toordinal") else str(int(v))
    if pa.types.is_time32(t) or pa.types.is_time64(t):
        return str(int(v))
    if pa.types.is_timestamp(t):
        return str(int(v))
    if pa.types.is_null(t):
        return None
    raise SystemExit(f"oracle: no rendering for arrow type {t}")


def unwrap(t):
    """Strip extension types down to their storage type."""
    while hasattr(t, "storage_type"):
        t = t.storage_type
    return t


def value(scalar, t):
    """Render a pyarrow scalar (possibly nested) to the oracle's JSON shape."""
    t = unwrap(t)
    if isinstance(scalar, pa.ExtensionScalar):
        if not scalar.is_valid:
            return None
        scalar = scalar.value
    if not scalar.is_valid:
        return None
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return [value(x, t.value_type) for x in scalar.values]
    if pa.types.is_map(t):
        return [[value(k, t.key_type), value(v, t.item_type)] for k, v in zip(*_map_parts(scalar))]
    if pa.types.is_struct(t):
        return {t.field(i).name: value(scalar[t.field(i).name], t.field(i).type) for i in range(t.num_fields)}
    if (
        pa.types.is_date(t)
        or pa.types.is_time(t)
        or pa.types.is_timestamp(t)
        or pa.types.is_duration(t)
    ):
        # `.value` is the integer the column stores, in the column's own unit.
        return str(scalar.value)
    py = scalar.as_py()
    if isinstance(py, _uuid.UUID):
        return py.bytes.hex()
    return render(py, t)


def _map_parts(scalar):
    keys, vals = [], []
    for entry in scalar.values:
        keys.append(entry["key"])
        vals.append(entry["value"])
    return keys, vals


def leaf_stats(cc, arrow_type):
    s = cc.statistics
    if s is None:
        return None
    t = unwrap(arrow_type)
    out = {"null_count": None if s.null_count is None else int(s.null_count),
           "distinct_count": None if s.distinct_count is None else int(s.distinct_count),
           "has_min_max": bool(s.has_min_max)}
    if s.has_min_max:
        try:
            out["min"] = _stat_render(s.min, t)
            out["max"] = _stat_render(s.max, t)
        except Exception:
            out["min"] = None
            out["max"] = None
    return out


def _stat_render(v, t):
    import datetime

    if isinstance(v, datetime.date) and not isinstance(v, datetime.datetime):
        return str((v - datetime.date(1970, 1, 1)).days)
    if isinstance(v, datetime.datetime):
        return None  # units are ambiguous through pyarrow's Python surface
    if isinstance(v, datetime.time):
        return None
    if isinstance(v, bytes):
        return v.hex()
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return dbits(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v
    import decimal as _d

    if isinstance(v, _d.Decimal):
        return (
            str(int(v.scaleb(t.scale, _d.Context(prec=80))))
            if pa.types.is_decimal(t)
            else str(v)
        )
    return None


def flat_leaves(schema):
    """The leaf columns in Parquet order, as (dotted path, arrow type)."""
    out = []

    def walk(prefix, field):
        t = unwrap(field.type)
        name = prefix + field.name
        if pa.types.is_struct(t):
            for i in range(t.num_fields):
                walk(name + ".", t.field(i))
        elif pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
            walk(name + ".", t.value_field)
        elif pa.types.is_map(t):
            walk(name + ".", t.key_field)
            walk(name + ".", t.item_field)
        else:
            out.append((name, t))

    for f in schema:
        walk("", f)
    return out


def main():
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".parquet"))
    for name in files:
        path = os.path.join(OUT, name)
        pf = pq.ParquetFile(path)
        table = pf.read()
        md = pf.metadata
        doc = {
            "file": name,
            "pyarrow": pa.__version__,
            "num_rows": md.num_rows,
            "num_columns": md.num_columns,
            "num_row_groups": md.num_row_groups,
            "created_by": md.created_by,
            "row_groups": [],
            "columns": [],
        }
        for r in range(md.num_row_groups):
            rg = md.row_group(r)
            first = rg.column(0)
            start = first.dictionary_page_offset if first.dictionary_page_offset else first.data_page_offset
            doc["row_groups"].append(
                {"num_rows": rg.num_rows, "total_byte_size": rg.total_byte_size, "start_offset": start}
            )
        leaves = flat_leaves(table.schema)
        doc["leaves"] = [
            {
                "path": p,
                "physical": md.row_group(0).column(i).physical_type,
                "stats": [leaf_stats(md.row_group(r).column(i), t) for r in range(md.num_row_groups)],
            }
            for i, (p, t) in enumerate(leaves)
        ]
        for i, field in enumerate(table.schema):
            col = table.column(i)
            vals = []
            for chunk in col.chunks:
                for j in range(len(chunk)):
                    vals.append(value(chunk[j], field.type))
            entry = {"name": field.name, "arrow_type": str(field.type), "num_values": len(vals)}
            parts = []
            for v in vals:
                canon(v, parts)
            entry["digest"] = "%08x" % (zlib.crc32("".join(parts).encode()) & 0xFFFFFFFF)
            entry["null_count"] = sum(1 for v in vals if v is None)
            entry["values"] = vals if len(vals) <= EXPLICIT_ROWS else vals[:PREVIEW_ROWS]
            entry["explicit"] = len(entry["values"]) == len(vals)
            doc["columns"].append(entry)
        with open(path + ".oracle.json", "w") as fh:
            json.dump(doc, fh, separators=(",", ":"), sort_keys=False)
        print(f"  {name + '.oracle.json':40s} {os.path.getsize(path + '.oracle.json'):>9,d} bytes")


main()
