"""Per-stage decode profile of a *projected* scan over a real file.

    profile-scan <file.parquet> [--select a,b,c] [--repeat N]

`bench/profile_parquet.mojo` profiles the whole of a small fixture. This
profiles the shape a query engine actually asks for: a handful of columns out
of a wide file, on a file big enough that the per-stage costs are not noise.

Every stage row carries the bytes it moved and the bandwidth that implies,
because a stage already running at memory bandwidth cannot be made faster —
only skipped. `slurp` is there for the same reason: the reader takes whole
file bytes, so a four-column projection out of a nineteen-column file still
pays to bring the whole file in.

Percentiles, not best-of-N: p50 is the headline and p90 is printed beside it,
so a run that the machine interfered with is visible rather than hidden.
"""

from std.sys import argv
from std.time import perf_counter_ns

from parquet import ColumnData, ParquetReader, PhysBuffer
from parquet.encoding import (
    decode_plain,
    decode_plain_into,
    gather_dict_into,
    physical_kind,
    physical_width,
)
from parquet_full import AllCodecs
from parquet.page import (
    _decode_values,
    _read_levels,
    _take_defs,
    chunk_start,
)
from parquet.schema import LeafColumn, build_schema
from thrift import (
    ColumnMetaData,
    Encoding,
    PageType,
    read_footer,
    read_page_header,
    read_parquet_file,
)


struct Stages(Copyable, Defaultable, Movable):
    """Nanoseconds per stage, and the bytes each stage moved."""

    var slurp: Int
    var footer: Int
    var hdr: Int
    var decomp: Int
    var levels: Int
    var values: Int
    var gather: Int
    var concat: Int
    var index: Int
    var assemble: Int
    var total: Int
    var pages: Int
    var comp_bytes: Int
    var raw_bytes: Int
    var out_bytes: Int
    var file_bytes: Int
    var alloc_bytes: Int
    var alloc_count: Int

    def __init__(out self):
        self.slurp = 0
        self.footer = 0
        self.hdr = 0
        self.decomp = 0
        self.levels = 0
        self.values = 0
        self.gather = 0
        self.concat = 0
        self.index = 0
        self.assemble = 0
        self.total = 0
        self.pages = 0
        self.comp_bytes = 0
        self.raw_bytes = 0
        self.out_bytes = 0
        self.file_bytes = 0
        self.alloc_bytes = 0
        self.alloc_count = 0

    def __init__(out self, *, copy: Self):
        self.slurp = copy.slurp
        self.footer = copy.footer
        self.hdr = copy.hdr
        self.decomp = copy.decomp
        self.levels = copy.levels
        self.values = copy.values
        self.gather = copy.gather
        self.concat = copy.concat
        self.index = copy.index
        self.assemble = copy.assemble
        self.total = copy.total
        self.pages = copy.pages
        self.comp_bytes = copy.comp_bytes
        self.raw_bytes = copy.raw_bytes
        self.out_bytes = copy.out_bytes
        self.file_bytes = copy.file_bytes
        self.alloc_bytes = copy.alloc_bytes
        self.alloc_count = copy.alloc_count

    def __init__(out self, *, deinit move: Self):
        self = Self(copy=move)


def _ms(ns: Int) -> String:
    var us = ns // 1000
    return String(us // 1000, ".", (us % 1000) // 100, (us % 100) // 10)


def _pad(var text: String, width: Int) -> String:
    var out = String()
    for _ in range(width - text.byte_length()):
        out += " "
    return out + text


def _gbs(bytes: Int, ns: Int) -> String:
    """GB/s to one decimal, or a dash when the stage did nothing."""
    if ns <= 0 or bytes <= 0:
        return String("-")
    var tenths = (bytes * 10_000) // ns // 1000
    return String(tenths // 10, ".", tenths % 10, " GB/s")


def _row(
    name: StringSlice, ns: Int, total: Int, bytes: Int, note: StringSlice
) -> String:
    var pct = 0
    if total > 0:
        pct = (ns * 1000) // total
    var body = String(
        "  ",
        name,
        _pad(String(), 11 - name.byte_length()),
        _pad(_ms(ns), 8),
        " ms  ",
        _pad(String(pct // 10, ".", pct % 10), 5),
        " %  ",
        _pad(_gbs(bytes, ns), 9),
    )
    if note.byte_length() > 0:
        body += String("  ", note)
    return body


def _sorted(var samples: List[Int]) -> List[Int]:
    for i in range(1, len(samples)):
        var value = samples[i]
        var j = i - 1
        while j >= 0 and samples[j] > value:
            samples[j + 1] = samples[j]
            j -= 1
        samples[j + 1] = value
    return samples^


def _walk_chunk(
    file: Span[UInt8, _],
    cm: ColumnMetaData,
    leaf: LeafColumn,
    mut st: Stages,
) raises:
    """`read_column_chunk`, with a timer and a byte counter around each stage.
    """
    var values = PhysBuffer(
        physical_kind(leaf.physical),
        physical_width(leaf.physical, leaf.type_length),
    )
    var nv = Int(cm.num_values)
    if nv > 0 and nv < (1 << 30) and values.width > 0:
        if nv * values.width <= (1 << 28):
            values.bytes.reserve(nv * values.width)
    var cd = ColumnData()
    cd.packed = leaf.max_def == 1 and leaf.max_rep == 0
    ref defs = cd.defs
    ref reps = cd.reps
    var offset = chunk_start(cm)
    var limit = offset + Int(cm.total_compressed_size)
    var want = Int(cm.num_values)
    var dict = PhysBuffer()
    var has_dict = False
    var codec = cm.codec.value
    var slots = 0
    var scratch = List[UInt8]()
    var scratch_high = 0

    while slots < want and offset < limit:
        var t0 = perf_counter_ns()
        var hdr = read_page_header(file, offset)
        ref ph = hdr[0]
        var t1 = perf_counter_ns()
        st.hdr += t1 - t0
        st.pages += 1
        var body_at = offset + hdr[1]
        var csize = Int(ph.compressed_page_size)
        var usize = Int(ph.uncompressed_page_size)
        var body = file[body_at : body_at + csize]
        offset = body_at + csize
        st.comp_bytes += csize
        st.raw_bytes += usize

        if ph.type_ == PageType.DICTIONARY_PAGE:
            var n = Int(ph.dictionary_page_header.value().num_values)
            t0 = perf_counter_ns()
            var raw = AllCodecs.decompress(codec, body, usize, scratch)
            t1 = perf_counter_ns()
            st.decomp += t1 - t0
            if len(scratch) > scratch_high:
                st.alloc_bytes += len(scratch) - scratch_high
                st.alloc_count += 1
                scratch_high = len(scratch)
            dict = decode_plain(leaf.physical, leaf.type_length, raw, n)
            st.values += perf_counter_ns() - t1
            has_dict = True
            continue
        if ph.type_ == PageType.INDEX_PAGE:
            continue

        var n: Int
        var non_null: Int
        var enc: Int32

        if ph.type_ == PageType.DATA_PAGE:
            ref h = ph.data_page_header
            n = Int(h.value().num_values)
            enc = h.value().encoding.value
            t0 = perf_counter_ns()
            var raw = AllCodecs.decompress(codec, body, usize, scratch)
            t1 = perf_counter_ns()
            st.decomp += t1 - t0
            if len(scratch) > scratch_high:
                st.alloc_bytes += len(scratch) - scratch_high
                st.alloc_count += 1
                scratch_high = len(scratch)
            var buf = raw
            var pos = 0
            if leaf.max_rep > 0:
                pos = _read_levels(
                    buf,
                    pos,
                    h.value().repetition_level_encoding.value,
                    leaf.max_rep,
                    n,
                    True,
                    0,
                    reps,
                )
            non_null = n
            if leaf.max_def > 0:
                var got = _take_defs(
                    buf,
                    pos,
                    h.value().definition_level_encoding.value,
                    leaf.max_def,
                    n,
                    True,
                    0,
                    cd,
                )
                pos = got[0]
                non_null = got[1]
            var t2 = perf_counter_ns()
            st.levels += t2 - t1
            _profile_values(
                values, enc, leaf, buf[pos:], non_null, dict, has_dict, st
            )
        else:
            ref h = ph.data_page_header_v2
            n = Int(h.value().num_values)
            enc = h.value().encoding.value
            var nulls = Int(h.value().num_nulls)
            var rep_len = Int(h.value().repetition_levels_byte_length)
            var def_len = Int(h.value().definition_levels_byte_length)
            t0 = perf_counter_ns()
            if leaf.max_rep > 0:
                _ = _read_levels(
                    body[0:rep_len],
                    0,
                    Encoding.RLE.value,
                    leaf.max_rep,
                    n,
                    False,
                    rep_len,
                    reps,
                )
            if leaf.max_def > 0:
                _ = _take_defs(
                    body[rep_len : rep_len + def_len],
                    0,
                    Encoding.RLE.value,
                    leaf.max_def,
                    n,
                    False,
                    def_len,
                    cd,
                )
            t1 = perf_counter_ns()
            st.levels += t1 - t0
            non_null = n - nulls
            var vbytes = body[rep_len + def_len :]
            if h.value().is_compressed.or_else(True):
                var raw = AllCodecs.decompress(
                    codec, vbytes, usize - rep_len - def_len, scratch
                )
                var t2 = perf_counter_ns()
                st.decomp += t2 - t1
                if len(scratch) > scratch_high:
                    st.alloc_bytes += len(scratch) - scratch_high
                    st.alloc_count += 1
                    scratch_high = len(scratch)
                _profile_values(
                    values, enc, leaf, raw, non_null, dict, has_dict, st
                )
            else:
                _profile_values(
                    values, enc, leaf, vbytes, non_null, dict, has_dict, st
                )

        slots += n
        cd.num_slots = slots

    st.out_bytes += len(values.bytes)
    st.alloc_bytes += len(values.bytes) + len(defs) * 2 + len(reps) * 2
    st.alloc_count += 3


def _profile_values(
    mut values: PhysBuffer,
    encoding: Int32,
    leaf: LeafColumn,
    data: Span[UInt8, _],
    count: Int,
    dict: PhysBuffer,
    has_dict: Bool,
    mut st: Stages,
) raises:
    """The value stage, split between the dictionary gather and everything else.
    """
    var dictionary = (
        encoding == Encoding.RLE_DICTIONARY.value
        or encoding == Encoding.PLAIN_DICTIONARY.value
    )
    if dictionary and has_dict and count > 0:
        var t0 = perf_counter_ns()
        gather_dict_into(values, dict, data, count)
        st.gather += perf_counter_ns() - t0
        st.alloc_bytes += 1024 * 4
        st.alloc_count += 1
        return
    var t0 = perf_counter_ns()
    var before = len(values.bytes)
    if encoding == Encoding.PLAIN.value:
        decode_plain_into(values, leaf.physical, leaf.type_length, data, count)
        st.values += perf_counter_ns() - t0
    else:
        var vals = _decode_values(encoding, leaf, data, count, dict, has_dict)
        var t1 = perf_counter_ns()
        st.values += t1 - t0
        values.extend(vals)
        st.concat += perf_counter_ns() - t1
        st.alloc_bytes += len(vals.bytes)
        st.alloc_count += 1
    _ = before


def _split_commas(text: String) -> List[String]:
    var out = List[String]()
    var current = String()
    for slice in text.codepoint_slices():
        if slice == ",":
            if current.byte_length() > 0:
                out.append(current.copy())
            current = String()
        else:
            current += String(slice)
    if current.byte_length() > 0:
        out.append(current^)
    return out^


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: profile-scan <file.parquet> [--select a,b] [--repeat N]")
        return
    var path = String(args[1])
    var names = List[String]()
    var repeat = 5
    var i = 2
    while i < len(args):
        var flag = String(args[i])
        if flag == "--select" and i + 1 < len(args):
            names = _split_commas(String(args[i + 1]))
            i += 2
        elif flag == "--repeat" and i + 1 < len(args):
            repeat = Int(String(args[i + 1]))
            i += 2
        else:
            raise Error(
                String("profile-scan: unexpected argument '", flag, "'")
            )

    # Which leaves the projection reaches. A reader with the projection applied
    # answers that, so the profile walks exactly the chunks the reader would.
    var probe = ParquetReader[AllCodecs].open(path)
    if len(names) > 0:
        probe.select_columns(names.copy())
    var wanted = List[Int]()
    for leaf in range(len(probe.schema.leaves)):
        if probe._needed[leaf]:
            wanted.append(leaf)

    var samples = List[Int]()
    var best = Stages()
    var rows = 0
    for run in range(repeat):
        var st = Stages()
        var t0 = perf_counter_ns()
        var bytes = read_parquet_file(path)
        var t1 = perf_counter_ns()
        st.slurp = t1 - t0
        st.file_bytes = len(bytes)

        var meta = read_footer(Span(bytes))
        var schema = build_schema(meta.schema)
        var t2 = perf_counter_ns()
        st.footer = t2 - t1

        for rg in range(len(meta.row_groups)):
            ref cols = meta.row_groups[rg].columns
            for k in range(len(wanted)):
                _walk_chunk(
                    Span(bytes),
                    cols[wanted[k]].meta_data.value(),
                    schema.leaves[wanted[k]],
                    st,
                )

        # `index` and `assemble` come from the real reader rather than from a
        # replay — `_load` minus the chunk walk, and `read_table` minus `_load`
        # — so neither can drift away from what the reader actually does.
        var walked = (
            st.hdr + st.decomp + st.levels + st.values + st.gather + st.concat
        )
        var loader = ParquetReader[AllCodecs](bytes.copy())
        if len(names) > 0:
            loader.select_columns(names.copy())
        var load_ns = 0
        for rg in range(len(meta.row_groups)):
            var l0 = perf_counter_ns()
            loader._load(rg)
            load_ns += perf_counter_ns() - l0

        var t3 = perf_counter_ns()
        var r = ParquetReader[AllCodecs](bytes.copy())
        if len(names) > 0:
            r.select_columns(names.copy())
        var table = r.read_table()
        var t4 = perf_counter_ns()
        rows = table.num_rows
        st.total = (t4 - t3) + st.slurp

        st.index = load_ns - walked
        if st.index < 0:
            st.index = 0
        st.assemble = (t4 - t3) - st.footer - load_ns
        if st.assemble < 0:
            st.assemble = 0
        samples.append(st.total)
        if run == 0 or st.total < best.total:
            best = Stages(copy=st)

    var ordered = _sorted(samples^)
    var last = len(ordered) - 1
    var p50 = ordered[(50 * last + 50) // 100]
    var p90 = ordered[(90 * last + 50) // 100]

    print(
        String(
            "\n",
            path,
            "\n  ",
            rows,
            " rows, ",
            len(wanted),
            " of ",
            len(probe.schema.leaves),
            " leaves, ",
            best.file_bytes // 1024,
            " KiB file, ",
            best.pages,
            " pages",
        )
    )
    print(
        String(
            "  total p50 ",
            _ms(p50),
            " ms, p90 ",
            _ms(p90),
            " ms over ",
            len(ordered),
            " runs (rows below are the fastest run's split)",
        )
    )
    print("  stage         time     share   bandwidth  bytes")
    print(
        _row(
            "slurp",
            best.slurp,
            best.total,
            best.file_bytes,
            String(best.file_bytes // 1024, " KiB read"),
        )
    )
    print(_row("footer", best.footer, best.total, 0, ""))
    print(
        _row(
            "hdr",
            best.hdr,
            best.total,
            0,
            String(best.pages, " page headers"),
        )
    )
    print(
        _row(
            "decomp",
            best.decomp,
            best.total,
            best.raw_bytes,
            String(
                best.comp_bytes // 1024,
                " KiB in -> ",
                best.raw_bytes // 1024,
                " KiB out",
            ),
        )
    )
    print(_row("levels", best.levels, best.total, 0, ""))
    print(_row("values", best.values, best.total, 0, ""))
    print(
        _row(
            "gather",
            best.gather,
            best.total,
            best.out_bytes,
            String(best.out_bytes // 1024, " KiB materialised"),
        )
    )
    print(_row("concat", best.concat, best.total, 0, ""))
    print(_row("index", best.index, best.total, 0, "per-row slot/value index"))
    print(
        _row(
            "assemble",
            best.assemble,
            best.total,
            best.out_bytes,
            "the copy into the Arrow buffers",
        )
    )
    print(_row("TOTAL", best.total, best.total, 0, ""))
    print(
        String(
            "  allocations: ",
            best.alloc_count,
            " buffers, ",
            best.alloc_bytes // 1024,
            " KiB",
        )
    )

    # What the `slurp` row would cost if the reader only fetched the chunks
    # the projection reaches. Both legs are the public API, timed the same
    # way, and the row counts are compared before either time is printed.
    var whole = List[Int]()
    var ranged = List[Int]()
    var fetched = 0
    for run in range(repeat + 1):
        var t0 = perf_counter_ns()
        var a = ParquetReader[AllCodecs].open(path)
        if len(names) > 0:
            a.select_columns(names.copy())
        var ta = a.read_table()
        var t1 = perf_counter_ns()
        var b = ParquetReader[AllCodecs].open_projected(path, names.copy())
        var tb = b.read_table()
        var t2 = perf_counter_ns()
        if ta.num_rows != tb.num_rows:
            raise Error("profile-scan: the two legs disagree on the row count")
        if run == 0:
            var rs = b.needed_byte_ranges()
            fetched = 0
            for k in range(len(rs)):
                fetched += rs[k][1]
            continue
        whole.append(t1 - t0)
        ranged.append(t2 - t1)
    var w = _sorted(whole^)
    var g = _sorted(ranged^)
    var wl = len(w) - 1
    print(
        String(
            "  read_table, whole file  p50 ",
            _ms(w[(50 * wl + 50) // 100]),
            " ms  p90 ",
            _ms(w[(90 * wl + 50) // 100]),
            " ms   (",
            best.file_bytes // 1024,
            " KiB fetched)",
        )
    )
    print(
        String(
            "  read_table, ranged      p50 ",
            _ms(g[(50 * wl + 50) // 100]),
            " ms  p90 ",
            _ms(g[(90 * wl + 50) // 100]),
            " ms   (",
            fetched // 1024,
            " KiB fetched)",
        )
    )
