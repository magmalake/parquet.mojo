"""Per-stage decode profile. `pixi run profile`.

Splits the cost of reading one file into the stages that actually do work, so
an optimisation can be aimed at the stage that dominates rather than guessed
at. The page walk here mirrors `parquet.page.read_column_chunk` and calls the
same helpers, so the numbers are the library's own costs:

| stage | what it covers |
|---|---|
| footer | Thrift footer parse and schema build |
| hdr | page headers (Thrift compact) |
| decomp | `Codecs.decompress` per page, including the UNCOMPRESSED copy |
| levels | definition and repetition levels into `List[UInt16]` |
| values | the value encoding: PLAIN / dictionary indices / DELTA / BSS |
| gather | dictionary index -> value materialisation |
| concat | `PhysBuffer.extend`, the per-page copy into the chunk buffer |
| index | `_load`'s per-row slot/value index building |
| assemble | Dremel assembly into Arrow buffers (`build_field`) |

`index` and `assemble` are measured at the public API — `_load` minus the
chunk decode, and `read_table` minus `_load` — so they cannot drift.
"""

from parquet import DefaultCodecs, ParquetReader
from parquet.codec import CodecSet
from parquet.encoding import (
    PhysBuffer,
    decode_dict_indices,
    decode_plain,
    decode_plain_into,
    gather_into,
    physical_kind,
    physical_width,
)
from parquet.page import (
    ColumnData,
    _decode_values,
    _read_levels,
    _take_defs,
    chunk_start,
    read_column_chunk,
)
from parquet.schema import LeafColumn, build_schema
from std.time import perf_counter_ns
from thrift import (
    ColumnMetaData,
    Encoding,
    PageType,
    read_footer,
    read_page_header,
    read_parquet_file,
)


struct Stages(Copyable, Defaultable, Movable):
    var footer: Int
    var hdr: Int
    var decomp: Int
    var levels: Int
    var values: Int
    var gather: Int
    var concat: Int
    var index: Int
    var assemble: Int
    var walk: Int
    var load: Int
    var total: Int
    var alloc_bytes: Int
    var alloc_count: Int

    def __init__(out self):
        self.footer = 0
        self.hdr = 0
        self.decomp = 0
        self.levels = 0
        self.values = 0
        self.gather = 0
        self.concat = 0
        self.index = 0
        self.assemble = 0
        self.walk = 0
        self.load = 0
        self.total = 0
        self.alloc_bytes = 0
        self.alloc_count = 0

    def __init__(out self, *, copy: Self):
        self.footer = copy.footer
        self.hdr = copy.hdr
        self.decomp = copy.decomp
        self.levels = copy.levels
        self.values = copy.values
        self.gather = copy.gather
        self.concat = copy.concat
        self.index = copy.index
        self.assemble = copy.assemble
        self.walk = copy.walk
        self.load = copy.load
        self.total = copy.total
        self.alloc_bytes = copy.alloc_bytes
        self.alloc_count = copy.alloc_count

    def __init__(out self, *, deinit move: Self):
        self = Self(copy=move)

    def keep_min(mut self, other: Self):
        if self.total == 0 or other.total < self.total:
            self = Self(copy=other)


def _us(ns: Int) -> String:
    var us = ns // 1000
    return String(us // 1000, ".", (us % 1000) // 100, (us % 100) // 10)


def _row(name: StringSlice, ns: Int, total: Int) -> String:
    var pct = 0
    if total > 0:
        pct = (ns * 1000) // total
    var pad = String()
    for _ in range(10 - name.byte_length()):
        pad += " "
    var ms = _us(ns)
    var mpad = String()
    for _ in range(8 - ms.byte_length()):
        mpad += " "
    return String(
        "  ", name, pad, mpad, ms, " ms   ", pct // 10, ".", pct % 10, " %"
    )


def _walk_chunk(
    file: Span[UInt8, _],
    cm: ColumnMetaData,
    leaf: LeafColumn,
    mut st: Stages,
) raises:
    """`read_column_chunk`, with a timer around each stage."""
    var values = PhysBuffer(
        physical_kind(leaf.physical),
        physical_width(leaf.physical, leaf.type_length),
    )
    var nv = Int(cm.num_values)
    if nv > 0 and nv < (1 << 30) and values.width > 0:
        if nv * values.width <= (1 << 28):
            values.bytes.reserve(nv * values.width)
    var cd = ColumnData()
    ref defs = cd.defs
    ref reps = cd.reps
    var offset = chunk_start(cm)
    var limit = offset + Int(cm.total_compressed_size)
    var want = Int(cm.num_values)
    var dict = PhysBuffer()
    var has_dict = False
    var codec = cm.codec.value
    var slots = 0

    while slots < want and offset < limit:
        var t0 = perf_counter_ns()
        var hdr = read_page_header(file, offset)
        ref ph = hdr[0]
        var t1 = perf_counter_ns()
        st.hdr += t1 - t0
        var body_at = offset + hdr[1]
        var csize = Int(ph.compressed_page_size)
        var usize = Int(ph.uncompressed_page_size)
        var body = file[body_at : body_at + csize]
        offset = body_at + csize

        if ph.type_ == PageType.DICTIONARY_PAGE:
            var n = Int(ph.dictionary_page_header.value().num_values)
            t0 = perf_counter_ns()
            var raw = DefaultCodecs.decompress(codec, body, usize)
            t1 = perf_counter_ns()
            st.decomp += t1 - t0
            st.alloc_bytes += len(raw)
            st.alloc_count += 1
            dict = decode_plain(leaf.physical, leaf.type_length, Span(raw), n)
            st.values += perf_counter_ns() - t1
            has_dict = True
            continue
        if ph.type_ == PageType.INDEX_PAGE:
            continue

        var n = 0
        var non_null = 0
        var enc = Encoding.PLAIN.value

        if ph.type_ == PageType.DATA_PAGE:
            ref h = ph.data_page_header
            n = Int(h.value().num_values)
            enc = h.value().encoding.value
            t0 = perf_counter_ns()
            var raw = DefaultCodecs.decompress(codec, body, usize)
            t1 = perf_counter_ns()
            st.decomp += t1 - t0
            st.alloc_bytes += len(raw)
            st.alloc_count += 1
            var buf = Span(raw)
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
                var raw = DefaultCodecs.decompress(
                    codec, vbytes, usize - rep_len - def_len
                )
                var t2 = perf_counter_ns()
                st.decomp += t2 - t1
                st.alloc_bytes += len(raw)
                st.alloc_count += 1
                _profile_values(
                    values, enc, leaf, Span(raw), non_null, dict, has_dict, st
                )
            else:
                _profile_values(
                    values, enc, leaf, vbytes, non_null, dict, has_dict, st
                )

        slots += n
        cd.num_slots = slots

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
    """`_decode_values_into`, splitting the dictionary path into index + gather.
    """
    var t0 = perf_counter_ns()
    if (
        encoding == Encoding.RLE_DICTIONARY.value
        or encoding == Encoding.PLAIN_DICTIONARY.value
    ) and has_dict:
        var indices = decode_dict_indices(data, count)
        var t1 = perf_counter_ns()
        st.values += t1 - t0
        st.alloc_bytes += len(indices) * 4
        st.alloc_count += 1
        gather_into(values, dict, indices)
        st.gather += perf_counter_ns() - t1
        return
    if encoding == Encoding.PLAIN.value:
        decode_plain_into(values, leaf.physical, leaf.type_length, data, count)
        st.values += perf_counter_ns() - t0
        return
    var out = _decode_values(encoding, leaf, data, count, dict, has_dict)
    st.alloc_bytes += len(out.bytes)
    st.alloc_count += 1
    var t2 = perf_counter_ns()
    values.extend(out)
    st.concat += perf_counter_ns() - t2
    st.values += t2 - t0


def _profile(path: StringSlice, label: StringSlice, repeats: Int) raises:
    var bytes = read_parquet_file(String(path))
    var best = Stages()
    var rows = 0
    for _ in range(repeats):
        var st = Stages()
        var t0 = perf_counter_ns()

        var meta = read_footer(Span(bytes))
        var schema = build_schema(meta.schema)
        var t1 = perf_counter_ns()
        st.footer = t1 - t0

        for rg in range(len(meta.row_groups)):
            ref cols = meta.row_groups[rg].columns
            for i in range(len(schema.leaves)):
                _walk_chunk(
                    Span(bytes),
                    cols[i].meta_data.value(),
                    schema.leaves[i],
                    st,
                )

        # `index` and `assemble` come from the real reader: _load minus the
        # chunk decode we just timed, and read_table minus _load.
        var chunk_ns = (
            st.hdr + st.decomp + st.levels + st.values + st.gather + st.concat
        )
        var t2 = perf_counter_ns()
        var r = ParquetReader[DefaultCodecs](bytes.copy())
        r.verify_crc = False
        var load_ns = 0
        for rg in range(len(meta.row_groups)):
            var l0 = perf_counter_ns()
            r._load(rg)
            load_ns += perf_counter_ns() - l0
        var t3 = perf_counter_ns()

        var r2 = ParquetReader[DefaultCodecs](bytes.copy())
        r2.verify_crc = False
        var t4 = perf_counter_ns()
        var tbl = r2.read_table()
        var t5 = perf_counter_ns()
        rows = tbl.num_rows

        st.walk = chunk_ns
        st.load = load_ns
        st.index = load_ns - chunk_ns
        if st.index < 0:
            st.index = 0
        st.assemble = (t5 - t4) - load_ns
        if st.assemble < 0:
            st.assemble = 0
        st.total = t5 - t4
        _ = t2
        _ = t3
        best.keep_min(st)

    print(
        String("\n", label, " — ", rows, " rows, ", len(bytes) // 1024, " KiB")
    )
    print(_row("footer", best.footer, best.total))
    print(_row("hdr", best.hdr, best.total))
    print(_row("decomp", best.decomp, best.total))
    print(_row("levels", best.levels, best.total))
    print(_row("values", best.values, best.total))
    print(_row("gather", best.gather, best.total))
    print(_row("concat", best.concat, best.total))
    print(_row("index", best.index, best.total))
    print(
        String(
            (
                "    (the stage rows above replay the chunk walk with timers in"
                " it: "
            ),
            _us(best.walk),
            " ms against the reader's own ",
            _us(best.load),
            " ms)",
        )
    )
    print(_row("assemble", best.assemble, best.total))
    print(_row("TOTAL", best.total, best.total))
    print(
        String(
            "  allocations: ",
            best.alloc_count,
            " buffers, ",
            best.alloc_bytes // 1024,
            " KiB (file is ",
            len(bytes) // 1024,
            " KiB)",
        )
    )


def main() raises:
    print("parquet.mojo decode profile (single threaded, best of N)")
    try:
        _profile(
            "build/bench-wide.parquet",
            "bench-wide.parquet (1M rows, int64/double/dict)",
            3,
        )
    except e:
        print("bench-wide.parquet: not built — `pixi run bench-pyarrow` first")
    _profile(
        "tests/fixtures/big.parquet",
        "big.parquet (100k rows, 5 mixed cols, snappy)",
        5,
    )
    _profile(
        "tests/fixtures/v2pages.parquet",
        "v2pages.parquet (500 rows, v2 pages)",
        20,
    )
