"""Per-stage decode and encode profile. `pixi run profile`.

Splits the cost of reading and of writing one file into the stages that
actually do work, so an optimisation can be aimed at the stage that dominates
rather than guessed at. Both walks mirror the library's own code path and call
the same helpers, so the numbers are the library's own costs.

Decode — the page walk mirrors `parquet.page.read_column_chunk`:

| stage | what it covers |
|---|---|
| footer | Thrift footer parse and schema build |
| hdr | page headers (Thrift compact) |
| decomp | `Codecs.decompress` per page — a borrow, so UNCOMPRESSED is free |
| levels | definition and repetition levels into `List[UInt16]` |
| values | the value encoding: PLAIN / DELTA / BSS |
| gather | `gather_dict_into`: dictionary indices decoded, bounds-checked and materialised, which is one fused pass and so one row |
| concat | `PhysBuffer.extend`, the per-page copy into the chunk buffer |
| index | `_load`'s per-row slot/value index building |
| assemble | Dremel assembly into Arrow buffers (`build_field`) |

`index` and `assemble` are measured at the public API — `_load` minus the
chunk decode, and `read_table` minus `_load` — so they cannot drift.

Encode — the row-group walk mirrors `ParquetWriter._write_row_group`,
`_write_chunk` and `finish`:

| stage | what it covers |
|---|---|
| schema | `build_write_schema`, the Arrow field tree -> Parquet schema |
| shred | `shred_flat` / `shred`: Arrow arrays -> levels + `PhysBuffer` |
| dict | the `DictBuilder` pass, *including* one that is then abandoned |
| levels | `encode_levels` for rep/def, plus the per-page non-null scan |
| values | `_plain_into` / `encode_hybrid` into the page body |
| stats | `min_max` per page and the chunk-bound merge |
| codec | `Codecs.compress` — for UNCOMPRESSED, a whole-page copy |
| pagehdr | the Thrift compact `PageHeader` per page |
| emit | `out.extend` of header and body: the copy into the file buffer |
| pgindex | `OffsetIndex` / `ColumnIndex` serialisation in `finish` |
| footer | `write_footer` and the trailer |

`TOTAL` is the real `ParquetWriter` timed end to end at the public API, so the
gap between it and the stage sum is visible rather than assumed, and each case
prints two more public-API measurements — the same writer with
`use_dictionary=False` and with statistics and the page index off — whose
deltas are what the `dict` and `stats` rows should say.

One caveat, and it is why those cross-checks are printed. Replaying the write
path means calling `DictBuilder` from a second place in the binary, and
`DictBuilder.index_of` — the byte-keyed probe a `BYTE_ARRAY` dictionary uses —
is only as fast as it is because it inlines into `_write_chunk`, where it is
otherwise the sole call. Referencing it from anywhere else costs a string
dictionary column about half again its write time (measured: 14.1 ms -> 21.6 ms
on a 1M-row `BYTE_ARRAY` column) in *this* binary only. So for byte-array
dictionary columns the `dict` row and `TOTAL` both read high here, and the
`use_dictionary=False` cross-check is the number a normal caller sees. The
fixed-width paths (`index_of_key`) are unaffected, and so is every other stage.
"""

from parquet import (
    DefaultCodecs,
    ParquetReader,
    ParquetWriter,
    Table,
    WriterOptions,
)
from parquet.arrow import ArrayArena
from parquet.bitio import HYBRID_BLOCK
from parquet.codec import CodecSet
from parquet.encoding import (
    PK_BOOL,
    PhysBuffer,
    decode_plain,
    decode_plain_into,
    gather_dict_into,
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
from parquet.rle_encode import encode_hybrid, encode_levels
from parquet.schema import LeafColumn, build_schema
from parquet.writer import (
    DictBuilder,
    LeafBuffer,
    WriteSchema,
    _bit_width,
    _plain_bytes,
    _plain_into,
    _stat_less,
    build_write_schema,
    dict_value_cap,
    min_max,
    shred,
    shred_flat,
)
from std.time import perf_counter_ns
from thrift import (
    BoundaryOrder,
    ColumnIndex,
    ColumnMetaData,
    CompressionCodec,
    DataPageHeader,
    DictionaryPageHeader,
    Encoding,
    FileMetaData,
    OffsetIndex,
    PageHeader,
    PageLocation,
    PageType,
    TCompactProtocolWriter,
    read_footer,
    read_page_header,
    read_parquet_file,
    write_footer,
    write_footer_trailer,
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


def _count_scratch(mut st: Stages, scratch: List[UInt8], mut high: Int):
    """Charge the decompression buffer for what it actually allocates.

    One buffer serves a whole column chunk and never shrinks, so it allocates
    only when a page pushes it past its high-water mark — and an uncompressed
    page never touches it at all. Adding `len(raw)` per page, as this profile
    did while every page got a buffer of its own, would now count bytes nothing
    ever allocated.
    """
    if len(scratch) > high:
        st.alloc_bytes += len(scratch) - high
        st.alloc_count += 1
        high = len(scratch)


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
    # Mirrors `read_column_chunk`: a leaf whose definition levels are one bit
    # wide decodes them straight into a validity mask, and the `levels` row has
    # to be timing what the library actually does.
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
        var body_at = offset + hdr[1]
        var csize = Int(ph.compressed_page_size)
        var usize = Int(ph.uncompressed_page_size)
        var body = file[body_at : body_at + csize]
        offset = body_at + csize

        if ph.type_ == PageType.DICTIONARY_PAGE:
            var n = Int(ph.dictionary_page_header.value().num_values)
            t0 = perf_counter_ns()
            var raw = DefaultCodecs.decompress(codec, body, usize, scratch)
            t1 = perf_counter_ns()
            st.decomp += t1 - t0
            _count_scratch(st, scratch, scratch_high)
            dict = decode_plain(leaf.physical, leaf.type_length, raw, n)
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
            var raw = DefaultCodecs.decompress(codec, body, usize, scratch)
            t1 = perf_counter_ns()
            st.decomp += t1 - t0
            _count_scratch(st, scratch, scratch_high)
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
                var raw = DefaultCodecs.decompress(
                    codec, vbytes, usize - rep_len - def_len, scratch
                )
                var t2 = perf_counter_ns()
                st.decomp += t2 - t1
                _count_scratch(st, scratch, scratch_high)
                _profile_values(
                    values, enc, leaf, raw, non_null, dict, has_dict, st
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
    """`_decode_values_into`, with the dictionary path timed as `gather`.

    Index decoding, the bounds check and the gather are one fused pass over a
    4 KiB scratch block, so there is no page-sized index array to time or to
    count as an allocation — which is the point of it, and why `values` and
    `gather` no longer split the same page between them.
    """
    var t0 = perf_counter_ns()
    if (
        encoding == Encoding.RLE_DICTIONARY.value
        or encoding == Encoding.PLAIN_DICTIONARY.value
    ) and has_dict:
        gather_dict_into(values, dict, data, count)
        st.gather += perf_counter_ns() - t0
        # The scratch block `gather_dict_into` decodes into, which is what
        # replaces the page-sized index array in this count.
        st.alloc_bytes += HYBRID_BLOCK * 4
        st.alloc_count += 1
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


# ── the write side ─────────────────────────────────────────────────────────


struct WStages(Copyable, Defaultable, Movable):
    var schema: Int
    var shred: Int
    var dict: Int
    var levels: Int
    var values: Int
    var stats: Int
    var codec: Int
    var pagehdr: Int
    var emit: Int
    var pgindex: Int
    var footer: Int
    var replay: Int
    var total: Int
    var alloc_bytes: Int
    var alloc_count: Int
    var out_bytes: Int

    def __init__(out self):
        self.schema = 0
        self.shred = 0
        self.dict = 0
        self.levels = 0
        self.values = 0
        self.stats = 0
        self.codec = 0
        self.pagehdr = 0
        self.emit = 0
        self.pgindex = 0
        self.footer = 0
        self.replay = 0
        self.total = 0
        self.alloc_bytes = 0
        self.alloc_count = 0
        self.out_bytes = 0

    def __init__(out self, *, copy: Self):
        self.schema = copy.schema
        self.shred = copy.shred
        self.dict = copy.dict
        self.levels = copy.levels
        self.values = copy.values
        self.stats = copy.stats
        self.codec = copy.codec
        self.pagehdr = copy.pagehdr
        self.emit = copy.emit
        self.pgindex = copy.pgindex
        self.footer = copy.footer
        self.replay = copy.replay
        self.total = copy.total
        self.alloc_bytes = copy.alloc_bytes
        self.alloc_count = copy.alloc_count
        self.out_bytes = copy.out_bytes

    def __init__(out self, *, deinit move: Self):
        self = Self(copy=move)

    def sum(self) -> Int:
        return (
            self.schema
            + self.shred
            + self.dict
            + self.levels
            + self.values
            + self.stats
            + self.codec
            + self.pagehdr
            + self.emit
            + self.pgindex
            + self.footer
        )

    def keep_min(mut self, other: Self):
        if self.total == 0 or other.total < self.total:
            self = Self(copy=other)


def _write_chunk_stages(
    s: WriteSchema,
    leaf_index: Int,
    buf: LeafBuffer,
    options: WriterOptions,
    mut out: List[UInt8],
    mut offs: List[OffsetIndex],
    mut cidx: List[ColumnIndex],
    mut st: WStages,
) raises:
    """`ParquetWriter._write_chunk`, with a timer around each stage."""
    ref leaf = s.leaves[leaf_index]
    var codec = options.codec
    var n_slots = len(buf.defs)
    var n_values = buf.values.count
    var want_bounds = options.write_statistics or options.write_page_index

    # ── dictionary ────────────────────────────────────────────────────────
    var t0 = perf_counter_ns()
    var use_dict = (
        options.use_dictionary and buf.values.kind != PK_BOOL and n_values > 0
    )
    var indices = List[UInt16]()
    var dict = PhysBuffer()
    if use_dict:
        var cap = dict_value_cap(n_values)
        var builder = DictBuilder(buf.values.kind, buf.values.width, cap)
        indices.resize(n_values, 0)
        var out_idx = indices.unsafe_ptr()
        var kw = builder.key_width
        var src = buf.values.bytes.unsafe_ptr()
        for i in range(n_values):
            var k: Int
            if kw == 8:
                k = builder.index_of_key(
                    src.unsafe_offset(i * 8)
                    .unsafe_bitcast[UInt64]()
                    .unsafe_load[alignment=1](0)
                )
            elif kw == 4:
                k = builder.index_of_key(
                    UInt64(
                        src.unsafe_offset(i * 4)
                        .unsafe_bitcast[UInt32]()
                        .unsafe_load[alignment=1](0)
                    )
                )
            else:
                k = builder.index_of(buf.values.value_span(i))
            if k > cap:
                use_dict = False
                break
            out_idx.unsafe_store(i, UInt16(k))
        st.alloc_bytes += n_values * 2 + (Int(builder.mask) + 1) * 4
        st.alloc_count += 2
        if use_dict:
            dict = builder.values.copy()
            st.alloc_bytes += len(dict.bytes)
            st.alloc_count += 1
        else:
            indices.clear()
    st.dict += perf_counter_ns() - t0
    var encoding = (
        Encoding.RLE_DICTIONARY.value if use_dict else Encoding.PLAIN.value
    )

    if use_dict:
        var body = _plain_bytes(leaf, dict)
        st.alloc_bytes += len(body)
        st.alloc_count += 1
        var t1 = perf_counter_ns()
        var compressed = DefaultCodecs.compress(codec, Span(body))
        var t2 = perf_counter_ns()
        st.codec += t2 - t1
        st.alloc_bytes += len(compressed)
        st.alloc_count += 1
        var ph = PageHeader()
        ph.type_ = PageType.DICTIONARY_PAGE
        ph.uncompressed_page_size = Int32(len(body))
        ph.compressed_page_size = Int32(len(compressed))
        var dh = DictionaryPageHeader()
        dh.num_values = Int32(dict.count)
        dh.encoding = Encoding.PLAIN
        ph.dictionary_page_header = dh^
        var w = TCompactProtocolWriter()
        ph.write(w)
        var hdr = w^.take()
        var t3 = perf_counter_ns()
        st.pagehdr += t3 - t2
        out.extend(Span(hdr))
        out.extend(Span(compressed))
        st.emit += perf_counter_ns() - t3

    var flat = leaf.max_rep == 0
    var starts = List[Int]()
    var n_records = n_slots
    if not flat:
        var t1 = perf_counter_ns()
        for i in range(n_slots):
            if buf.reps[i] == 0:
                starts.append(i)
        starts.append(n_slots)
        n_records = len(starts) - 1
        st.levels += perf_counter_ns() - t1
    var oi = OffsetIndex()
    var ci = ColumnIndex()
    var page_nulls = List[Int64]()
    var value_width = leaf.type_length if leaf.type_length > 0 else 8
    var rec = 0
    var first_row: Int64 = 0
    var value_cursor = 0
    var def_width = _bit_width(leaf.max_def)
    var rep_width = _bit_width(leaf.max_rep)
    var chunk_lo = List[UInt8]()
    var chunk_hi = List[UInt8]()
    var have_chunk_bounds = False
    while rec < n_records:
        var rec_end = rec
        var bytes_so_far = 0
        while rec_end < n_records:
            var slots = 1 if flat else starts[rec_end + 1] - starts[rec_end]
            bytes_so_far += slots * value_width
            rec_end += 1
            if bytes_so_far >= options.data_page_size:
                break
        var s0 = rec if flat else starts[rec]
        var s1 = rec_end if flat else starts[rec_end]

        # ── levels ────────────────────────────────────────────────────────
        var t1 = perf_counter_ns()
        var page_values = s1 - s0
        if leaf.max_def > 0:
            page_values = 0
            var dp = buf.defs.unsafe_ptr()
            for i in range(s0, s1):
                if Int(dp.unsafe_load(i)) == leaf.max_def:
                    page_values += 1
        var body = List[UInt8]()
        if leaf.max_rep > 0:
            body.extend(Span(encode_levels(Span(buf.reps)[s0:s1], rep_width)))
        if leaf.max_def > 0:
            body.extend(Span(encode_levels(Span(buf.defs)[s0:s1], def_width)))
        var t2 = perf_counter_ns()
        st.levels += t2 - t1

        # ── values ────────────────────────────────────────────────────────
        if use_dict:
            var width = _bit_width(dict.count - 1)
            body.append(UInt8(width))
            body.extend(
                Span(
                    encode_hybrid(
                        Span(indices)[
                            value_cursor : value_cursor + page_values
                        ],
                        width,
                    )
                )
            )
        else:
            _plain_into(body, leaf, buf.values, value_cursor, page_values)
        value_cursor += page_values
        var t3 = perf_counter_ns()
        st.values += t3 - t2
        st.alloc_bytes += len(body)
        st.alloc_count += 1

        # ── codec ─────────────────────────────────────────────────────────
        var compressed = DefaultCodecs.compress(codec, Span(body))
        var t4 = perf_counter_ns()
        st.codec += t4 - t3
        st.alloc_bytes += len(compressed)
        st.alloc_count += 1

        # ── page header ───────────────────────────────────────────────────
        var ph = PageHeader()
        ph.type_ = PageType.DATA_PAGE
        ph.uncompressed_page_size = Int32(len(body))
        ph.compressed_page_size = Int32(len(compressed))
        var dph = DataPageHeader()
        dph.num_values = Int32(s1 - s0)
        dph.encoding = Encoding(encoding)
        dph.definition_level_encoding = Encoding.RLE
        dph.repetition_level_encoding = Encoding.RLE
        ph.data_page_header = dph^
        var page_at = Int64(len(out))
        var w = TCompactProtocolWriter()
        ph.write(w)
        var hdr = w^.take()
        var t5 = perf_counter_ns()
        st.pagehdr += t5 - t4

        # ── emit ──────────────────────────────────────────────────────────
        out.extend(Span(hdr))
        out.extend(Span(compressed))
        var t6 = perf_counter_ns()
        st.emit += t6 - t5

        var loc = PageLocation()
        loc.offset = page_at
        loc.compressed_page_size = Int32(len(hdr) + len(compressed))
        loc.first_row_index = first_row
        oi.page_locations.append(loc^)

        # ── statistics ────────────────────────────────────────────────────
        var has_bounds = False
        var page_lo = List[UInt8]()
        var page_hi = List[UInt8]()
        if want_bounds:
            var bounds = min_max(
                leaf, buf.values, value_cursor - page_values, page_values
            )
            page_lo = bounds[0].copy()
            page_hi = bounds[1].copy()
            has_bounds = bounds[2]
            if has_bounds:
                if not have_chunk_bounds:
                    chunk_lo = page_lo.copy()
                    chunk_hi = page_hi.copy()
                    have_chunk_bounds = True
                else:
                    if _stat_less(leaf, Span(page_lo), Span(chunk_lo)):
                        chunk_lo = page_lo.copy()
                    if _stat_less(leaf, Span(chunk_hi), Span(page_hi)):
                        chunk_hi = page_hi.copy()
        ci.null_pages.append(not has_bounds)
        ci.min_values.append(page_lo^)
        ci.max_values.append(page_hi^)
        page_nulls.append(Int64(s1 - s0 - page_values))
        st.stats += perf_counter_ns() - t6
        first_row += Int64(rec_end - rec)
        rec = rec_end
    ci.boundary_order = BoundaryOrder.UNORDERED
    ci.null_counts = page_nulls^
    offs.append(oi^)
    cidx.append(ci^)


def _write_replay(
    arena: ArrayArena,
    roots: List[Int],
    options: WriterOptions,
    mut st: WStages,
) raises:
    """`write_batch` + `finish`, with a timer around each stage."""
    var t0 = perf_counter_ns()
    var s = build_write_schema(arena, roots)
    st.schema += perf_counter_ns() - t0

    var out = List[UInt8]()
    out.extend(StringSlice("PAR1").as_bytes())
    var all_offs = List[List[OffsetIndex]]()
    var all_cidx = List[List[ColumnIndex]]()
    var rows = arena.nodes[roots[0]].length
    var n_leaves = len(s.leaves)
    var start = 0
    while start < rows:
        var end = start + options.row_group_size
        if end > rows:
            end = rows
        var cols = List[LeafBuffer]()
        for i in range(n_leaves):
            var b = LeafBuffer()
            b.values = PhysBuffer(
                physical_kind(s.leaves[i].physical),
                physical_width(s.leaves[i].physical, s.leaves[i].type_length),
            )
            cols.append(b^)
        var t1 = perf_counter_ns()
        for k in range(len(roots)):
            if shred_flat(s, s.roots[k], arena, roots[k], start, end, cols):
                continue
            for i in range(start, end):
                shred(s, s.roots[k], arena, roots[k], i, 0, 0, cols)
        st.shred += perf_counter_ns() - t1
        for i in range(n_leaves):
            st.alloc_bytes += (
                len(cols[i].values.bytes)
                + len(cols[i].values.offsets) * 4
                + len(cols[i].defs) * 2
                + len(cols[i].reps) * 2
            )
            st.alloc_count += 4
        var offs = List[OffsetIndex]()
        var cidx = List[ColumnIndex]()
        for i in range(n_leaves):
            _write_chunk_stages(s, i, cols[i], options, out, offs, cidx, st)
        all_offs.append(offs^)
        all_cidx.append(cidx^)
        start = end

    # ── page index ────────────────────────────────────────────────────────
    var t2 = perf_counter_ns()
    if options.write_page_index:
        for g in range(len(all_offs)):
            for c in range(len(all_offs[g])):
                var w = TCompactProtocolWriter()
                all_offs[g][c].write(w)
                var body = w^.take()
                out.extend(Span(body))
                st.alloc_bytes += len(body)
                st.alloc_count += 1
        for g in range(len(all_cidx)):
            for c in range(len(all_cidx[g])):
                var w = TCompactProtocolWriter()
                all_cidx[g][c].write(w)
                var body = w^.take()
                out.extend(Span(body))
                st.alloc_bytes += len(body)
                st.alloc_count += 1
    var t3 = perf_counter_ns()
    st.pgindex += t3 - t2

    # ── footer ────────────────────────────────────────────────────────────
    var meta = FileMetaData()
    meta.version = 2
    meta.schema = s.elements.copy()
    meta.num_rows = Int64(rows)
    meta.created_by = options.created_by.copy()
    var body = write_footer(meta)
    out.extend(Span(body))
    write_footer_trailer(out, len(body))
    st.footer += perf_counter_ns() - t3
    st.alloc_bytes += len(body) + len(out)
    st.alloc_count += 2


def _profile_write(
    table: Table, label: StringSlice, repeats: Int, col: Int
) raises:
    """Profile writing `col` (or every column, when `col < 0`) of a table.

    The Arrow arrays are decoded by the caller, outside the timers, so what is
    measured is purely the write path — and one table serves every scenario,
    because decoding a fresh one between them leaves the allocator in a state
    that inflates the next total by up to 50%.

    The real writer and the instrumented replay get a loop each rather than
    sharing one, for the same reason: the replay allocates and frees tens of
    megabytes a pass.
    """
    var best = WStages()
    var rows = table.num_rows
    var out_bytes = 0
    var total = _time_writer(table, col, repeats, 0)
    var no_dict = _time_writer(table, col, repeats, 1)
    var no_stats = _time_writer(table, col, repeats, 2)
    for _ in range(repeats):
        var w = ParquetWriter[DefaultCodecs](_wopts())
        for b in table.batches:
            var roots = b.roots.copy() if col < 0 else [b.roots[col]]
            w.write_batch(b.arena, roots)
        out_bytes = len(w^.finish())
    for _ in range(repeats):
        var st = WStages()
        var t0 = perf_counter_ns()
        for b in table.batches:
            var roots = b.roots.copy() if col < 0 else [b.roots[col]]
            _write_replay(b.arena, roots, _wopts(), st)
        st.replay = perf_counter_ns() - t0
        st.total = st.replay
        best.keep_min(st)
    best.total = total
    best.out_bytes = out_bytes

    print(
        String(
            "\n",
            label,
            " — ",
            rows,
            " rows -> ",
            best.out_bytes // 1024,
            " KiB",
        )
    )
    print(_row("schema", best.schema, best.total))
    print(_row("shred", best.shred, best.total))
    print(_row("dict", best.dict, best.total))
    print(_row("levels", best.levels, best.total))
    print(_row("values", best.values, best.total))
    print(_row("stats", best.stats, best.total))
    print(_row("codec", best.codec, best.total))
    print(_row("pagehdr", best.pagehdr, best.total))
    print(_row("emit", best.emit, best.total))
    print(_row("pgindex", best.pgindex, best.total))
    print(_row("footer", best.footer, best.total))
    print(_row("(sum)", best.sum(), best.total))
    print(
        String(
            "    (the stage rows above replay write_batch + finish with timers",
            " in it: ",
            _us(best.replay),
            " ms of replay against the writer's own ",
            _us(best.total),
            " ms)",
        )
    )
    print(_row("TOTAL", best.total, best.total))
    # Two cross-checks at the public API: the same writer with one option off.
    # The delta is what that feature costs a normal caller, and it is what the
    # `dict` and `stats` rows above should say.
    print(
        String(
            "    cross-check: use_dictionary=False ",
            _us(no_dict),
            " ms, so dict costs ",
            _us(best.total - no_dict),
            " ms; no stats or page index ",
            _us(no_stats),
            " ms, so stats costs ",
            _us(best.total - no_stats),
            " ms",
        )
    )
    print(
        String(
            "  allocations: ",
            best.alloc_count,
            " buffers, ",
            best.alloc_bytes // 1024,
            " KiB (output is ",
            best.out_bytes // 1024,
            " KiB)",
        )
    )


def _time_writer(table: Table, col: Int, repeats: Int, knob: Int) raises -> Int:
    """The real writer, best of `repeats`, with one `WriterOptions` field off.
    """
    var best = 0
    for _ in range(repeats):
        var opts = _wopts()
        if knob == 1:
            opts.use_dictionary = False
        elif knob == 2:
            opts.write_statistics = False
            opts.write_page_index = False
        var t0 = perf_counter_ns()
        var w = ParquetWriter[DefaultCodecs](opts^)
        for b in table.batches:
            var roots = b.roots.copy() if col < 0 else [b.roots[col]]
            w.write_batch(b.arena, roots)
        _ = w^.finish()
        var dt = perf_counter_ns() - t0
        if best == 0 or dt < best:
            best = dt
    return best


def _read_table(path: StringSlice) raises -> Table:
    """Decode a fixture into Arrow, outside every timer."""
    var raw = read_parquet_file(String(path))
    var r = ParquetReader[DefaultCodecs](raw^)
    r.verify_crc = False
    return r.read_table()


def _wopts() -> WriterOptions:
    """What the write benchmark uses: no codec, so no codec time is measured."""
    var o = WriterOptions()
    o.codec = CompressionCodec.UNCOMPRESSED.value
    return o^


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

    print("\n\nparquet.mojo encode profile (single threaded, UNCOMPRESSED)")
    try:
        var wide = _read_table("build/bench-wide.parquet")
        _profile_write(
            wide, "bench-wide, column a only (1M int64, PLAIN)", 10, 0
        )
        _profile_write(
            wide, "bench-wide, column b only (1M double, PLAIN)", 10, 1
        )
        _profile_write(
            wide, "bench-wide, column c only (1M int64, dictionary)", 10, 2
        )
        _profile_write(
            wide, "bench-wide, column d only (1M string, dictionary)", 10, 3
        )
        _profile_write(wide, "bench-wide, all four columns", 10, -1)
    except e:
        print("bench-wide.parquet: not built — `pixi run bench-pyarrow` first")
    _profile_write(
        _read_table("tests/fixtures/big.parquet"),
        "big.parquet (100k rows, 5 mixed cols)",
        20,
        -1,
    )
