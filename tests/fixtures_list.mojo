"""The fixture list the parity tests walk, and which columns need which codecs."""


def core_fixtures() -> List[String]:
    """Every fixture the default codec set (uncompressed / snappy / gzip) reads.
    """
    return [
        String("primitives"),
        String("logical"),
        String("extension"),
        String("float16"),
        String("nested"),
        String("encodings"),
        String("delta_length"),
        String("v1legacy"),
        String("v1pages"),
        String("v2pages"),
        String("v2pages_uncompressed"),
        String("pageindex"),
        String("manypages"),
        String("bloom"),
        String("nostats"),
        String("int96"),
        String("decimal_int"),
        String("legacy_list"),
        String("empty"),
        String("allnull"),
        String("fieldids"),
        String("prune"),
        String("big"),
    ]


def iceberg_fixtures() -> List[String]:
    """Real Iceberg data files written by parquet-rs and by PyIceberg."""
    return [
        String("unpartitioned"),
        String("ident_part_eu"),
        String("day_part"),
        String("bucket_part"),
        String("nullable"),
        String("orders_apac"),
        String("position_deletes"),
    ]


def iceberg_zstd_fixtures() -> List[String]:
    """The Iceberg fixtures PyIceberg wrote with ZSTD."""
    return [String("deletes_data"), String("evolved")]


def codec_columns() -> List[String]:
    """`codecs.parquet`'s columns, one per compression codec."""
    return [
        String("none"),
        String("snappy"),
        String("gzip"),
        String("zstd"),
        String("lz4"),
        String("brotli"),
        String("ints"),
    ]


def default_codec_columns() -> List[String]:
    """The subset of `codecs.parquet` that needs no FFI."""
    return [String("none"), String("snappy"), String("gzip"), String("ints")]


def full_codec_columns() -> List[String]:
    """Every column of `codecs.parquet` — `AllCodecs` covers all seven Parquet
    codecs, so this is `codec_columns()`. Kept as its own name because the
    default set still covers only a subset."""
    return codec_columns()
