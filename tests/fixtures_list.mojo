"""The fixture list the parity tests walk, and which columns need which codecs."""


def core_fixtures() -> List[String]:
    """Every fixture the default codec set (uncompressed / snappy / gzip) reads."""
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
    """Everything but Brotli, which no Mojo library implements yet."""
    return [
        String("none"),
        String("snappy"),
        String("gzip"),
        String("zstd"),
        String("lz4"),
        String("ints"),
    ]
