"""`parquet.mojo` — Apache Parquet in pure Mojo.

Part of magmalake: data lake building blocks in Mojo.

```mojo
from parquet import ParquetReader

var r = ParquetReader.open("part-0.parquet")
var t = r.read_table()
for i in range(t.num_columns()):
    print(t.name(i), String(t.type(i)))
```

Metadata comes from `thrift.mojo`, page CRC32s from `hashes.mojo`, Snappy from
`snappy.mojo` and the DEFLATE half of GZIP from `avro.mojo` — all pure Mojo and
all consumed by source path. `ZSTD` and `LZ4` need `parquet.ext_full`, which
pulls in the two FFI tins.
"""

from parquet.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_DATE32,
    AT_DECIMAL128,
    AT_FIXED_SIZE_BINARY,
    AT_FLOAT16,
    AT_FLOAT32,
    AT_FLOAT64,
    AT_INT16,
    AT_INT32,
    AT_INT64,
    AT_INT8,
    AT_LARGE_BINARY,
    AT_LARGE_LIST,
    AT_LARGE_UTF8,
    AT_LIST,
    AT_MAP,
    AT_NULL,
    AT_STRUCT,
    AT_TIME32,
    AT_TIME64,
    AT_TIMESTAMP,
    AT_UINT16,
    AT_UINT32,
    AT_UINT64,
    AT_UINT8,
    AT_UTF8,
    TU_MICRO,
    TU_MILLI,
    TU_NANO,
    TU_SECOND,
    ArrayArena,
    ArrayData,
    ArrowType,
    bit_get,
    unit_name,
)
from parquet.bloom import BloomFilter, read_bloom_filter
from parquet.carrow import CArrowArray, CArrowSchema, ExportedArray, export_c
from parquet.codec import CodecSet, DefaultCodecs
from parquet.encoding import PhysBuffer
from parquet.page import ColumnData, read_column_chunk
from parquet.reader import (
    OP_EQ,
    OP_GE,
    OP_GT,
    OP_LE,
    OP_LT,
    OP_NE,
    ParquetReader,
    Predicate,
    RecordBatch,
    Table,
    array_bool,
    array_bool_into,
    array_f64,
    array_f64_into,
    array_i64,
    array_i64_into,
    array_str,
    array_str_into,
    op_name,
)
from parquet.schema import (
    ArrowField,
    LeafColumn,
    ParquetSchema,
    SchemaNode,
    arrow_type_of,
    build_schema,
)
from parquet.stats import (
    SV_BOOL,
    SV_BYTES,
    SV_FLOAT,
    SV_INT,
    SV_NONE,
    SV_UINT,
    ScalarValue,
    TypedStats,
    compare_scalars,
    decode_stats,
)
