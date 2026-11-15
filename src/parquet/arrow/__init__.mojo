"""Compatibility re-export. The Arrow layer lives in `arrow-mlake.mojo` now.

`ArrayData`, `ArrayArena` and `ArrowType` are Arrow's memory layout, not
Parquet's; they lived here only because this is where they were first needed.
They moved to [arrow-mlake.mojo](https://github.com/magmalake/arrow-mlake.mojo)
when `lancedb.mojo` needed the same C Data Interface, since depending on
`parquet-mojo` for it would have taken a binding with no tin dependencies at
all to nine, three of them compression codecs it would never call.

This module exists so that `from parquet.arrow import ArrayData` keeps
resolving for consumers written before the split. New code should import from
`arrow_mlake.arrow` directly; this shim will go away when arrow-mlake itself is
replaced by [marrow](https://github.com/kszucs/marrow).
"""

from arrow_mlake.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_DATE32,
    AT_DECIMAL128,
    AT_FIXED_SIZE_BINARY,
    AT_FLOAT16,
    AT_FLOAT32,
    AT_FLOAT64,
    AT_INT8,
    AT_INT16,
    AT_INT32,
    AT_INT64,
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
    AT_UINT8,
    AT_UINT16,
    AT_UINT32,
    AT_UINT64,
    AT_UTF8,
    TU_MICRO,
    TU_MILLI,
    TU_NANO,
    TU_SECOND,
    ArrayArena,
    ArrayData,
    ArrowType,
    at,
    at_decimal,
    at_fixed,
    at_time,
    at_timestamp,
    bit_fill_valid,
    bit_get,
    bit_set,
    bitmap_bytes,
    load_f32,
    load_f64,
    load_i32,
    load_i64,
    load_u64,
    store_u32,
    store_u64,
    unit_name,
    unit_suffix,
)
