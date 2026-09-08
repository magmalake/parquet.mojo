"""Compatibility re-export. The C Data Interface import lives in
`arrow-mlake.mojo` now — see `parquet.arrow` for why.

`from parquet.carrow_import import import_c` keeps resolving; new code should
import from `arrow_mlake.carrow_import` directly.
"""

from arrow_mlake.carrow_import import (
    MAX_IMPORT_DEPTH,
    CArrowArrayStream,
    ImportedArray,
    ImportedStream,
    StreamGetLastErrorFn,
    StreamGetNextFn,
    StreamGetSchemaFn,
    StreamReleaseFn,
    extension_name,
    import_batch_c,
    import_c,
    n_children_for_type,
    parquet_field_id,
    parse_format,
    release_c_array,
    release_c_schema,
)
