"""Compatibility re-export. The C Data Interface export lives in
`arrow-mlake.mojo` now — see `parquet.arrow` for why.

`from parquet.carrow import export_c` keeps resolving; new code should import
from `arrow_mlake.carrow` directly.
"""

from arrow_mlake.carrow import (
    ARROW_FLAG_NULLABLE,
    ArrayReleaseFn,
    CArrowArray,
    CArrowSchema,
    ExportedArray,
    SchemaReleaseFn,
    export_c,
    n_buffers_for,
    n_buffers_for_type,
    release_array,
    release_child_array,
    release_child_schema,
    release_schema,
)
