#!/usr/bin/env bash
# Write every fixture back out through our writer, then have pyarrow read it.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build/written
mojo build tools/write_fixtures.mojo -I src -I tests \
    -I ../thrift.mojo/src -I ../hashes.mojo/src -I ../snappy.mojo/src \
    -I ../avro.mojo/src -o build/write-fixtures
./build/write-fixtures > /dev/null
VENV="${TMPDIR:-/tmp}/parquet-mojo-fixtures-venv"
uv venv --quiet --allow-existing "$VENV" 2>/dev/null || uv venv --quiet "$VENV"
VIRTUAL_ENV="$VENV" uv pip install --quiet 'pyarrow>=21,<26'
"$VENV/bin/python" tools/verify_written.py
