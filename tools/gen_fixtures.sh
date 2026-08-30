#!/usr/bin/env bash
# Regenerate tests/fixtures/*.parquet and their pyarrow oracles.
# Needs `uv` on PATH; builds a throwaway venv so nothing leaks into pixi.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${TMPDIR:-/tmp}/parquet-mojo-fixtures-venv"
uv venv --quiet "$VENV"
VIRTUAL_ENV="$VENV" uv pip install --quiet 'pyarrow>=21,<26' numpy
"$VENV/bin/python" tools/gen_fixtures.py tests/fixtures
"$VENV/bin/python" tools/oracle_pyarrow.py tests/fixtures
