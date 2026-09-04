#!/bin/sh
# Build and run tests/stress_threaded_read.mojo, optionally under a sanitizer.
#
#   tests/run_stress.sh plain    # no sanitizer, many rounds
#   tests/run_stress.sh tsan     # --sanitize thread, zero tolerance
#   tests/run_stress.sh asan     # --sanitize address (+ LeakSanitizer)
#
# The shape of this script — watchdog, grep-not-exit-code verdict, symbolizer —
# is threads.mojo's `tests/run_stress.sh`, and the reasoning behind each part
# lives there. The short version:
#
#  1. **A watchdog**, because a bug in a join path hangs rather than fails, and
#     macOS ships no `timeout(1)`. It self-tests against `sleep` every run.
#  2. **A verdict by grep**, because a sanitizer's exit status has varied
#     across versions: any `WARNING: ThreadSanitizer` / `ERROR:
#     (Address|Leak)Sanitizer` in the captured output fails the task.
#  3. **`LLVM_SYMBOLIZER_PATH`**, without which a report names an address
#     instead of a Mojo frame.
#
# Which sanitizer runs where is a property of the toolchain, not a choice: with
# Mojo 1.0.0 `--sanitize thread` works on osx-arm64 and `--sanitize address`
# does not link there; on linux-64 it is the other way round, because a TSan
# binary aborts before `main` when the Mojo runtime's bundled TCMalloc cannot
# get its 1 GiB-aligned mmap inside TSan's reservation.
#
# **`tsan` currently reports findings, and they are not this repository's.**
# The Mojo 1.0.0 runtime does not route its allocator through the malloc/free
# ThreadSanitizer intercepts, so a block freed on one worker and recycled on
# another looks like two threads touching one address. Every report over this
# workload carries that signature: no "Location is heap block …" line, and
# accesses of different widths at one address. `tests/tsan_control.mojo` —
# `pixi run -e stable stress-tsan-control` — reproduces it in forty lines with
# no parquet, no thrift and no shared state, and shows it disappearing when the
# tasks stop allocating; threads.mojo 0.4.0's own TSan stress, whose tasks
# never allocate, is clean on the same machine. So `stress-tsan` is a manual
# investigation task rather than a CI gate, and `stress` — the same workload,
# many more rounds, no sanitizer — is what CI runs.
#
# Overridable: STRESS_ROUNDS, STRESS_TIMEOUT (seconds), TSAN_OPTIONS,
# ASAN_OPTIONS.
set -eu

mode="${1:-plain}"
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
mkdir -p build

marker="build/.stress-timed-out"
includes="-I src -I tests -I ../threads.mojo/src -I ../thrift.mojo/src -I ../hashes.mojo/src -I ../snappy.mojo/src -I ../avro.mojo/src"

# run_with_timeout SECONDS COMMAND... — returns the command's status, or fails
# with the marker file present if the watchdog had to kill it.
run_with_timeout() {
    secs=$1
    shift
    rm -f "$marker"
    "$@" &
    job=$!
    # One-second ticks rather than one long `sleep`, so the watchdog notices
    # the job finishing and exits on its own.
    (
        waited=0
        while [ "$waited" -lt "$secs" ]; do
            sleep 1
            waited=$((waited + 1))
            kill -0 "$job" 2>/dev/null || exit 0
        done
        : >"$marker"
        kill -9 "$job" 2>/dev/null || true
    ) >/dev/null 2>&1 &
    dog=$!
    status=0
    wait "$job" || status=$?
    kill "$dog" 2>/dev/null || true
    wait "$dog" 2>/dev/null || true
    return "$status"
}

# A watchdog is only worth having if it fires, so prove it every run.
selftest_watchdog() {
    if run_with_timeout 1 sleep 30 >/dev/null 2>&1; then
        echo "run_stress: watchdog self-test FAILED — sleep 30 was not killed" >&2
        exit 1
    fi
    if [ ! -f "$marker" ]; then
        echo "run_stress: watchdog self-test FAILED — no timeout marker" >&2
        exit 1
    fi
    rm -f "$marker"
    echo "run_stress: watchdog ok (killed a 30s sleep after 1s)"
}

export_symbolizer() {
    if [ -z "${LLVM_SYMBOLIZER_PATH:-}" ]; then
        symbolizer=$(command -v llvm-symbolizer || true)
        if [ -n "$symbolizer" ]; then
            LLVM_SYMBOLIZER_PATH="$symbolizer"
            export LLVM_SYMBOLIZER_PATH
        fi
    fi
}

selftest_watchdog

case "$mode" in
plain)
    STRESS_ROUNDS="${STRESS_ROUNDS:-40}"
    timeout_s="${STRESS_TIMEOUT:-900}"
    binary=build/stress-parquet
    log=build/stress.log
    verdict=
    echo "run_stress: building $binary"
    # shellcheck disable=SC2086
    mojo build tests/stress_threaded_read.mojo $includes -o "$binary"
    ;;
tsan)
    STRESS_ROUNDS="${STRESS_ROUNDS:-3}"
    timeout_s="${STRESS_TIMEOUT:-1800}"
    binary=build/stress-parquet-tsan
    log=build/stress-tsan.log
    verdict="WARNING: ThreadSanitizer"
    echo "run_stress: building $binary with --sanitize thread"
    # shellcheck disable=SC2086
    mojo build --sanitize thread tests/stress_threaded_read.mojo $includes \
        -o "$binary"
    TSAN_OPTIONS="${TSAN_OPTIONS:-halt_on_error=1}"
    export TSAN_OPTIONS
    export_symbolizer
    echo "run_stress: TSAN_OPTIONS=$TSAN_OPTIONS"
    echo "run_stress: LLVM_SYMBOLIZER_PATH=${LLVM_SYMBOLIZER_PATH:-<unset>}"
    ;;
asan)
    STRESS_ROUNDS="${STRESS_ROUNDS:-3}"
    timeout_s="${STRESS_TIMEOUT:-1800}"
    binary=build/stress-parquet-asan
    log=build/stress-asan.log
    verdict="ERROR: (Address|Leak)Sanitizer"
    echo "run_stress: building $binary with --sanitize address"
    # shellcheck disable=SC2086
    mojo build --sanitize address tests/stress_threaded_read.mojo $includes \
        -o "$binary"
    ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1}"
    export ASAN_OPTIONS
    export_symbolizer
    echo "run_stress: ASAN_OPTIONS=$ASAN_OPTIONS"
    echo "run_stress: LLVM_SYMBOLIZER_PATH=${LLVM_SYMBOLIZER_PATH:-<unset>}"
    ;;
*)
    echo "usage: $0 [plain|tsan|asan]" >&2
    exit 2
    ;;
esac

export STRESS_ROUNDS
echo "run_stress: running $binary with STRESS_ROUNDS=$STRESS_ROUNDS (watchdog ${timeout_s}s)"
status=0
run_with_timeout "$timeout_s" "$binary" >"$log" 2>&1 || status=$?
cat "$log"

if [ -f "$marker" ]; then
    rm -f "$marker"
    echo "run_stress: FAILED — killed by the watchdog after ${timeout_s}s (a hung join?)" >&2
    exit 1
fi

# Not a race: the sanitizer never got as far as `main`.
if grep -q "TCMalloc assumes a 48-bit virtual address space" "$log"; then
    echo "run_stress: FAILED — the Mojo runtime's TCMalloc aborted during startup." >&2
    echo "run_stress: this is the known linux-64 + --sanitize thread limitation in" >&2
    echo "run_stress: Mojo 1.0.0 — a hello-world fails the same way. Run the" >&2
    echo "run_stress: ThreadSanitizer leg on osx-arm64." >&2
    exit 1
fi

if [ -n "$verdict" ] && grep -qE "$verdict" "$log"; then
    echo "run_stress: FAILED — the sanitizer reported a finding (see above)" >&2
    if [ "$mode" = tsan ]; then
        echo "run_stress: before reading these as races in parquet.mojo, run" >&2
        echo "run_stress:   pixi run -e stable stress-tsan-control" >&2
        echo "run_stress: and see the header of tests/tsan_control.mojo. On Mojo" >&2
        echo "run_stress: 1.0.0 the runtime allocator is invisible to TSan, and" >&2
        echo "run_stress: any program that allocates on two threads reports." >&2
    fi
    exit 1
fi

if [ "$status" -ne 0 ]; then
    echo "run_stress: FAILED — $binary exited $status" >&2
    exit "$status"
fi

if [ -n "$verdict" ]; then
    echo "run_stress: ok — no sanitizer findings"
else
    echo "run_stress: ok"
fi
