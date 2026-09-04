"""The control for `stress-tsan`: is a report a race here, or in the runtime?

    pixi run -e stable stress-tsan-control

`tests/run_stress.sh tsan` reports data races against the threaded read path.
This program exists so that the next person does not have to take anyone's word
for what they mean. It contains no parquet, no thrift, no shared data and no
pointer arithmetic — three `parallel_for` bodies over per-task slots, differing
only in whether they allocate:

| mode    | what a task does                              | TSan on Mojo 1.0.0 |
| ------- | --------------------------------------------- | ------------------ |
| `read`  | reads its own slot                            | clean              |
| `write` | writes its own slot                           | clean              |
| `alloc` | grows and drops `List`s only it ever touches  | reports races      |

Allocation is the whole difference. The Mojo 1.0.0 runtime does not route its
allocator through the `malloc`/`free` ThreadSanitizer intercepts, so the tool
never learns that a block was freed on one thread before being handed to
another; a recycled address then looks like two threads touching one location.
The reports over parquet decode carry the same signature — TSan prints no
"Location is heap block …" line for any of them, and pairs accesses of
different widths (a 2-byte field against an 8-byte buffer append) at one
address, which is two objects at one address rather than one object in a race.

For contrast, threads.mojo 0.4.0's own `stress-tsan` — whose tasks touch
preallocated cells and never allocate — is clean on the same machine with the
same toolchain. So the setup works; it is allocation the tool cannot follow.

Run all three modes; `MODE` picks one.
"""

from std.os import getenv
from threads import parallel_for


struct Slots(Movable):
    """One `Int` per task, and nothing else."""

    var sink: List[Int]

    def __init__(out self, n: Int):
        self.sink = List[Int](length=n, fill=0)

    def __init__(out self, *, deinit move: Self):
        self.sink = move.sink^


def _read_only(i: Int, mut s: Slots) -> None:
    var total = 0
    for _ in range(200):
        total += s.sink[i]
    if total == -1:
        s.sink[i] = 0


def _write_only(i: Int, mut s: Slots) -> None:
    for k in range(200):
        s.sink[i] = k


def _allocate(i: Int, mut s: Slots) -> None:
    var total = 0
    for r in range(60):
        var buf = List[UInt8]()
        for k in range(700 + r):
            buf.append(UInt8(k & 0xFF))
        total += len(buf)
        # `buf` dies here and its block goes back to the allocator, which may
        # hand it to another worker mid-flight. That is the situation TSan
        # cannot follow.
    s.sink[i] = total


def _run(mode: StringSlice, rounds: Int) raises:
    var s = Slots(10)
    for _ in range(rounds):
        if mode == "read":
            parallel_for[_read_only](10, s, num_workers=10)
        elif mode == "write":
            parallel_for[_write_only](10, s, num_workers=10)
        else:
            parallel_for[_allocate](10, s, num_workers=10)
    var total = 0
    for i in range(len(s.sink)):
        total += s.sink[i]
    print("control: mode=", mode, " total=", total, sep="")


def main() raises:
    var rounds = 40
    var env_rounds = getenv("STRESS_ROUNDS")
    if env_rounds != "":
        rounds = Int(env_rounds)
    var mode = getenv("MODE")
    if mode != "":
        _run(mode, rounds)
        return
    for m in [String("read"), String("write"), String("alloc")]:
        _run(m, rounds)
