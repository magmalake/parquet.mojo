"""Does this machine have a Metal device the Mojo GPU stack can drive?

Exits non-zero when it does not, so `pixi run -e gpu gpu-probe` is a gate and
not a demo. Everything else in this directory assumes it passes.
"""

from std.sys import has_accelerator
from max.gpu.host import DeviceContext


def main() raises:
    """Print the device and fail if there isn't one.

    Raises:
        If no accelerator is present or the context will not open.
    """
    print("has_accelerator:", has_accelerator())
    if not has_accelerator():
        raise Error(
            "experiments.gpu_gather: no GPU accelerator — the spike needs Metal"
        )
    var ctx = DeviceContext()
    print("device:", ctx.name())
    print("api:", ctx.api())
