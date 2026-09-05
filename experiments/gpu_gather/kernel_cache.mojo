"""Process-lifetime store of compiled GPU kernels.

Lifted, almost unchanged, from `millfolio/engine`'s `src/runtime/kernel_cache.mojo`
(Apache-2.0, same author) — the working precedent for hand-written Mojo Metal
kernels on this hardware. Kept here so the spike can *measure* whether the
cache is needed for a Parquet-shaped workload rather than assume it.

`ctx.enqueue_function[k](...)` compiles the kernel fresh on every call: it runs
`compile_function` and then discards the `DeviceFunction`. For a decode loop
that dispatches once per page that is the Metal shader compiler on the hot
path. `cached_enqueue` holds the compiled `DeviceFunction` in a global slot
keyed by the kernel's mangled linkage name — unique per monomorphization, and
shape-agnostic because the layouts carry their dimensions as runtime values.
"""

from std.ffi import _Global
from max.gpu.host import DeviceContext
from max.gpu.host.dim import Dim
from std.builtin.device_passable import DevicePassable
from std.reflection import reflect_fn


def _empty_slot[T: Movable]() -> Optional[T]:
    """A global cache slot before its kernel has been compiled.

    Parameters:
        T: The compiled-function type the slot will hold.

    Returns:
        An empty slot.
    """
    return None


@always_inline
def cached_enqueue[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
    *actual_arg_types: DevicePassable,
](
    ctx: DeviceContext,
    *args: *actual_arg_types,
    grid_dim: Dim,
    block_dim: Dim,
) raises:
    """Enqueue `func`, reusing its process-lifetime compiled `DeviceFunction`.

    Drop-in for `ctx.enqueue_function[func](args..., grid_dim=, block_dim=)`,
    but compiles the Metal pipeline only on the first dispatch of this exact
    monomorphization.

    Parameters:
        declared_arg_types: The kernel's declared argument types.
        func: The kernel.
        actual_arg_types: The types of the arguments passed here.

    Args:
        ctx: The GPU device context.
        args: The kernel arguments.
        grid_dim: The grid dimensions.
        block_dim: The block dimensions.

    Raises:
        If compilation or the launch fails.
    """
    comptime FnT = type_of(ctx.compile_function[func]())
    comptime key = reflect_fn[func].linkage_name()
    var slot = _Global[key, _empty_slot[FnT]].get_or_create_ptr()
    if not slot[]:
        slot[] = ctx.compile_function[func]()
    ctx.enqueue_function(
        slot[].value(), *args, grid_dim=grid_dim, block_dim=block_dim
    )
