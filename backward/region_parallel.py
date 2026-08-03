"""Single-process, multi-device region-parallel native backward.

The LBI interface bottleneck makes the backward's expensive phases
region-independent (see `interface_state_jacobian_t_for_region` and
`native_region_backward`), so they can run concurrently on separate GPUs with only
the rank-r interface Jacobians crossing device boundaries. This driver realizes
that on one node with multiple CUDA devices:

  * Regions are round-robin placed on the given devices; the model is replicated
    per device (a copy of the weights), so region k's params live on its device.
  * Phase 1 (`A_k^T`) and Phase 3 (region param grads + canvas cotangent) run on
    each region's device, dispatched from one process without intervening syncs
    so that independent devices overlap.
  * Phase 2 (the suffix scan) gathers the small rank-r `A_k` to device 0, runs the
    serial scan, and scatters the boundary adjoints back.
  * Region parameter grads are LOCAL to their device (no gradient all-reduce,
    unlike data-parallel); only the canvas cotangent and shared-param partials are
    reduced across devices.

Weights are replicated per device (compute parallelism only, no memory
sharding); the multi-process NCCL runtime shares this structure with
collectives in place of the gather/scatter and reduction.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import torch

from backward.local_vjp import (
    NativeLocalVJPProvider,
    native_initial_backward,
    native_region_backward,
    reduce_region_results,
)
from backward.pullbacks import interface_state_jacobian_t_for_region
from backward.suffix_scan import propagate_state_adjoint_from_last_region_input


def move_to_device(obj: Any, device: torch.device | str) -> Any:
    """Recursively move all tensors inside tensors/dataclasses/lists/tuples/dicts to
    `device`, leaving other objects as-is."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj, **{f.name: move_to_device(getattr(obj, f.name), device) for f in dataclasses.fields(obj)}
        )
    if isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return type(obj)(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def build_replicas(base_model: torch.nn.Module, devices: list[int]) -> dict[int, torch.nn.Module]:
    """Replicate `base_model` (assumed on `devices[0]`) onto each device. The
    device-0 entry is the base model itself so its grads land in the caller's model."""
    replicas: dict[int, torch.nn.Module] = {devices[0]: base_model}
    for dev in devices[1:]:
        replicas[dev] = copy.deepcopy(base_model).to(f"cuda:{dev}")
    return replicas


def _sync(devices: list[int]) -> None:
    for dev in devices:
        torch.cuda.synchronize(dev)


def region_parallel_backward(
    *,
    base_model: torch.nn.Module,
    replicas: dict[int, torch.nn.Module],
    loss: torch.Tensor,
    cache: dict[str, Any],
    devices: list[int],
) -> dict[str, torch.Tensor | None]:
    """Full parameter gradient via multi-device region-parallel native backward.

    `cache`/`loss` come from a `forward_with_cache(..., native_backward=True)` on
    device 0. Returns the grad map (tensors on device 0), matching the sequential
    native backward. Regions are round-robin placed on `devices`."""
    dev0 = devices[0]
    region_caches = list(cache["region_caches"])
    num_regions = len(region_caches)
    states = list(cache["states"])
    region_devices = [devices[k % len(devices)] for k in range(num_regions)]

    provider = NativeLocalVJPProvider()
    grad_map = provider.new_grad_map(base_model)

    # --- Serial glue on device 0 (readout / seed / initial) ---
    provider.store_output_head_grads(model=base_model, loss=loss, grad_map=grad_map, cache=cache)
    g_last = provider.state_adjoint_from_loss(model=base_model, loss=loss, state=states[-2], cache=cache)
    last_g_region_output = torch.autograd.grad(
        loss, region_caches[-1].region_output, retain_graph=True, allow_unused=False
    )[0]

    # Place each region's cache on its device.
    region_caches_dev = [move_to_device(region_caches[k], f"cuda:{region_devices[k]}") for k in range(num_regions)]

    # --- Phase 1: interface Jacobians A_k, one per device, dispatched then synced ---
    a_k: list[torch.Tensor | None] = [None] * num_regions
    for k in range(num_regions):
        with torch.cuda.device(region_devices[k]):
            a_k[k] = interface_state_jacobian_t_for_region(
                model=replicas[region_devices[k]], region_cache=region_caches_dev[k]
            )
    _sync(devices)

    # --- Phase 2: suffix scan on device 0 (rank-r gather -> scan -> boundary adjoints) ---
    sjt = [a.to(dev0) for a in a_k]
    g_last = g_last.to(device=sjt[0].device, dtype=sjt[0].dtype)
    g_state_inputs = propagate_state_adjoint_from_last_region_input(sjt, g_last, num_regions=num_regions)

    # Initial-state glue (device 0) -> canvas partial folded into the reduction.
    init_grads, canvas_init = native_initial_backward(
        model=base_model, state0=states[0], state0_adjoint=g_state_inputs[0], cache=cache
    )
    for name, grad in init_grads.items():
        if name in grad_map:
            grad_map[name] = grad

    # --- Phase 3: region param grads + canvas, one per device, dispatched then synced ---
    parallel_cache = {"region_caches": region_caches_dev}
    results: list[Any] = [None] * num_regions
    for k in range(num_regions):
        dev = region_devices[k]
        with torch.cuda.device(dev):
            adjoints = [g.to(f"cuda:{dev}") for g in g_state_inputs]
            g_ro = last_g_region_output.to(f"cuda:{dev}") if k == num_regions - 1 else None
            results[k] = native_region_backward(
                model=replicas[dev], loss=None, states=None, region_index=k,
                num_regions=num_regions, state_adjoints=adjoints, cache=parallel_cache,
                g_region_output=g_ro,
            )
    _sync(devices)

    # --- Reduce across devices (grads local; canvas + shared reduced to device 0) ---
    results0 = [move_to_device(r, f"cuda:{dev0}") for r in results]
    reduced = reduce_region_results(results0)
    for name, grad in reduced.param_grads.items():
        if name in grad_map:
            grad_map[name] = grad

    canvas_total = reduced.canvas_cotangent
    if canvas_init is not None:
        canvas_init = canvas_init.to(f"cuda:{dev0}")
        canvas_total = canvas_init if canvas_total is None else canvas_total + canvas_init
    provider._canvas_grad = canvas_total
    provider._shared_grads = dict(reduced.shared_partials)
    provider.store_shared_canvas_grads(model=base_model, loss=loss, grad_map=grad_map, cache=cache)
    return grad_map
