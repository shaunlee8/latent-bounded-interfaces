from .autograd import AutogradEngine, autograd_backward_step
from .base import ADEngine, ADResult, GradMap, LBIBackwardResult, ScanBackpropModel, assign_grad_map, collect_named_grads, store_named_grads
from .local_vjp import (
    LocalVJPProvider,
    NativeLocalVJPProvider,
    RegionBackwardResult,
    TorchAutogradLocalVJPProvider,
    native_initial_backward,
    native_region_backward,
    reduce_region_results,
)
from .pullbacks import (
    ForwardModeInterfacePullbackProvider,
    InterfacePullbackProvider,
    NativeInterfacePullbackProvider,
    TorchGraphInterfacePullbackProvider,
    build_interface_pullback_provider,
    interface_state_jacobian_for_region_forward,
    interface_state_jacobian_t_for_region,
    materialize_interface_state_jacobian_t_graph,
    materialize_interface_state_jacobian_t_native,
)
from .scan import ScanADEngine, lbi_scan_backward_step
from .suffix_scan import (
    apply_jacobian_t,
    compose_suffix_jacobian_t,
    propagate_state_adjoint_from_last_region_input,
)

__all__ = [
    "ADEngine",
    "ADResult",
    "AutogradEngine",
    "GradMap",
    "ForwardModeInterfacePullbackProvider",
    "InterfacePullbackProvider",
    "interface_state_jacobian_for_region_forward",
    "LBIBackwardResult",
    "LocalVJPProvider",
    "NativeLocalVJPProvider",
    "ScanADEngine",
    "ScanBackpropModel",
    "TorchAutogradLocalVJPProvider",
    "NativeInterfacePullbackProvider",
    "RegionBackwardResult",
    "TorchGraphInterfacePullbackProvider",
    "assign_grad_map",
    "autograd_backward_step",
    "apply_jacobian_t",
    "build_interface_pullback_provider",
    "collect_named_grads",
    "compose_suffix_jacobian_t",
    "interface_state_jacobian_t_for_region",
    "lbi_scan_backward_step",
    "materialize_interface_state_jacobian_t_graph",
    "materialize_interface_state_jacobian_t_native",
    "native_initial_backward",
    "native_region_backward",
    "reduce_region_results",
    "propagate_state_adjoint_from_last_region_input",
    "store_named_grads",
]
