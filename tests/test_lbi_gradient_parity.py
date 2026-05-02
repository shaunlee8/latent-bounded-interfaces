from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import pytest
import torch

from backbones.general import BackboneSpec
from models.native_region_interface import NativeRegionInterfaceModel
from train.train_region_interface import _autograd_backward_step, _native_backward_step, _next_token_loss


_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _hybrid_pattern(repeats: int) -> tuple[str, ...]:
    if repeats <= 0:
        raise ValueError("hybrid repeats must be > 0")
    return ("mamba3", "mamba3", "mamba3", "transformer") * repeats


@dataclass(frozen=True)
class GradParityCaseSpec:
    name: str
    backbone_spec: BackboneSpec
    region_size: int
    vocab_size: int = 64
    message_dim: int = 8
    message_hidden_dim: int = 16
    batch_size: int = 2
    seq_len: int = 12
    device_type: str = "cpu"
    dtype_name: str = "float32"
    atol: float = 1e-5
    rtol: float = 1e-4
    loss_atol: float = 1e-7
    loss_rtol: float = 1e-6
    scan_rms_max: float = 1e-5
    seeds: tuple[int, ...] = (1234,)
    batches_per_seed: int = 1
    requires_cuda: bool = False
    required_modules: tuple[str, ...] = ()

    @property
    def backbone_name(self) -> str:
        return self.backbone_spec.name

    @property
    def region_mode(self) -> str:
        return "multi_region" if self.backbone_spec.layers > self.region_size else "one_region"

    def torch_device(self) -> torch.device:
        return torch.device(self.device_type)

    def torch_dtype(self) -> torch.dtype:
        try:
            return _DTYPE_BY_NAME[self.dtype_name]
        except KeyError as exc:
            supported = ", ".join(sorted(_DTYPE_BY_NAME.keys()))
            raise ValueError(f"unsupported dtype_name '{self.dtype_name}', expected one of: {supported}") from exc


@dataclass(frozen=True)
class ParameterGradParityMetrics:
    parameter_name: str
    shape: tuple[int, ...]
    has_ref_grad: bool
    has_native_grad: bool
    max_abs_diff: float
    max_rel_diff: float
    mean_rel_diff: float
    median_rel_diff: float
    within_tolerance: bool

    def to_row(self, *, case_name: str, seed: int, batch_index: int) -> dict[str, object]:
        return {
            "case_name": case_name,
            "seed": seed,
            "batch_index": batch_index,
            "parameter_name": self.parameter_name,
            "shape": "x".join(str(dim) for dim in self.shape),
            "has_ref_grad": self.has_ref_grad,
            "has_native_grad": self.has_native_grad,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "mean_rel_diff": self.mean_rel_diff,
            "median_rel_diff": self.median_rel_diff,
            "within_tolerance": self.within_tolerance,
        }


@dataclass(frozen=True)
class GradParityTrialMetrics:
    case_name: str
    backbone: str
    region_mode: str
    interface_jacobian_mode: str
    jacobian_basis_chunk: int
    device: str
    dtype: str
    seed: int
    batch_index: int
    loss_ref: float
    loss_native: float
    loss_abs_diff: float
    loss_rel_diff: float
    loss_within_tolerance: bool
    interface_scan_rms: float
    scan_within_tolerance: bool
    num_parameters_total: int
    num_parameters_with_grad: int
    num_parameters_within_tolerance: int
    fraction_parameters_within_tolerance: float
    global_grad_cosine: float
    relative_l2_error: float
    max_abs_grad_diff: float
    max_rel_grad_diff: float
    median_rel_grad_diff: float
    p95_rel_grad_diff: float
    all_gradients_within_tolerance: bool
    worst_parameter_by_abs: str
    worst_parameter_by_rel: str
    parameter_metrics: tuple[ParameterGradParityMetrics, ...]

    @property
    def trial_id(self) -> str:
        return f"seed{self.seed}_batch{self.batch_index}"

    @property
    def trial_passed(self) -> bool:
        return self.loss_within_tolerance and self.scan_within_tolerance and self.all_gradients_within_tolerance

    def to_summary_row(self) -> dict[str, object]:
        return {
            "case_name": self.case_name,
            "backbone": self.backbone,
            "region_mode": self.region_mode,
            "interface_jacobian_mode": self.interface_jacobian_mode,
            "jacobian_basis_chunk": self.jacobian_basis_chunk,
            "device": self.device,
            "dtype": self.dtype,
            "seed": self.seed,
            "batch_index": self.batch_index,
            "trial_id": self.trial_id,
            "loss_ref": self.loss_ref,
            "loss_native": self.loss_native,
            "loss_abs_diff": self.loss_abs_diff,
            "loss_rel_diff": self.loss_rel_diff,
            "loss_within_tolerance": self.loss_within_tolerance,
            "interface_scan_rms": self.interface_scan_rms,
            "scan_within_tolerance": self.scan_within_tolerance,
            "num_parameters_total": self.num_parameters_total,
            "num_parameters_with_grad": self.num_parameters_with_grad,
            "num_parameters_within_tolerance": self.num_parameters_within_tolerance,
            "fraction_parameters_within_tolerance": self.fraction_parameters_within_tolerance,
            "global_grad_cosine": self.global_grad_cosine,
            "relative_l2_error": self.relative_l2_error,
            "max_abs_grad_diff": self.max_abs_grad_diff,
            "max_rel_grad_diff": self.max_rel_grad_diff,
            "median_rel_grad_diff": self.median_rel_grad_diff,
            "p95_rel_grad_diff": self.p95_rel_grad_diff,
            "all_gradients_within_tolerance": self.all_gradients_within_tolerance,
            "trial_passed": self.trial_passed,
            "worst_parameter_by_abs": self.worst_parameter_by_abs,
            "worst_parameter_by_rel": self.worst_parameter_by_rel,
        }

    def per_parameter_rows(self) -> list[dict[str, object]]:
        return [
            parameter_metrics.to_row(case_name=self.case_name, seed=self.seed, batch_index=self.batch_index)
            for parameter_metrics in self.parameter_metrics
        ]


@dataclass(frozen=True)
class GradParityAggregateMetrics:
    case_name: str
    backbone: str
    region_mode: str
    interface_jacobian_mode: str
    jacobian_basis_chunk: int
    device: str
    dtype: str
    num_trials: int
    num_trials_passed: int
    fraction_trials_passed: float
    worst_loss_abs_diff: float
    worst_loss_rel_diff: float
    worst_interface_scan_rms: float
    min_global_grad_cosine: float
    max_relative_l2_error: float
    worst_max_abs_grad_diff: float
    worst_max_rel_grad_diff: float
    median_p95_rel_grad_diff: float
    worst_p95_rel_grad_diff: float
    median_fraction_parameters_within_tolerance: float
    all_trials_within_tolerance: bool
    worst_trial_by_abs: str
    worst_trial_by_rel: str
    trial_metrics: tuple[GradParityTrialMetrics, ...]

    def to_summary_row(self) -> dict[str, object]:
        return {
            "case_name": self.case_name,
            "backbone": self.backbone,
            "region_mode": self.region_mode,
            "interface_jacobian_mode": self.interface_jacobian_mode,
            "jacobian_basis_chunk": self.jacobian_basis_chunk,
            "device": self.device,
            "dtype": self.dtype,
            "num_trials": self.num_trials,
            "num_trials_passed": self.num_trials_passed,
            "fraction_trials_passed": self.fraction_trials_passed,
            "worst_loss_abs_diff": self.worst_loss_abs_diff,
            "worst_loss_rel_diff": self.worst_loss_rel_diff,
            "worst_interface_scan_rms": self.worst_interface_scan_rms,
            "min_global_grad_cosine": self.min_global_grad_cosine,
            "max_relative_l2_error": self.max_relative_l2_error,
            "worst_max_abs_grad_diff": self.worst_max_abs_grad_diff,
            "worst_max_rel_grad_diff": self.worst_max_rel_grad_diff,
            "median_p95_rel_grad_diff": self.median_p95_rel_grad_diff,
            "worst_p95_rel_grad_diff": self.worst_p95_rel_grad_diff,
            "median_fraction_parameters_within_tolerance": self.median_fraction_parameters_within_tolerance,
            "all_trials_within_tolerance": self.all_trials_within_tolerance,
            "worst_trial_by_abs": self.worst_trial_by_abs,
            "worst_trial_by_rel": self.worst_trial_by_rel,
        }


def _make_native_model(
    *,
    backbone_spec: BackboneSpec,
    vocab_size: int,
    region_size: int,
    message_dim: int,
    message_hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> NativeRegionInterfaceModel:
    model = NativeRegionInterfaceModel(
        vocab_size=vocab_size,
        region_size=region_size,
        message_dim=message_dim,
        backbone_spec=backbone_spec,
        message_hidden_dim=message_hidden_dim,
        message_scale_init=0.5,
    )
    return model.to(device=device, dtype=dtype)


def grad_parity_skip_reason(case: GradParityCaseSpec) -> str | None:
    for module_name in case.required_modules:
        if importlib.util.find_spec(module_name) is None:
            return f"{module_name} not installed"
    if case.requires_cuda and not torch.cuda.is_available():
        return f"CUDA required for {case.backbone_name} parity"
    return None


def _get_case_collection(preset: Literal["test", "report"]) -> tuple[GradParityCaseSpec, ...]:
    if preset == "test":
        return GRAD_PARITY_TEST_CASES
    if preset == "report":
        return GRAD_PARITY_REPORT_CASES
    raise ValueError(f"unsupported preset '{preset}'")


def get_grad_parity_cases(
    case_names: Sequence[str] | None = None,
    *,
    preset: Literal["test", "report"] = "test",
) -> list[GradParityCaseSpec]:
    cases = _get_case_collection(preset)
    if case_names is None:
        return list(cases)
    selected = []
    by_name = {case.name: case for case in cases}
    for case_name in case_names:
        try:
            selected.append(by_name[case_name])
        except KeyError as exc:
            available = ", ".join(sorted(by_name.keys()))
            raise ValueError(f"unknown grad parity case '{case_name}', expected one of: {available}") from exc
    return selected


def _relative_error(diff: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    denom = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-12)
    return diff / denom


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _compute_parameter_grad_metrics(
    grad_ref: dict[str, torch.Tensor | None],
    grad_native: dict[str, torch.Tensor | None],
    *,
    atol: float,
    rtol: float,
) -> tuple[ParameterGradParityMetrics, ...]:
    assert set(grad_ref.keys()) == set(grad_native.keys())
    rows: list[ParameterGradParityMetrics] = []
    for name in sorted(grad_ref.keys()):
        lhs = grad_ref[name]
        rhs = grad_native[name]
        has_ref_grad = lhs is not None
        has_native_grad = rhs is not None
        if lhs is None and rhs is None:
            rows.append(
                ParameterGradParityMetrics(
                    parameter_name=name,
                    shape=(),
                    has_ref_grad=False,
                    has_native_grad=False,
                    max_abs_diff=0.0,
                    max_rel_diff=0.0,
                    mean_rel_diff=0.0,
                    median_rel_diff=0.0,
                    within_tolerance=True,
                )
            )
            continue
        if lhs is None or rhs is None:
            shape = tuple(rhs.shape if lhs is None else lhs.shape)
            rows.append(
                ParameterGradParityMetrics(
                    parameter_name=name,
                    shape=shape,
                    has_ref_grad=has_ref_grad,
                    has_native_grad=has_native_grad,
                    max_abs_diff=float("inf"),
                    max_rel_diff=float("inf"),
                    mean_rel_diff=float("inf"),
                    median_rel_diff=float("inf"),
                    within_tolerance=False,
                )
            )
            continue
        assert lhs.shape == rhs.shape, f"gradient shape mismatch for parameter {name}: {lhs.shape} vs {rhs.shape}"
        lhs_f = lhs.detach().to(dtype=torch.float32)
        rhs_f = rhs.detach().to(dtype=torch.float32)
        abs_diff = (lhs_f - rhs_f).abs()
        rel_diff = _relative_error(abs_diff, lhs_f, rhs_f)
        rows.append(
            ParameterGradParityMetrics(
                parameter_name=name,
                shape=tuple(lhs.shape),
                has_ref_grad=True,
                has_native_grad=True,
                max_abs_diff=float(abs_diff.max().item()),
                max_rel_diff=float(rel_diff.max().item()),
                mean_rel_diff=float(rel_diff.mean().item()),
                median_rel_diff=float(rel_diff.median().item()),
                within_tolerance=bool(torch.allclose(lhs, rhs, atol=atol, rtol=rtol)),
            )
        )
    return tuple(rows)


def _compute_global_grad_metrics(
    grad_ref: dict[str, torch.Tensor | None],
    grad_native: dict[str, torch.Tensor | None],
) -> tuple[float, float]:
    flat_ref: list[torch.Tensor] = []
    flat_native: list[torch.Tensor] = []
    for name in sorted(grad_ref.keys()):
        lhs = grad_ref[name]
        rhs = grad_native[name]
        if lhs is None or rhs is None:
            continue
        flat_ref.append(lhs.detach().to(dtype=torch.float32).reshape(-1))
        flat_native.append(rhs.detach().to(dtype=torch.float32).reshape(-1))
    if not flat_ref:
        return 1.0, 0.0
    ref_vec = torch.cat(flat_ref)
    native_vec = torch.cat(flat_native)
    ref_norm = ref_vec.norm()
    native_norm = native_vec.norm()
    diff_norm = (native_vec - ref_vec).norm()
    if ref_norm.item() == 0.0 and native_norm.item() == 0.0:
        cosine = 1.0
    elif ref_norm.item() == 0.0 or native_norm.item() == 0.0:
        cosine = 0.0
    else:
        cosine = float(torch.dot(ref_vec, native_vec).item() / (ref_norm.item() * native_norm.item()))
    relative_l2 = float(diff_norm.item() / max(ref_norm.item(), 1e-12))
    return cosine, relative_l2


def _sample_trial_batch(
    *,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    batch_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_gen = torch.Generator(device="cpu")
    sample_gen.manual_seed(seed * 1000 + batch_index)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, generator=sample_gen)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, generator=sample_gen)
    return input_ids.to(device=device), targets.to(device=device)


def compute_grad_parity_metrics(
    case: GradParityCaseSpec,
    *,
    seed: int | None = None,
    batch_index: int = 0,
    interface_jacobian_mode: Literal["graph", "recompute"] = "graph",
    jacobian_basis_chunk: int = 1,
) -> GradParityTrialMetrics:
    skip_reason = grad_parity_skip_reason(case)
    if skip_reason is not None:
        raise RuntimeError(f"cannot execute grad parity case '{case.name}': {skip_reason}")
    if interface_jacobian_mode not in {"graph", "recompute"}:
        raise ValueError("interface_jacobian_mode must be one of: graph, recompute")
    if jacobian_basis_chunk <= 0:
        raise ValueError("jacobian_basis_chunk must be > 0")

    if seed is None:
        if len(case.seeds) != 1:
            raise ValueError(f"case '{case.name}' has multiple seeds; pass seed explicitly")
        seed = case.seeds[0]

    device = case.torch_device()
    dtype = case.torch_dtype()
    torch.manual_seed(seed)

    model_ref = _make_native_model(
        backbone_spec=case.backbone_spec,
        vocab_size=case.vocab_size,
        region_size=case.region_size,
        message_dim=case.message_dim,
        message_hidden_dim=case.message_hidden_dim,
        device=device,
        dtype=dtype,
    )
    model_native = _make_native_model(
        backbone_spec=case.backbone_spec,
        vocab_size=case.vocab_size,
        region_size=case.region_size,
        message_dim=case.message_dim,
        message_hidden_dim=case.message_hidden_dim,
        device=device,
        dtype=dtype,
    )
    model_native.load_state_dict(model_ref.state_dict())
    model_ref.train()
    model_native.train()

    input_ids, targets = _sample_trial_batch(
        vocab_size=case.vocab_size,
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        seed=seed,
        batch_index=batch_index,
        device=device,
    )

    logits_ref, _ = model_ref.forward_with_cache(input_ids)
    ce_loss_ref = _next_token_loss(logits_ref, targets)

    logits_native, cache_native = model_native.forward_with_cache(input_ids)
    ce_loss_native = _next_token_loss(logits_native, targets)

    grad_ref = _autograd_backward_step(model_ref, ce_loss=ce_loss_ref)
    native_backward = _native_backward_step(
        model_native,
        ce_loss=ce_loss_native,
        cache=cache_native,
        interface_jacobian_mode=interface_jacobian_mode,
        jacobian_basis_chunk=jacobian_basis_chunk,
    )
    grad_native = native_backward.grad_map

    parameter_metrics = _compute_parameter_grad_metrics(grad_ref, grad_native, atol=case.atol, rtol=case.rtol)
    grad_rows_with_tensors = [row for row in parameter_metrics if row.has_ref_grad and row.has_native_grad]
    grad_rows_out_of_tolerance = [row for row in grad_rows_with_tensors if not row.within_tolerance]

    max_abs_row = max(grad_rows_with_tensors, key=lambda row: row.max_abs_diff, default=None)
    max_rel_row = max(grad_rows_with_tensors, key=lambda row: row.max_rel_diff, default=None)
    rel_values = [row.max_rel_diff for row in grad_rows_with_tensors]

    loss_ref_value = float(ce_loss_ref.detach().float().item())
    loss_native_value = float(ce_loss_native.detach().float().item())
    loss_abs_diff = abs(loss_ref_value - loss_native_value)
    loss_rel_diff = loss_abs_diff / max(abs(loss_ref_value), abs(loss_native_value), 1e-12)
    loss_within_tolerance = bool(
        torch.allclose(
            ce_loss_ref.detach(),
            ce_loss_native.detach(),
            atol=case.loss_atol,
            rtol=case.loss_rtol,
        )
    )

    interface_scan_rms = float(native_backward.interface_scan_rms)
    if case.region_mode == "multi_region":
        scan_within_tolerance = interface_scan_rms < case.scan_rms_max
    else:
        scan_within_tolerance = interface_scan_rms == 0.0

    num_parameters_total = len(parameter_metrics)
    num_parameters_with_grad = len(grad_rows_with_tensors)
    num_parameters_within_tolerance = sum(1 for row in parameter_metrics if row.within_tolerance)
    fraction_parameters_within_tolerance = (
        num_parameters_within_tolerance / num_parameters_total if num_parameters_total > 0 else 1.0
    )
    global_grad_cosine, relative_l2_error = _compute_global_grad_metrics(grad_ref, grad_native)

    return GradParityTrialMetrics(
        case_name=case.name,
        backbone=case.backbone_name,
        region_mode=case.region_mode,
        interface_jacobian_mode=interface_jacobian_mode,
        jacobian_basis_chunk=jacobian_basis_chunk,
        device=case.device_type,
        dtype=case.dtype_name,
        seed=seed,
        batch_index=batch_index,
        loss_ref=loss_ref_value,
        loss_native=loss_native_value,
        loss_abs_diff=loss_abs_diff,
        loss_rel_diff=loss_rel_diff,
        loss_within_tolerance=loss_within_tolerance,
        interface_scan_rms=interface_scan_rms,
        scan_within_tolerance=scan_within_tolerance,
        num_parameters_total=num_parameters_total,
        num_parameters_with_grad=num_parameters_with_grad,
        num_parameters_within_tolerance=num_parameters_within_tolerance,
        fraction_parameters_within_tolerance=fraction_parameters_within_tolerance,
        global_grad_cosine=global_grad_cosine,
        relative_l2_error=relative_l2_error,
        max_abs_grad_diff=max_abs_row.max_abs_diff if max_abs_row is not None else 0.0,
        max_rel_grad_diff=max_rel_row.max_rel_diff if max_rel_row is not None else 0.0,
        median_rel_grad_diff=_quantile(rel_values, 0.5),
        p95_rel_grad_diff=_quantile(rel_values, 0.95),
        all_gradients_within_tolerance=(len(grad_rows_out_of_tolerance) == 0),
        worst_parameter_by_abs=max_abs_row.parameter_name if max_abs_row is not None else "",
        worst_parameter_by_rel=max_rel_row.parameter_name if max_rel_row is not None else "",
        parameter_metrics=parameter_metrics,
    )


def compute_grad_parity_case_aggregate(
    case: GradParityCaseSpec,
    *,
    interface_jacobian_mode: Literal["graph", "recompute"] = "graph",
    jacobian_basis_chunk: int = 1,
) -> GradParityAggregateMetrics:
    trials = tuple(
        compute_grad_parity_metrics(
            case,
            seed=seed,
            batch_index=batch_index,
            interface_jacobian_mode=interface_jacobian_mode,
            jacobian_basis_chunk=jacobian_basis_chunk,
        )
        for seed in case.seeds
        for batch_index in range(case.batches_per_seed)
    )
    worst_abs_trial = max(trials, key=lambda trial: trial.max_abs_grad_diff)
    worst_rel_trial = max(trials, key=lambda trial: trial.max_rel_grad_diff)
    p95_values = [trial.p95_rel_grad_diff for trial in trials]
    frac_values = [trial.fraction_parameters_within_tolerance for trial in trials]
    num_trials_passed = sum(1 for trial in trials if trial.trial_passed)
    return GradParityAggregateMetrics(
        case_name=case.name,
        backbone=case.backbone_name,
        region_mode=case.region_mode,
        interface_jacobian_mode=interface_jacobian_mode,
        jacobian_basis_chunk=jacobian_basis_chunk,
        device=case.device_type,
        dtype=case.dtype_name,
        num_trials=len(trials),
        num_trials_passed=num_trials_passed,
        fraction_trials_passed=(num_trials_passed / len(trials)) if trials else 1.0,
        worst_loss_abs_diff=max(trial.loss_abs_diff for trial in trials),
        worst_loss_rel_diff=max(trial.loss_rel_diff for trial in trials),
        worst_interface_scan_rms=max(trial.interface_scan_rms for trial in trials),
        min_global_grad_cosine=min(trial.global_grad_cosine for trial in trials),
        max_relative_l2_error=max(trial.relative_l2_error for trial in trials),
        worst_max_abs_grad_diff=worst_abs_trial.max_abs_grad_diff,
        worst_max_rel_grad_diff=worst_rel_trial.max_rel_grad_diff,
        median_p95_rel_grad_diff=_quantile(p95_values, 0.5),
        worst_p95_rel_grad_diff=max(p95_values) if p95_values else 0.0,
        median_fraction_parameters_within_tolerance=_quantile(frac_values, 0.5),
        all_trials_within_tolerance=(num_trials_passed == len(trials)),
        worst_trial_by_abs=worst_abs_trial.trial_id,
        worst_trial_by_rel=worst_rel_trial.trial_id,
        trial_metrics=trials,
    )


def assert_grad_parity_metrics(case: GradParityCaseSpec, metrics: GradParityTrialMetrics) -> None:
    assert metrics.loss_within_tolerance, (
        f"{case.name}[{metrics.trial_id}]: loss mismatch exceeds tolerance "
        f"(abs_diff={metrics.loss_abs_diff}, rel_diff={metrics.loss_rel_diff})"
    )
    assert metrics.scan_within_tolerance, (
        f"{case.name}[{metrics.trial_id}]: interface scan RMS exceeds tolerance "
        f"(scan_rms={metrics.interface_scan_rms}, limit={case.scan_rms_max})"
    )
    assert metrics.all_gradients_within_tolerance, (
        f"{case.name}[{metrics.trial_id}]: gradient parity failed "
        f"(worst_abs={metrics.worst_parameter_by_abs}, worst_rel={metrics.worst_parameter_by_rel}, "
        f"max_abs={metrics.max_abs_grad_diff}, max_rel={metrics.max_rel_grad_diff})"
    )


def assert_grad_parity_aggregate(case: GradParityCaseSpec, aggregate: GradParityAggregateMetrics) -> None:
    assert aggregate.all_trials_within_tolerance, (
        f"{case.name}: not all trials passed parity checks "
        f"(passed={aggregate.num_trials_passed}/{aggregate.num_trials}, "
        f"worst_abs_trial={aggregate.worst_trial_by_abs}, worst_rel_trial={aggregate.worst_trial_by_rel}, "
        f"worst_max_abs={aggregate.worst_max_abs_grad_diff}, worst_max_rel={aggregate.worst_max_rel_grad_diff})"
    )


GRAD_PARITY_TEST_CASES = (
    GradParityCaseSpec(
        name="transformer_one_region",
        backbone_spec=BackboneSpec(
            name="transformer",
            dim=32,
            layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_intermediate=64,
            attn_head_dim=8,
        ),
        region_size=2,
        message_dim=12,
        message_hidden_dim=24,
        vocab_size=96,
    ),
    GradParityCaseSpec(
        name="transformer_multi_region",
        backbone_spec=BackboneSpec(
            name="transformer",
            dim=32,
            layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_intermediate=64,
            attn_head_dim=8,
        ),
        region_size=1,
        message_dim=12,
        message_hidden_dim=24,
        vocab_size=96,
    ),
    GradParityCaseSpec(
        name="mamba2_one_region",
        backbone_spec=BackboneSpec(
            name="mamba2",
            dim=16,
            layers=2,
            d_state=4,
            d_conv=4,
            expand=2,
            headdim=4,
            ngroups=1,
            chunk_size=16,
            use_mem_eff_path=False,
        ),
        region_size=2,
        device_type="cuda",
        requires_cuda=True,
        required_modules=("triton", "einops"),
    ),
    GradParityCaseSpec(
        name="mamba2_multi_region",
        backbone_spec=BackboneSpec(
            name="mamba2",
            dim=16,
            layers=2,
            d_state=4,
            d_conv=4,
            expand=2,
            headdim=4,
            ngroups=1,
            chunk_size=16,
            use_mem_eff_path=False,
        ),
        region_size=1,
        device_type="cuda",
        requires_cuda=True,
        required_modules=("triton", "einops"),
    ),
    GradParityCaseSpec(
        name="mamba3_one_region",
        backbone_spec=BackboneSpec(
            name="mamba3",
            dim=16,
            layers=2,
            d_state=16,
            expand=2,
            headdim=16,
            ngroups=1,
            chunk_size=16,
        ),
        region_size=2,
        device_type="cuda",
        dtype_name="bfloat16",
        atol=5e-2,
        rtol=5e-2,
        loss_atol=5e-3,
        loss_rtol=5e-3,
        requires_cuda=True,
        required_modules=("triton", "einops"),
    ),
    GradParityCaseSpec(
        name="mamba3_multi_region",
        backbone_spec=BackboneSpec(
            name="mamba3",
            dim=16,
            layers=2,
            d_state=16,
            expand=2,
            headdim=16,
            ngroups=1,
            chunk_size=16,
        ),
        region_size=1,
        device_type="cuda",
        dtype_name="bfloat16",
        atol=5e-2,
        rtol=5e-2,
        loss_atol=5e-3,
        loss_rtol=5e-3,
        scan_rms_max=5e-5,
        requires_cuda=True,
        required_modules=("triton", "einops"),
    ),
    GradParityCaseSpec(
        name="hybrid_one_region",
        backbone_spec=BackboneSpec(
            name="hybrid",
            dim=32,
            layers=4,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=16,
            ngroups=1,
            chunk_size=16,
            n_heads=4,
            n_kv_heads=2,
            d_intermediate=128,
            attn_head_dim=8,
            layer_types=_hybrid_pattern(1),
        ),
        region_size=4,
        vocab_size=96,
        message_dim=12,
        message_hidden_dim=24,
        device_type="cuda",
        dtype_name="bfloat16",
        atol=5e-2,
        rtol=5e-2,
        loss_atol=5e-3,
        loss_rtol=5e-3,
        requires_cuda=True,
        required_modules=("triton", "einops"),
    ),
    GradParityCaseSpec(
        name="hybrid_multi_region",
        backbone_spec=BackboneSpec(
            name="hybrid",
            dim=32,
            layers=4,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=16,
            ngroups=1,
            chunk_size=16,
            n_heads=4,
            n_kv_heads=2,
            d_intermediate=128,
            attn_head_dim=8,
            layer_types=_hybrid_pattern(1),
        ),
        region_size=2,
        vocab_size=96,
        message_dim=12,
        message_hidden_dim=24,
        device_type="cuda",
        dtype_name="bfloat16",
        atol=5e-2,
        rtol=5e-2,
        loss_atol=5e-3,
        loss_rtol=5e-3,
        scan_rms_max=5e-4,
        requires_cuda=True,
        required_modules=("triton", "einops"),
    ),
)


GRAD_PARITY_REPORT_CASES = (
    GradParityCaseSpec(
        name="transformer_one_region",
        backbone_spec=BackboneSpec(
            name="transformer",
            dim=128,
            layers=6,
            n_heads=4,
            n_kv_heads=2,
            d_intermediate=256,
            attn_head_dim=32,
        ),
        region_size=6,
        vocab_size=256,
        message_dim=32,
        message_hidden_dim=64,
        batch_size=4,
        seq_len=64,
        seeds=(1234, 2345, 3456),
        batches_per_seed=2,
    ),
    GradParityCaseSpec(
        name="transformer_multi_region",
        backbone_spec=BackboneSpec(
            name="transformer",
            dim=128,
            layers=6,
            n_heads=4,
            n_kv_heads=2,
            d_intermediate=256,
            attn_head_dim=32,
        ),
        region_size=2,
        vocab_size=256,
        message_dim=32,
        message_hidden_dim=64,
        batch_size=4,
        seq_len=64,
        seeds=(1234, 2345, 3456),
        batches_per_seed=2,
        scan_rms_max=5e-5,
    ),
    GradParityCaseSpec(
        name="mamba2_one_region",
        backbone_spec=BackboneSpec(
            name="mamba2",
            dim=64,
            layers=4,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=8,
            ngroups=1,
            chunk_size=32,
            use_mem_eff_path=False,
        ),
        region_size=4,
        vocab_size=256,
        message_dim=32,
        message_hidden_dim=64,
        batch_size=4,
        seq_len=64,
        device_type="cuda",
        dtype_name="bfloat16",
        atol=5e-2,
        rtol=5e-2,
        loss_atol=5e-3,
        loss_rtol=5e-3,
        seeds=(1234, 2345, 3456),
        batches_per_seed=2,
        requires_cuda=True,
        required_modules=("triton", "einops"),
    ),
    GradParityCaseSpec(
        name="mamba2_multi_region",
        backbone_spec=BackboneSpec(
            name="mamba2",
            dim=64,
            layers=4,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=8,
            ngroups=1,
            chunk_size=32,
            use_mem_eff_path=False,
        ),
        region_size=2,
        vocab_size=256,
        message_dim=32,
        message_hidden_dim=64,
        batch_size=4,
        seq_len=64,
        device_type="cuda",
        seeds=(1234, 2345, 3456),
        batches_per_seed=2,
        scan_rms_max=5e-5,
        requires_cuda=True,
        required_modules=("triton", "einops"),
    ),
    GradParityCaseSpec(
        name="mamba3_one_region",
        backbone_spec=BackboneSpec(
            name="mamba3",
            dim=64,
            layers=4,
            d_state=16,
            expand=2,
            headdim=16,
            ngroups=1,
            chunk_size=32,
        ),
        region_size=4,
        vocab_size=256,
        message_dim=32,
        message_hidden_dim=64,
        batch_size=4,
        seq_len=64,
        device_type="cuda",
        dtype_name="bfloat16",
        atol=5e-2,
        rtol=5e-2,
        loss_atol=5e-3,
        loss_rtol=5e-3,
        seeds=(1234, 2345, 3456),
        batches_per_seed=2,
        requires_cuda=True,
        required_modules=("triton", "einops"),
    ),
    GradParityCaseSpec(
        name="mamba3_multi_region",
        backbone_spec=BackboneSpec(
            name="mamba3",
            dim=64,
            layers=4,
            d_state=16,
            expand=2,
            headdim=16,
            ngroups=1,
            chunk_size=32,
        ),
        region_size=2,
        vocab_size=256,
        message_dim=32,
        message_hidden_dim=64,
        batch_size=4,
        seq_len=64,
        device_type="cuda",
        dtype_name="bfloat16",
        atol=5e-2,
        rtol=5e-2,
        loss_atol=5e-3,
        loss_rtol=5e-3,
        scan_rms_max=1e-4,
        seeds=(1234, 2345, 3456),
        batches_per_seed=2,
        requires_cuda=True,
        required_modules=("triton", "einops"),
    ),
    GradParityCaseSpec(
        name="hybrid_one_region",
        backbone_spec=BackboneSpec(
            name="hybrid",
            dim=64,
            layers=8,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=16,
            ngroups=1,
            chunk_size=32,
            n_heads=4,
            n_kv_heads=2,
            d_intermediate=256,
            attn_head_dim=16,
            layer_types=_hybrid_pattern(2),
        ),
        region_size=8,
        vocab_size=256,
        message_dim=32,
        message_hidden_dim=64,
        batch_size=4,
        seq_len=64,
        device_type="cuda",
        dtype_name="bfloat16",
        atol=5e-2,
        rtol=5e-2,
        loss_atol=5e-3,
        loss_rtol=5e-3,
        seeds=(1234, 2345, 3456),
        batches_per_seed=2,
        requires_cuda=True,
        required_modules=("triton", "einops"),
    ),
    GradParityCaseSpec(
        name="hybrid_multi_region",
        backbone_spec=BackboneSpec(
            name="hybrid",
            dim=64,
            layers=8,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=16,
            ngroups=1,
            chunk_size=32,
            n_heads=4,
            n_kv_heads=2,
            d_intermediate=256,
            attn_head_dim=16,
            layer_types=_hybrid_pattern(2),
        ),
        region_size=2,
        vocab_size=256,
        message_dim=32,
        message_hidden_dim=64,
        batch_size=4,
        seq_len=64,
        device_type="cuda",
        dtype_name="bfloat16",
        atol=5e-2,
        rtol=5e-2,
        loss_atol=5e-3,
        loss_rtol=5e-3,
        seeds=(1234, 2345, 3456),
        batches_per_seed=2,
        scan_rms_max=5e-4,
        requires_cuda=True,
        required_modules=("triton", "einops"),
    ),
)


@pytest.mark.parametrize("case", GRAD_PARITY_TEST_CASES, ids=lambda case: case.name)
def test_grad_parity_case(case: GradParityCaseSpec) -> None:
    skip_reason = grad_parity_skip_reason(case)
    if skip_reason is not None:
        pytest.skip(skip_reason)
    aggregate = compute_grad_parity_case_aggregate(case)
    assert_grad_parity_aggregate(case, aggregate)
