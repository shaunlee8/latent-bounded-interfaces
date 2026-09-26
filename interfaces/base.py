from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class InterfaceSpec:
    """Static state and region-conditioning dimensions for an interface."""

    state_flat_dim: int
    region_condition_dim: int


@dataclass
class InterfaceStep:
    """Result of one boundary update: the new state and the intermediates the
    Jacobian applications reuse."""

    state: torch.Tensor
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


class InterfaceModule(nn.Module, ABC):
    """Boundary-state interface between canvas features and region backends."""

    spec: InterfaceSpec

    @abstractmethod
    def validate_region_count(self, num_regions: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def initialize(self, canvas_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(self, state: torch.Tensor, region_index: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        state: torch.Tensor,
        region_features: torch.Tensor,
        region_index: int,
    ) -> InterfaceStep:
        raise NotImplementedError

    def initial_vjp_parameters(self) -> list[nn.Parameter]:
        """Parameters touched by the initial-state VJP."""
        return []

    def region_vjp_parameters(self, region_index: int) -> list[nn.Parameter]:
        """Parameters touched by the per-region interface VJP."""
        del region_index
        return []

    def shared_vjp_parameters(self) -> list[nn.Parameter]:
        """Interface parameters shared across regions and touched by local VJPs."""
        return []

    def apply_update_jacobian_t_to_region_output(
        self,
        *,
        region_cache: Any,
        state_out_cotangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the update-side Jacobian transpose to state cotangent basis vectors."""
        raise NotImplementedError(f"{type(self).__name__} does not expose structured update pullbacks.")

    def apply_decode_jacobian_t_to_state_input(
        self,
        *,
        region_cache: Any,
        region_input_cotangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the decode-side Jacobian transpose to a basis of region-input cotangents."""
        raise NotImplementedError(f"{type(self).__name__} does not expose structured decode pullbacks.")
