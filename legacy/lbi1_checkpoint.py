from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch


_PREFIX_MIGRATIONS = (
    ("embedding.", "canvas.embedding."),
    ("norm.", "readout.norm."),
    ("lm_head.", "readout.lm_head."),
    ("backbone.", "region_backend.backbone."),
    ("blocks.", "region_backend.backbone.blocks."),
    ("input_to_message.", "interface.initial_encoder."),
    ("message_to_hidden.", "interface.decoders."),
    ("hidden_to_message.", "interface.encoders."),
    ("message_norm.", "interface.norms."),
)


def migrate_lbi1_state_key(key: str) -> str:
    """Map one released LBI-1 model state key into the LBI-2 owned schema."""
    if key == "message_alpha":
        return "interface.update_scale"
    for old_prefix, new_prefix in _PREFIX_MIGRATIONS:
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix) :]
    return key


def migrate_lbi1_state_dict(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Translate an LBI-1 state dict for strict loading into ``LBILanguageModel``."""
    migrated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        new_key = migrate_lbi1_state_key(key)
        if new_key in migrated:
            if torch.equal(migrated[new_key], value):
                continue
            raise ValueError(f"LBI-1 migration produced duplicate key with different value: {new_key}")
        migrated[new_key] = value
    return migrated
