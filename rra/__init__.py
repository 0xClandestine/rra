"""Resolution-Routed Attention (RRA) — MLX implementation."""
from .routed_patch import (
    reorder_model_weights_random,
    enable_random_pooling,
    disable_random_pooling,
    # backward compat aliases
    enable_rra_routed,
    disable_rra_routed,
)

__all__ = [
    "reorder_model_weights_random",
    "enable_random_pooling",
    "disable_random_pooling",
    "enable_rra_routed",
    "disable_rra_routed"
]
