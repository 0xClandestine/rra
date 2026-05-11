"""
Supported model registry.

Add a model here before using it in an experiment. Pinning the exact
mlx-community ID ensures results are reproducible across machines.

Each entry declares:
  id       — mlx-community model ID (exact, copy from HuggingFace)
  family   — model family (used for grouping in figures)
  size     — parameter count string (for axis labels)
  attn     — attention variant: "mha" | "gqa" | "mqa" | "moe"
  notes    — anything worth knowing (sliding window, MoE routing, etc.)

Usage in an experiment:
    from experiments.models import MODELS, get
    model_id = get("qwen3.6-27b").id
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Model:
    id: str
    family: str
    size: str
    attn: Literal["mha", "gqa", "mqa", "moe"]
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry — add models here, one per line, grouped by family
# ---------------------------------------------------------------------------

_REGISTRY: list[Model] = [

    # ------------------------------------------------------------------
    # Qwen 2.5  (baseline family, all confirmed working)
    # ------------------------------------------------------------------
    Model("mlx-community/Qwen2.5-0.5B-Instruct-4bit", family="qwen2.5", size="0.5B", attn="gqa"),
    Model("mlx-community/Qwen2.5-7B-Instruct-4bit",   family="qwen2.5", size="7B",   attn="gqa"),

    # ------------------------------------------------------------------
    # Qwen 3.5  (no 3.6 models below 27B exist yet)
    # ------------------------------------------------------------------
    Model("mlx-community/Qwen3.5-0.8B-4bit", family="qwen3.5", size="0.8B", attn="gqa"),
    Model("mlx-community/Qwen3.5-2B-4bit",   family="qwen3.5", size="2B",   attn="gqa"),
    Model("mlx-community/Qwen3.5-9B-4bit",   family="qwen3.5", size="9B",   attn="gqa"),
    Model("mlx-community/Qwen3.5-27B-4bit",  family="qwen3.5", size="27B",  attn="gqa"),
    Model("mlx-community/Qwen3.5-35B-A3B-4bit", family="qwen3.5", size="35B", attn="moe",
          notes="MoE: 35B total, 3B active"),

    # ------------------------------------------------------------------
    # Gemma 4
    # ------------------------------------------------------------------
    Model("mlx-community/gemma-4-e2b-it-4bit",     family="gemma4", size="2B",  attn="gqa"),
    Model("mlx-community/gemma-4-e4b-it-4bit",     family="gemma4", size="4B",  attn="gqa"),
    Model("mlx-community/gemma-4-26b-a4b-it-4bit", family="gemma4", size="26B", attn="moe",
          notes="MoE: 26B total, 4B active"),
    Model("mlx-community/gemma-4-31b-it-4bit",     family="gemma4", size="31B", attn="gqa"),

    # ------------------------------------------------------------------
    # Llama 3.x  ← uncomment when ready to test
    # ------------------------------------------------------------------
    Model("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",  family="llama3", size="8B",  attn="gqa"),
    # Model("mlx-community/Meta-Llama-3.1-70B-Instruct-4bit", family="llama3", size="70B", attn="gqa"),

    # ------------------------------------------------------------------
    # Mistral  ← uncomment when ready to test
    # ------------------------------------------------------------------
    Model("mlx-community/Mistral-7B-Instruct-v0.3-4bit", family="mistral", size="7B", attn="gqa",
           notes="sliding window attention"),

]

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

MODELS: dict[str, Model] = {m.id: m for m in _REGISTRY}


def get(query: str) -> Model:
    """
    Look up a model by exact ID or by a substring of the ID.

    get("qwen2.5-0.5b")  →  Model(id="mlx-community/Qwen2.5-0.5B-Instruct-4bit", ...)
    """
    # exact match first
    if query in MODELS:
        return MODELS[query]
    # case-insensitive substring match
    matches = [m for m in _REGISTRY if query.lower() in m.id.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = "\n  ".join(m.id for m in matches)
        raise ValueError(f"Ambiguous query '{query}' matches:\n  {ids}")
    raise KeyError(f"No model matching '{query}'. Add it to experiments/models.py first.")


def by_family(family: str) -> list[Model]:
    """Return all models for a given family, sorted by size."""
    return [m for m in _REGISTRY if m.family == family]


def all_models() -> list[Model]:
    return list(_REGISTRY)


def models_for_mode(mode: str) -> list[Model]:
    """Return the model list for a named set.

    mode="all"      → every model in the registry
    mode=<family>   → all models for that family (e.g. "qwen3.5", "gemma4")
    """
    if mode == "all":
        return list(_REGISTRY)
    matches = by_family(mode)
    if not matches:
        known = sorted({m.family for m in _REGISTRY})
        raise ValueError(
            f"Unknown model set: {mode!r}. "
            f"Use 'all' or a family name: {known}"
        )
    return matches
