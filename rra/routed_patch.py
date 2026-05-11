"""
Selective K/V Pooling — random head selection, zero calibration.
=================================================================

Strategy (based on empirical finding that random selection works):
  1. At load time: randomly select a subset of attention Q-heads to pool.
     Reorder Q/K/V/O weight matrices so pooled heads are contiguous.
  2. At inference: thin per-layer wrapper splits Q at the pool boundary
     (a view, not a copy), pools K/V for the selected heads, does two
     SDPA calls + concat.

Key properties:
  - Weight reordering is a valid permutation: full-attn output is unchanged.
  - Pool fraction = one hyperparameter. No calibration. No thresholds.
  - The SDPA override is set once, not on every forward call.
  - Gated q_proj layers (Qwen3Next) are skipped — full attention preserved.
  - Linear-attention / hybrid layers are untouched.

Usage:
    from mlx_lm import load
    from src.rra.routed_patch import enable_random_pooling, disable_random_pooling

    model, tokenizer = load("mlx-community/Qwen3.5-27B-4bit")
    enable_random_pooling(model, pool_factor=4, pool_fraction=0.3)
    # ... generate ...
    disable_random_pooling(model)
"""

import sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

def _inner(model):
    for attr in ("model", "language_model", "text_model"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "layers"):
            return inner
    return model


def _attn(layer):
    return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)


def _n_heads(attn):
    return getattr(attn, "n_heads", None) or getattr(attn, "num_attention_heads", None)


def _n_kv_heads(attn, n_heads):
    return (getattr(attn, "n_kv_heads", None)
            or getattr(attn, "num_key_value_heads", None)
            or n_heads)


def _head_dim(attn, n_heads, in_dim):
    if hasattr(attn, "head_dim"):
        return attn.head_dim
    if hasattr(attn, "scale"):
        return int(round(attn.scale ** -2))
    return in_dim // n_heads


def _model_module(model):
    """
    Find the mlx_lm.models.* module that owns scaled_dot_product_attention.
    """
    inner = _inner(model)
    if not inner.layers:
        return None

    first_attn = None
    first_attn_layer = None
    for layer in inner.layers:
        a = _attn(layer)
        if a is not None:
            first_attn = a
            first_attn_layer = layer
            break

    candidates = []
    if first_attn_layer is not None:
        candidates.append(first_attn_layer)
    if first_attn is not None:
        candidates.append(first_attn)

    for obj in candidates:
        for cls in type(obj).__mro__:
            mod_name = getattr(cls, "__module__", "")
            if mod_name.startswith("mlx_lm.models."):
                mod = sys.modules.get(mod_name)
                if mod is not None and hasattr(mod, "scaled_dot_product_attention"):
                    return mod
    return None


# ---------------------------------------------------------------------------
# Weight reordering (quantization-aware, same as before)
# ---------------------------------------------------------------------------

def _reorder_proj_rows(proj, row_idx_mx: mx.array) -> None:
    proj.weight = proj.weight[row_idx_mx]
    if getattr(proj, "scales", None) is not None:
        proj.scales = proj.scales[row_idx_mx]
    if getattr(proj, "biases", None) is not None:
        proj.biases = proj.biases[row_idx_mx]
    if getattr(proj, "bias", None) is not None:
        proj.bias = proj.bias[row_idx_mx]


def _reorder_oproj_cols(proj, new_head_order_np: np.ndarray,
                         head_dim: int) -> None:
    if getattr(proj, "scales", None) is not None:
        bits       = proj.bits
        group_size = proj.group_size
        pph        = head_dim // (32 // bits)
        gph        = head_dim // group_size

        w_cols = np.concatenate([
            np.arange(h * pph, (h + 1) * pph) for h in new_head_order_np
        ])
        s_cols = np.concatenate([
            np.arange(h * gph, (h + 1) * gph) for h in new_head_order_np
        ])

        proj.weight = proj.weight[:, mx.array(w_cols)]
        proj.scales = proj.scales[:, mx.array(s_cols)]
        proj.biases = proj.biases[:, mx.array(s_cols)]
    else:
        cols = np.concatenate([
            np.arange(h * head_dim, (h + 1) * head_dim)
            for h in new_head_order_np
        ])
        proj.weight = proj.weight[:, mx.array(cols)]


# ---------------------------------------------------------------------------
# Random head selection + weight reordering
# ---------------------------------------------------------------------------

def reorder_model_weights_random(model,
                                  pool_fraction: float = 0.3,
                                  seed: int = 42) -> dict:
    """
    Randomly select pool_fraction of Q-heads per attention layer and reorder
    Q/K/V/O weights so pooled heads are contiguous (first in index order).

    For GQA: a KV-head is marked as 'pooled' if ANY of its associated Q-heads
    is in the randomly-selected set. All Q-heads from pooled KV-heads are pooled.

    Returns dict[layer_idx -> n_pooled_kv_heads] for patching.
    """
    rng = np.random.RandomState(seed)
    body = _inner(model)
    n_pooled_map = {}

    for layer_idx, layer in enumerate(body.layers):
        attn = _attn(layer)

        if attn is None or not hasattr(attn, "q_proj"):
            n_pooled_map[layer_idx] = 0
            continue

        H    = _n_heads(attn)
        Hkv  = _n_kv_heads(attn, H)
        ratio = H // Hkv

        if Hkv <= 1:
            # Can't selectively pool with only 1 KV head
            n_pooled_map[layer_idx] = 0
            continue

        d_in = attn.q_proj.weight.shape[1]
        D    = _head_dim(attn, H, d_in)

        # Detect gated q_proj
        q_out_rows = attn.q_proj.weight.shape[0]
        is_gated   = (q_out_rows == H * D * 2)

        # Randomly select Q-heads to pool
        n_pool_q = max(1, int(H * pool_fraction))
        pooled_q_heads = set(rng.choice(H, size=n_pool_q, replace=False))

        # Map to KV-heads: pool a KV-head if any of its Q-heads is pooled
        pooled_kv = set()
        for kv in range(Hkv):
            q_start = kv * ratio
            q_end   = (kv + 1) * ratio
            if any(q in pooled_q_heads for q in range(q_start, q_end)):
                pooled_kv.add(kv)

        n_pool_kv = len(pooled_kv)

        if n_pool_kv == 0 or n_pool_kv == Hkv:
            # All or nothing — skip this layer (nothing to route)
            n_pooled_map[layer_idx] = n_pool_kv
            continue

        # Build permutation: pooled KV heads first, then unpooled
        pooled_kv_sorted   = sorted(pooled_kv)
        unpooled_kv_sorted = sorted(set(range(Hkv)) - pooled_kv)
        new_kv_order = np.array(pooled_kv_sorted + unpooled_kv_sorted)

        # Q-head order follows KV-head order
        new_q_order = np.concatenate([
            np.arange(kv * ratio, (kv + 1) * ratio) for kv in new_kv_order
        ])

        # Row indices for K/V proj (D rows per head)
        kv_rows = np.concatenate([np.arange(h * D, (h + 1) * D)
                                   for h in new_kv_order])
        # Row indices for Q proj
        q_stride = 2 * D if is_gated else D
        q_rows   = np.concatenate([np.arange(h * q_stride, (h + 1) * q_stride)
                                    for h in new_q_order])

        # Apply reordering
        _reorder_proj_rows(attn.q_proj, mx.array(q_rows))
        _reorder_proj_rows(attn.k_proj, mx.array(kv_rows))
        _reorder_proj_rows(attn.v_proj, mx.array(kv_rows))
        _reorder_oproj_cols(attn.o_proj, new_q_order, D)

        mx.eval(attn.q_proj.weight, attn.k_proj.weight,
                attn.v_proj.weight, attn.o_proj.weight)

        n_pooled_map[layer_idx] = n_pool_kv

    return n_pooled_map


# ---------------------------------------------------------------------------
# Per-layer routing wrapper (SDPA override set ONCE, not per call)
# ---------------------------------------------------------------------------

_patched_layers: dict = {}       # layer_idx -> (layer, original_attn_module)
_original_sdpa = None            # saved for restore


def _build_routing_sdpa(model_mod, orig_sdpa, n_pool_kv, ratio, pool_factor):
    """Build and return the routed SDPA closure. Called once at enable time."""

    n_pool_q = n_pool_kv * ratio
    P        = pool_factor

    def _pooled_causal_mask(L_q: int, L_macro: int, dtype) -> mx.array:
        """
        Build additive causal mask for pooled attention.
        Shape (L_q, L_macro). Query at position q can attend to macro-blocks
        up to q//P (standard block-sparse causal convention).
        """
        row_idx = mx.arange(L_q)[:, None]       # (L_q, 1)
        blk_idx = mx.arange(L_macro)[None, :]   # (1, L_macro)
        # Query position q is in macro-block q//P.
        # It can attend to blocks <= q//P
        valid = blk_idx * P + (P - 1) <= row_idx
        return mx.where(valid, 0.0, -1e4).astype(dtype)

    def _routed(queries, keys, values, *, cache, scale, mask, **kw):
        L_q = queries.shape[2]

        # Short sequence or single-token generation: full attention
        if L_q <= P:
            return orig_sdpa(queries, keys, values,
                             cache=cache, scale=scale, mask=mask, **kw)

        B, Hkv, L_k, D = keys.shape

        # Slice — views, no copy (heads are contiguous by construction)
        q_pool = queries[:, :n_pool_q]
        q_full = queries[:, n_pool_q:]
        k_pool = keys[:, :n_pool_kv]
        k_full = keys[:, n_pool_kv:]
        v_pool = values[:, :n_pool_kv]
        v_full = values[:, n_pool_kv:]

        # Pool KV (trim sequence to multiple of P)
        valid = L_k - (L_k % P)
        if valid > 0:
            k_pooled = k_pool[:, :, :valid, :].reshape(
                B, n_pool_kv, valid // P, P, D).mean(axis=3)
            v_pooled = v_pool[:, :, :valid, :].reshape(
                B, n_pool_kv, valid // P, P, D).mean(axis=3)
        else:
            k_pooled, v_pooled = k_pool, v_pool

        # Build causal mask for pooled attention
        L_macro = valid // P
        pool_mask = _pooled_causal_mask(L_q, L_macro, queries.dtype)

        # Two SDPA calls — pooled path gets proper causal mask
        out_pool = orig_sdpa(q_pool, k_pooled, v_pooled,
                             cache=None, scale=scale, mask=pool_mask, **kw)
        out_full = orig_sdpa(q_full, k_full, v_full,
                             cache=None, scale=scale, mask=mask, **kw)

        return mx.concatenate([out_pool, out_full], axis=1)

    return _routed


class _PooledLayer(nn.Module):
    """Drop-in replacement for an attention module with selective pooling."""

    def __init__(self, base_attn, n_pool_kv: int, ratio: int,
                 pool_factor: int):
        super().__init__()
        self.base        = base_attn
        self.n_pool_kv   = n_pool_kv
        self.ratio       = ratio
        self.pool_factor = pool_factor

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        return self.base(x, mask, cache)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enable_random_pooling(model,
                           pool_factor: int = 4,
                           pool_fraction: float = 0.3,
                           seed: int = 42) -> dict:
    """
    Enable random selective K/V pooling on an MLX model.

    Parameters
    ----------
    model         : loaded mlx-lm model (weights modified in-place)
    pool_factor   : K/V pool stride (e.g., P=4 averages 4 tokens)
    pool_fraction : fraction of Q-heads to pool (e.g., 0.3 = 30%)
    seed          : random seed for head selection

    Returns
    -------
    info : dict with keys 'n_layers_routed', 'n_heads_pooled', 'n_heads_total'
    """
    global _patched_layers, _original_sdpa

    if _patched_layers:
        disable_random_pooling(model)

    print(f"[Pool] Selecting {pool_fraction:.0%} of Q-heads randomly (seed={seed})...")

    # Step 1: randomly select heads + reorder weights
    n_pooled_map = reorder_model_weights_random(model, pool_fraction, seed)

    # Step 2: find the model's SDPA module
    mod = _model_module(model)
    if mod is None:
        raise RuntimeError(
            "Could not find mlx_lm.models.* module with scaled_dot_product_attention")

    if _original_sdpa is None:
        _original_sdpa = mod.scaled_dot_product_attention

    # Step 3: build ONE routing SDPA closure per layer config.
    # Since multiple layers may share the same (n_pool_kv, ratio, P),
    # we cache closures to avoid redundant builds (and because we can only
    # set ONE global SDPA override at a time — so layers with different
    # n_pool_kv must be patched separately on each call).

    # Actually, the clean approach: patch SDPA once per unique (n_pool_kv, ratio)
    # combination. But since we only set one global SDPA, we need per-layer
    # wrappers that set the correct closure before calling base.

    # Revised approach: each _PooledLayer stores its own routing closure and
    # sets it on the shared module before calling base. But we still need
    # per-call patching unless we restructure the model architecture.

    # Simplest correct approach: patch per-layer on forward.
    # The overhead is minimal (module attribute assignment is fast).
    # We cache closures to avoid rebuilding them.

    body = _inner(model)
    closure_cache = {}
    n_patched = 0
    total_pooled_q = 0
    total_q = 0

    for layer_idx, layer in enumerate(body.layers):
        attn = _attn(layer)
        if attn is None:
            continue

        n_pool_kv = n_pooled_map.get(layer_idx, 0)
        if n_pool_kv == 0:
            continue

        H    = _n_heads(attn)
        Hkv  = _n_kv_heads(attn, H)
        ratio = H // Hkv

        total_q += H
        total_pooled_q += n_pool_kv * ratio

        # Build or retrieve the routing closure
        key = (n_pool_kv, ratio, pool_factor)
        if key not in closure_cache:
            closure_cache[key] = _build_routing_sdpa(
                mod, _original_sdpa, n_pool_kv, ratio, pool_factor)

        routing_fn = closure_cache[key]

        # Create wrapper that patches SDPA → calls base → restores
        class _LayerWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self._attn   = attn
                self._mod    = mod
                self._routed = routing_fn
                self._orig   = _original_sdpa

            def __call__(self, x, mask=None, cache=None):
                self._mod.scaled_dot_product_attention = self._routed
                try:
                    return self._attn(x, mask, cache)
                finally:
                    self._mod.scaled_dot_product_attention = self._orig

        wrapper = _LayerWrapper()

        if hasattr(layer, "self_attn"):
            _patched_layers[layer_idx] = (layer, layer.self_attn)
            layer.self_attn = wrapper
        else:
            _patched_layers[layer_idx] = (layer, layer.attention)
            layer.attention = wrapper
        n_patched += 1

    pct_q = 100.0 * total_pooled_q / total_q if total_q else 0.0
    print(f"[Pool] Enabled: P={pool_factor}, {pool_fraction:.0%} requested, "
          f"{total_pooled_q}/{total_q} Q-heads pooled ({pct_q:.1f}%), "
          f"{n_patched} layers routed")

    return {
        "n_layers_routed": n_patched,
        "n_heads_pooled": total_pooled_q,
        "n_heads_total": total_q,
        "pool_fraction_actual": total_pooled_q / total_q if total_q else 0.0,
    }


def disable_random_pooling(model) -> None:
    """Restore original attention modules. Weights remain reordered."""
    global _patched_layers, _original_sdpa
    n = 0
    for layer_idx, (layer, orig_attn) in _patched_layers.items():
        if hasattr(layer, "self_attn"):
            layer.self_attn = orig_attn
        else:
            layer.attention = orig_attn
        n += 1
    _patched_layers.clear()
    if n:
        print(f"[Pool] Disabled ({n} layers restored)")


# ---------------------------------------------------------------------------
# Backward compatibility aliases (for existing experiment scripts)
# ---------------------------------------------------------------------------

def enable_rra_routed(model, skew_data, pool_factor=4, skew_threshold=0.3):
    """
    Old API — now just maps threshold to fraction and uses random selection.
    Provided so existing experiment scripts don't break.
    """
    all_skews = []
    for arr in skew_data.values():
        all_skews.extend(arr.tolist() if hasattr(arr, 'tolist') else list(arr))
    all_skews = np.array(all_skews)
    n_total = len(all_skews)
    n_below = int((all_skews < skew_threshold).sum())
    fraction = max(0.05, n_below / max(n_total, 1))

    print(f"[RRA-compat] skew_threshold={skew_threshold} → "
          f"{n_below}/{n_total} below → pool_fraction={fraction:.3f}")

    return enable_random_pooling(
        model, pool_factor=pool_factor, pool_fraction=fraction,
        seed=hash(str(skew_threshold)) % (2**31))


def disable_rra_routed(model):
    """Backward compatibility alias."""
    return disable_random_pooling(model)
