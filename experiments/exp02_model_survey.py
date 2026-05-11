"""
Exp02: Model Survey — Head Topology and Pooling Compatibility

Null hypothesis: RRA selective pooling fails or causes >10% PPL degradation on
                 the majority of registered model architectures.
Success:          >=80% of registered models tolerate P=4/fraction=0.25 with
                  |ΔPPL| < 10%; at least two distinct non-Qwen families are
                  compatible.
Failure:          Fewer than 80% of models are compatible, or all non-Qwen
                  models degrade by >10%.

Methodology: Each model's pooled PPL is averaged over --seeds random seeds
             (fresh load per seed) to reduce head-selection variance.

Model:     All registered models (experiments/models.py)
Data:      WikiText-2 test set, 60 samples, seq_len=512
Hardware:  Apple M4 Max, 64GB
Date:      2026-05-11
"""

import argparse
import math
import numpy as np
import mlx.core as mx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from experiments.harness import Experiment
from experiments.models import models_for_mode

SEEDS = [42, 1337, 9999]


def _probe_heads(model):
    """Return (H, Hkv, ratio) from the first patchable attention layer, or Nones."""
    from rra.routed_patch import _inner, _attn, _n_heads, _n_kv_heads
    body = _inner(model)
    for layer in body.layers:
        attn = _attn(layer)
        if attn is None or not hasattr(attn, "q_proj"):
            continue
        H = _n_heads(attn)
        if H is None:
            continue
        Hkv = _n_kv_heads(attn, H)
        if Hkv and Hkv > 0:
            return H, Hkv, H // Hkv
    return None, None, None


class Exp02ModelSurvey(Experiment):
    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--pool-factor", type=int, default=4)
        parser.add_argument("--pool-fraction", type=float, default=0.25)
        parser.add_argument("--seq-len", type=int, default=512)
        parser.add_argument("--samples", type=int, default=60)
        parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
        parser.add_argument("--model-set", default="all",
                            help="'all' or a family name (e.g. 'qwen3.5')")
        args = parser.parse_args()

        P = args.pool_factor
        fraction = args.pool_fraction
        seeds = args.seeds

        from mlx_lm import load
        from rra.routed_patch import enable_random_pooling

        eval_text = self._load_wikitext(max_samples=args.samples)

        models = models_for_mode(args.model_set)
        all_results = {}

        for m in models:
            short = m.id.split("/")[-1]
            result = {
                "family": m.family, "size": m.size, "attn": m.attn,
                "model_id": m.id,
            }

            try:
                model, tokenizer = load(m.id)
            except Exception as e:
                print(f"  LOAD FAILED: {e}")
                result["error"] = f"load failed: {e}"
                all_results[short] = result
                continue

            # Probe head topology
            H, Hkv, ratio = _probe_heads(model)
            result["n_q_heads"] = H
            result["n_kv_heads"] = Hkv
            result["gqa_ratio"] = ratio
            print(f"  [{short}] H={H}, Hkv={Hkv}, ratio={ratio}")

            # Baseline PPL (clean load, no pooling)
            try:
                baseline = float(self._perplexity(
                    model, tokenizer, eval_text, args.seq_len))
                result["baseline_ppl"] = baseline
                print(f"  [{short}] Baseline PPL: {baseline:.3f}")
            except Exception as e:
                print(f"  BASELINE FAILED: {e}")
                result["error"] = f"baseline failed: {e}"
                del model
                all_results[short] = result
                continue
            del model

            # Multi-seed pooled eval — fresh load per seed to avoid weight leakage
            seed_ppls, seed_infos = [], []
            for seed in seeds:
                try:
                    model, tokenizer = load(m.id)
                except Exception as e:
                    print(f"  [{short}] seed={seed} LOAD FAILED: {e}")
                    break
                try:
                    info = enable_random_pooling(model, pool_factor=P,
                                                 pool_fraction=fraction, seed=seed)
                    ppl = float(self._perplexity(model, tokenizer, eval_text, args.seq_len))
                    seed_ppls.append(ppl)
                    seed_infos.append(info)
                except Exception as e:
                    print(f"  [{short}] seed={seed} POOL FAILED: {e}")
                finally:
                    del model

            if seed_ppls:
                mean_pooled = float(np.mean(seed_ppls))
                std_pooled  = float(np.std(seed_ppls)) if len(seed_ppls) > 1 else 0.0
                delta_pct   = 100.0 * (mean_pooled - baseline) / max(baseline, 0.01)
                delta_std   = 100.0 * std_pooled / max(baseline, 0.01)
                last        = seed_infos[-1]
                result.update({
                    "pooled_ppl_mean":      mean_pooled,
                    "pooled_ppl_std":       std_pooled,
                    "delta_pct":            delta_pct,
                    "delta_pct_std":        delta_std,
                    "n_seeds":              len(seed_ppls),
                    "n_q_heads_pooled":     last["n_heads_pooled"],
                    "n_q_heads_total":      last["n_heads_total"],
                    "pool_fraction_actual": last["pool_fraction_actual"],
                    "n_layers_routed":      last["n_layers_routed"],
                })
                print(f"  [{short}] Pooled PPL: {mean_pooled:.3f}±{std_pooled:.3f} "
                      f"(Δ={delta_pct:+.1f}%±{delta_std:.1f}%, "
                      f"{last['n_heads_pooled']}/{last['n_heads_total']} Q-heads, "
                      f"{last['pool_fraction_actual']:.1%} actual, {len(seed_ppls)} seeds)")
            else:
                result["error"] = "pool failed: all seeds failed"

            all_results[short] = result
            self.checkpoint({"pool_factor": P, "pool_fraction_requested": fraction,
                             "seq_len": args.seq_len, "models": all_results})

        # ── Hypothesis checks ──
        valid = {k: v for k, v in all_results.items() if "delta_pct" in v}
        total = len(all_results)
        n_ok = sum(1 for v in valid.values() if abs(v["delta_pct"]) < 10.0)
        pct_ok = 100.0 * n_ok / max(total, 1)

        print(f"\n  Survey: {n_ok}/{total} models within |ΔPPL| < 10% ({pct_ok:.0f}%)")
        for short, res in sorted(valid.items(), key=lambda kv: kv[1]["delta_pct"]):
            print(f"    {res['family']:10s} {res['size']:6s}  "
                  f"PPL {res['baseline_ppl']:.2f} → {res['pooled_ppl_mean']:.2f}±{res.get('pooled_ppl_std',0):.3f}  "
                  f"Δ={res['delta_pct']:+.1f}%  "
                  f"H={res['n_q_heads']}/Hkv={res['n_kv_heads']}")

        non_qwen_families = set(
            v["family"] for v in valid.values()
            if not v["family"].startswith("qwen")
            and abs(v["delta_pct"]) < 10.0
        )
        n_non_qwen_families = len(non_qwen_families)

        self.expect(pct_ok >= 80.0,
            f"{n_ok}/{total} models within |ΔPPL| < 10% ({pct_ok:.0f}% >= 80%)")
        self.expect(n_non_qwen_families >= 2,
            f"{n_non_qwen_families} non-Qwen families within |ΔPPL| < 10% (>= 2): "
            f"{non_qwen_families}")

        # ── Save ──
        self.save_results({
            "pool_factor": P,
            "pool_fraction_requested": fraction,
            "seq_len": args.seq_len,
            "n_models_tested": total,
            "n_models_ok": n_ok,
            "pct_ok": pct_ok,
            "models": all_results,
        })

        # ── Figure ──
        family_colors = {
            "qwen2.5": "#4472C4",
            "qwen3.5": "#2E75B6",
            "gemma4":  "#70AD47",
            "llama3":  "#ED7D31",
            "mistral": "#A9D18E",
        }
        default_color = "#888888"

        sorted_valid = sorted(valid.items(), key=lambda kv: kv[1]["delta_pct"])
        labels = [f"{v['size']}\n{v['family']}" for _, v in sorted_valid]
        deltas = [v["delta_pct"] for _, v in sorted_valid]
        colors = [family_colors.get(v["family"], default_color)
                  for _, v in sorted_valid]

        errors = [v.get("delta_pct_std", 0.0) for _, v in sorted_valid]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
        x = np.arange(len(labels))
        ax.bar(x, deltas, color=colors, width=0.6,
               yerr=errors, capsize=3, error_kw={"elinewidth": 1.0, "alpha": 0.7})
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.axhline(y=10,  color="red",   linestyle="--", alpha=0.4, label="10% threshold")
        ax.axhline(y=-10, color="green", linestyle="--", alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("ΔPPL (%)")
        ax.set_title(f"Model Survey: Pooling Effect (P={P}, fraction={fraction:.0%}, L={args.seq_len}, "
                     f"{len(seeds)} seeds/model)\n{n_ok}/{total} models within |ΔPPL|<10%")

        patches = [mpatches.Patch(color=c, label=f)
                   for f, c in family_colors.items()
                   if any(v.get("family") == f for v in valid.values())]
        patches += [
            plt.Line2D([0], [0], color="red", linestyle="--", label="50% threshold"),
            plt.Line2D([0], [0], color="green", linestyle="--", label="-10% improvement"),
        ]
        ax.legend(handles=patches, fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        self.save_figure(fig, "model_survey")
        plt.close()

        hkv_values = [v["n_kv_heads"] for v in valid.values() if v.get("n_kv_heads")]
        self.conclude(
            f"Model survey ({total} models, {len(seeds)} seeds/model): "
            f"{n_ok}/{total} models within |ΔPPL| < 10% at P={P}/fraction={fraction:.0%} "
            f"({pct_ok:.0f}%). {n_non_qwen_families} non-Qwen families compatible: {non_qwen_families}. "
            f"Hkv ranges {min(hkv_values, default='?')}–{max(hkv_values, default='?')} across registry."
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _load_wikitext(max_samples: int = 60) -> str:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = []
        for item in ds:
            t = item["text"].strip()
            if t and not t.startswith("= ") and len(t) > 20:
                texts.append(t)
            if max_samples and len(texts) >= max_samples:
                break
        return " ".join(texts)

    @staticmethod
    def _perplexity(model, tokenizer, text: str, seq_len: int) -> float:
        tokens = tokenizer.encode(text)
        n_chunks = max(1, len(tokens) // seq_len)
        sl = min(seq_len, len(tokens))
        tokens = tokens[:n_chunks * sl]
        total_nll = 0.0
        total_tokens = 0
        for ci in range(n_chunks):
            chunk = tokens[ci * sl:(ci + 1) * sl]
            inp = mx.array([chunk[:-1]])
            tgt = mx.array([chunk[1:]])
            logits = model(inp)
            mx.eval(logits)
            logits_np = np.array(logits.astype(mx.float32))
            tgt_np = np.array(tgt.astype(mx.int32))
            B, T, V = logits_np.shape
            for t in range(T):
                l = logits_np[0, t, :]
                target = int(tgt_np[0, t])
                l = l - np.max(l)
                log_soft = l - np.log(np.sum(np.exp(l)))
                total_nll += float(-log_soft[target])
                total_tokens += 1
        return float(math.exp(total_nll / max(total_tokens, 1)))


if __name__ == "__main__":
    Exp02ModelSurvey().main()
