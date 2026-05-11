"""
Exp01: Weight Reordering Identity Check

Null hypothesis: Weight reordering changes model outputs on at least one
                 registered model.
Success:          ALL registered models satisfy |ΔPPL| < 0.05 after weight
                  reordering (identity preserved universally).
Failure:          Any model produces |ΔPPL| >= 0.05 (reordering is not a no-op).

Model:     All registered models (experiments/models.py)
Data:      WikiText-2 test set, 50 samples, seq_len=512
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

from experiments.harness import Experiment
from experiments.models import models_for_mode


class Exp01IdentityCheck(Experiment):
    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-set", default="all",
                            help="'all' or a family name (e.g. 'qwen3.5')")
        args = parser.parse_args()

        from mlx_lm import load
        from rra.routed_patch import reorder_model_weights_random, _inner, _attn, _n_heads

        eval_text = self._load_wikitext(max_samples=50)

        models = models_for_mode(args.model_set)
        all_results = {}
        failures = []

        for m in models:
            short = m.id.split("/")[-1]

            try:
                model, tokenizer = load(m.id)
            except Exception as e:
                print(f"  [{short}] LOAD FAILED: {e}")
                all_results[short] = {"error": f"load failed: {e}"}
                continue

            # Count Q-heads
            body = _inner(model)
            total_heads = 0
            for layer in body.layers:
                attn = _attn(layer)
                if attn is not None:
                    H = _n_heads(attn)
                    if H is not None:
                        total_heads += H

            # Baseline
            try:
                baseline = float(self._perplexity(model, tokenizer, eval_text, seq_len=512))
            except Exception as e:
                print(f"  [{short}] BASELINE FAILED: {e}")
                all_results[short] = {"error": f"baseline failed: {e}"}
                del model
                continue

            # Reorder weights (seed=0, fraction=0.5)
            n_pooled_map = reorder_model_weights_random(model, pool_fraction=0.5, seed=0)
            n_reordered = sum(1 for v in n_pooled_map.values() if v > 0)

            # Measure post-reorder PPL with pooling DISABLED (should equal baseline)
            try:
                post_reorder = float(self._perplexity(model, tokenizer, eval_text, seq_len=512))
            except Exception as e:
                print(f"  [{short}] POST-REORDER FAILED: {e}")
                all_results[short] = {"error": f"post-reorder failed: {e}"}
                del model
                continue

            delta = post_reorder - baseline
            delta_pct = 100.0 * delta / max(baseline, 0.01)
            passed = abs(delta) < 0.05

            print(f"  [{short}] Baseline={baseline:.3f}  Post-reorder={post_reorder:.3f}"
                  f"  ΔPPL={delta:+.4f} ({delta_pct:+.2f}%)  "
                  f"layers reordered={n_reordered}  "
                  f"{'OK' if passed else 'FAIL'}")

            all_results[short] = {
                "model_id": m.id,
                "family": m.family,
                "size": m.size,
                "baseline_ppl": baseline,
                "post_reorder_ppl": post_reorder,
                "delta_ppl": delta,
                "delta_ppl_pct": delta_pct,
                "n_heads_total": total_heads,
                "n_layers_reordered": n_reordered,
                "identity_ok": passed,
            }
            if not passed:
                failures.append(short)
            del model
            self.checkpoint({"seq_len": 512, "models": all_results})

        # ── Hypothesis ──
        tested = {k: v for k, v in all_results.items() if "identity_ok" in v}
        n_pass = sum(1 for v in tested.values() if v["identity_ok"])
        n_total = len(tested)
        all_pass = len(failures) == 0

        self.expect(all_pass,
            f"All {n_total} models pass identity check (|ΔPPL| < 0.05)"
            f"{f'; failures: {failures}' if failures else ''}")

        # ── Save ──
        self.save_results({
            "seq_len": 512,
            "delta_tolerance": 0.05,
            "n_models_tested": n_total,
            "n_models_passed": n_pass,
            "failures": failures,
            "models": all_results,
        })

        # ── Figure ──
        valid = {k: v for k, v in all_results.items() if "delta_ppl" in v}
        sorted_items = sorted(valid.items(), key=lambda kv: abs(kv[1]["delta_ppl"]))
        labels = [f"{v['size']}\n{v['family']}" for _, v in sorted_items]
        deltas = [v["delta_ppl"] for _, v in sorted_items]
        colors = ["green" if abs(d) < 0.05 else "red" for d in deltas]

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4))
        x = np.arange(len(labels))
        ax.bar(x, deltas, color=colors, width=0.6)
        ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="|ΔPPL|=0.05 threshold")
        ax.axhline(y=-0.05, color="red", linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("ΔPPL (post-reorder − baseline)")
        ax.set_title(f"Weight Reordering Identity Check — All Models\n"
                     f"{n_pass}/{n_total} pass |ΔPPL| < 0.05")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        self.save_figure(fig, "identity_check")
        plt.close()

        self.conclude(
            f"Weight reordering identity: {n_pass}/{n_total} models pass |ΔPPL| < 0.05. "
            f"{'All models confirmed identity-preserving.' if all_pass else f'FAILURES: {failures}.'}"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _load_wikitext(max_samples: int = 50) -> str:
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
    Exp01IdentityCheck().main()
