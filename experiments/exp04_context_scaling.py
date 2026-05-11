"""
Exp04: Context-Length Scaling — Does Pooling Tolerance Increase With seq_len?

Null hypothesis: ΔPPL from selective pooling is independent of sequence length.
Success:          For large models (>=9B), ΔPPL at fixed pool settings
                  decreases (improves) as seq_len increases from 512 to 16384.
Failure:          No model shows decreasing ΔPPL with seq_len.

Motivation: RRA's value proposition scales with context length — at longer
            sequences, the pooled heads average over more tokens, acting as
            stronger regularization. This should manifest as decreasing
            (or even negative) ΔPPL as seq_len grows.

Model:     All registered models (experiments/models.py)
Data:      WikiText-2 test set, seq_len-scaled samples, pool_factor=4, fraction=0.25
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


def _spearmanr(x, y):
    """Spearman rank correlation (no scipy dependency)."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d2 = float(np.sum((rx - ry) ** 2))
    return 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384]

# Paragraphs to load per seq_len so we have at least 4 evaluation chunks.
# WikiText-2 test paragraphs avg ~150 tokens; 4 chunks × seq_len / 150 tokens/para.
SAMPLES_BY_SEQLEN = {
    512:   60,
    1024:  100,
    2048:  200,
    4096:  350,
    8192:  600,
    16384: 1000,
}


class Exp04ContextScaling(Experiment):
    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--pool-factor", type=int, default=4)
        parser.add_argument("--pool-fraction", type=float, default=0.25)
        parser.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS)
        parser.add_argument("--model-set", default="all",
                            help="'all' or a family name (e.g. 'qwen3.5')")
        args = parser.parse_args()

        P = args.pool_factor
        fraction = args.pool_fraction
        seq_lens = sorted(args.seq_lens)

        from mlx_lm import load
        from rra.routed_patch import enable_random_pooling, disable_random_pooling

        # Pre-load text for each required sample count
        texts: dict[int, str] = {}
        for sl in seq_lens:
            n = SAMPLES_BY_SEQLEN.get(sl, 1000)
            if n not in texts:
                texts[n] = self._load_wikitext(max_samples=n)
                print(f"  Loaded {len(texts[n]):,} chars (target samples={n})")

        all_results = {}

        for m in models_for_mode(args.model_set):
            model_id = m.id
            short = model_id.split("/")[-1]
            print(f"\n{'='*60}\n  {short}")

            model_result = {
                "family": m.family, "size": m.size,
                "model_id": model_id, "seq_lens": {},
            }

            for sl in seq_lens:
                text = texts[SAMPLES_BY_SEQLEN.get(sl, 1000)]

                # Fresh load per seq_len — weights must be clean for baseline
                try:
                    model, tokenizer = load(model_id)
                except Exception as e:
                    print(f"  [{short}] L={sl:>6} LOAD FAILED: {e}")
                    continue

                n_tokens = len(tokenizer.encode(text))
                if n_tokens < sl * 2:
                    print(f"  [{short}] L={sl:>6} only {n_tokens} tokens "
                          f"(need {sl*2}) — skipping")
                    del model
                    continue

                try:
                    baseline = float(self._perplexity(model, tokenizer, text, sl))
                    print(f"  [{short}] L={sl:>6} baseline={baseline:.4f}")
                except Exception as e:
                    print(f"  [{short}] L={sl:>6} BASELINE FAILED: {e}")
                    del model
                    continue

                try:
                    info = enable_random_pooling(
                        model, pool_factor=P, pool_fraction=fraction, seed=42)
                    pooled_ppl = float(self._perplexity(model, tokenizer, text, sl))
                    disable_random_pooling(model)
                    delta_pct = 100.0 * (pooled_ppl - baseline) / max(baseline, 0.01)
                    model_result["seq_lens"][str(sl)] = {
                        "seq_len": sl,
                        "baseline_ppl": baseline,
                        "pooled_ppl": pooled_ppl,
                        "delta_pct": delta_pct,
                        "n_q_pooled": info["n_heads_pooled"],
                        "pool_fraction_actual": info["pool_fraction_actual"],
                    }
                    print(f"  [{short}] L={sl:>6} pooled={pooled_ppl:.4f}  "
                          f"Δ={delta_pct:+.2f}%")
                except Exception as e:
                    print(f"  [{short}] L={sl:>6} POOL FAILED: {e}")

                del model

            all_results[short] = model_result
            self.checkpoint({"pool_factor": P, "pool_fraction": fraction,
                             "seq_lens": seq_lens, "models": all_results})

        # ── Hypothesis: large models (>=9B) show Spearman r < -0.5 between seq_len and ΔPPL ──
        # Require ≥50% of large models to show a moderate negative trend (not just endpoints).
        n_large_negative = 0
        n_large_total = 0
        for short, res in all_results.items():
            try:
                sz = float(res["size"].upper().replace("B", "").split("A")[0])
            except ValueError:
                sz = 0
            sl_data = sorted(
                [(int(k), v["delta_pct"]) for k, v in res["seq_lens"].items()])
            if len(sl_data) < 3:
                continue
            xs = [s for s, _ in sl_data]
            deltas = [d for _, d in sl_data]
            r = _spearmanr(xs, deltas)
            tag = "large" if sz >= 9 else "small"
            if sz >= 9:
                n_large_total += 1
                if r < -0.5:
                    n_large_negative += 1
            print(f"  {short} [{tag}]: "
                  f"{' → '.join(f'{d:+.1f}%' for _, d in sl_data)}"
                  f"  r={r:+.2f}{'  ↓' if r < -0.5 else ''}")

        pct_negative = n_large_negative / max(n_large_total, 1)
        large_improves = n_large_total == 0 or pct_negative >= 0.5
        self.expect(large_improves,
            f"≥50% of large models (>=9B) show Spearman r < −0.5 between seq_len and ΔPPL "
            f"({n_large_negative}/{n_large_total} = {100 * pct_negative:.0f}%)")

        # ── Save ──
        self.save_results({
            "pool_factor": P,
            "pool_fraction": fraction,
            "seq_lens": seq_lens,
            "models": all_results,
        })

        # ── Figures ──
        family_colors = {
            "qwen2.5": "#1f77b4", "qwen3.5": "#aec7e8",
            "gemma4":  "#2ca02c", "llama3":  "#d62728",
            "mistral": "#ff7f0e",
        }

        # Figure 1: ΔPPL vs seq_len (log x-axis), all models
        fig, ax = plt.subplots(figsize=(9, 5))
        for short, res in all_results.items():
            sl_data = sorted(
                [(int(k), v["delta_pct"]) for k, v in res["seq_lens"].items()])
            if len(sl_data) < 2:
                continue
            xs = [s for s, _ in sl_data]
            ys = [d for _, d in sl_data]
            col = family_colors.get(res.get("family", ""), "#888888")
            try:
                sz = float(res["size"].upper().replace("B", "").split("A")[0])
                lw = 2.2 if sz >= 9 else 1.2
                ls = "-" if sz >= 9 else "--"
            except ValueError:
                lw, ls = 1.5, "-"
            ax.plot(xs, ys, f"o{ls}", color=col,
                    label=f"{res['size']} {res['family']}",
                    markersize=5, linewidth=lw)

        ax.axhline(y=0, color="#444", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(seq_lens)
        ax.set_xticklabels([f"{sl:,}" for sl in seq_lens], fontsize=8)
        ax.set_xlabel("Sequence length (tokens, log₂ scale)", fontsize=9)
        ax.set_ylabel("ΔPPL (%)", fontsize=9)
        ax.set_title(
            f"RRA: Pooling Tolerance vs Context Length — All Models\n"
            f"(P={P}, fraction={fraction:.0%}, solid=large ≥9B, dashed=small)",
            fontsize=10)
        ax.legend(fontsize=7, ncol=2, framealpha=0.7)
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        self.save_figure(fig, "context_scaling_delta")
        plt.close()

        # Figure 2: Heatmap — model × seq_len, cell = ΔPPL
        model_labels = [f"{res['size']} {res['family']}"
                        for res in all_results.values()]
        matrix = np.full((len(all_results), len(seq_lens)), np.nan)
        for i, res in enumerate(all_results.values()):
            for j, sl in enumerate(seq_lens):
                entry = res["seq_lens"].get(str(sl))
                if entry:
                    matrix[i, j] = entry["delta_pct"]

        fig, ax = plt.subplots(
            figsize=(max(6, len(seq_lens) * 1.4), max(4, len(model_labels) * 0.45)))
        vmax = min(np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 50, 50)
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(seq_lens)))
        ax.set_xticklabels([f"{sl:,}" for sl in seq_lens], fontsize=8)
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels, fontsize=8)
        ax.set_xlabel("Sequence length", fontsize=9)
        ax.set_title(
            f"ΔPPL Heatmap: All Models × Context Length  "
            f"(P={P}, f={fraction:.0%})\n"
            f"Green = improvement  ·  Red = degradation",
            fontsize=10)
        plt.colorbar(im, ax=ax, label="ΔPPL (%)", shrink=0.8)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:+.0f}%",
                            ha="center", va="center", fontsize=7,
                            color="white" if abs(matrix[i, j]) > vmax * 0.6 else "black")
        fig.tight_layout()
        self.save_figure(fig, "context_scaling_heatmap")
        plt.close()

        self.conclude(
            f"Context scaling (P={P}, fraction={fraction:.0%}, "
            f"seq_len {min(seq_lens)}–{max(seq_lens):,}) across {len(all_results)} models: "
            f"{n_large_negative}/{n_large_total} large models (>=9B) show "
            f"Spearman r < −0.5 (negative trend across all seq_len points). "
            f"{'Pooling tolerance increases with context length.' if large_improves else 'No clear context-length benefit detected.'}"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _load_wikitext(max_samples: int = 1000) -> str:
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
    Exp04ContextScaling().main()
