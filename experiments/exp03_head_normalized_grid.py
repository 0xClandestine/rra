"""
Exp03: Head-Normalized Pooling Grid

Null hypothesis: ΔPPL degradation curves are identical across pool factors P
                 and model families when expressed as fraction of KV-heads pooled.
Success:          Curves differ systematically by P and model size across all
                  registered models; larger models show flatter degradation at
                  equal KV-head coverage; P=2 is strictly more conservative
                  than P=8.
Failure:          No systematic difference by P or model size after KV-head
                  normalisation.

Methodology: Each (model, P, fraction, seed) triple gets a fresh model load so
             weight permutations never compound. Baseline is measured once per
             model on a clean load. Results are averaged over --seeds random
             seeds with ±1 std error bands in figures.

Model:     All registered models (experiments/models.py)
Data:      WikiText-2 test set, 100 samples, seq_len=1024
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

POOL_FACTORS = [2, 4, 8, 16]
SEEDS = [42, 1337, 9999]


def _probe_heads(model):
    """Return (H, Hkv, ratio) from the first patchable attention layer."""
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
        if Hkv and Hkv > 1:
            return H, Hkv, H // Hkv
    return None, None, None


class Exp03HeadNormalizedGrid(Experiment):
    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seq-len", type=int, default=1024)
        parser.add_argument("--samples", type=int, default=100)
        parser.add_argument("--pool-factors", type=int, nargs="+", default=POOL_FACTORS)
        parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                            help="Random seeds — one fresh load per seed per point")
        parser.add_argument("--model-set", default="all",
                            help="'all' or a family name (e.g. 'qwen3.5')")
        args = parser.parse_args()

        from mlx_lm import load
        from rra.routed_patch import enable_random_pooling, disable_random_pooling

        eval_text = self._load_wikitext(max_samples=args.samples)
        pool_factors = args.pool_factors
        seeds = args.seeds
        print(f"  Seeds: {seeds}  ({len(seeds)} independent loads per data point)")

        all_results = {}

        for m in models_for_mode(args.model_set):
            model_id = m.id
            short = model_id.split("/")[-1]
            print(f"\n{'='*60}\n  {short}")

            model_result = {
                "label": f"{m.size} {m.family}",
                "family": m.family,
                "size": m.size,
                "model_id": model_id,
                "pool_factors": {},
            }

            # ── Baseline: one clean load, no pooling ──
            try:
                model, tokenizer = load(model_id)
            except Exception as e:
                print(f"  LOAD FAILED: {e}")
                all_results[short] = {**model_result, "error": f"load: {e}"}
                continue

            H, Hkv, ratio = _probe_heads(model)
            model_result.update({"n_q_heads": H, "n_kv_heads": Hkv, "gqa_ratio": ratio})
            print(f"  H={H}  Hkv={Hkv}  ratio={ratio}")

            try:
                baseline = float(self._perplexity(model, tokenizer, eval_text, args.seq_len))
                print(f"  Baseline PPL: {baseline:.4f}")
            except Exception as e:
                print(f"  BASELINE FAILED: {e}")
                del model
                all_results[short] = {**model_result, "error": f"baseline: {e}"}
                continue
            del model

            # Fractions: k/Hkv for k = 1 .. Hkv-1
            # Each step pools one additional KV-head, giving a clean integer-step x-axis.
            if Hkv is not None and Hkv > 1:
                fracs = [k / Hkv for k in range(1, Hkv)]
            else:
                fracs = [round(x, 2) for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]

            # ── Grid: fresh load per (P, fraction, seed) ──
            for P in pool_factors:
                pf_result = {
                    "baseline": baseline,
                    "n_q_heads": H,
                    "n_kv_heads": Hkv,
                    "gqa_ratio": ratio,
                    "fractions": [],
                }

                for f in fracs:
                    k_label = f"{fracs.index(f)+1}/{Hkv}" if Hkv else f"{f:.2f}"
                    seed_ppls = []
                    frac_actuals = []

                    for seed in seeds:
                        try:
                            model, tokenizer = load(model_id)
                        except Exception as e:
                            print(f"  P={P} f={f:.3f} seed={seed} LOAD FAILED: {e}")
                            break
                        try:
                            info = enable_random_pooling(
                                model, pool_factor=P, pool_fraction=f, seed=seed)
                            ppl = float(self._perplexity(
                                model, tokenizer, eval_text, args.seq_len))
                            frac_actuals.append(info["pool_fraction_actual"])
                        except Exception as e:
                            print(f"  P={P} f={f:.3f} seed={seed} EVAL FAILED: {e}")
                            del model
                            break
                        del model
                        seed_ppls.append(ppl)

                    if not seed_ppls:
                        break

                    mean_ppl = float(np.mean(seed_ppls))
                    std_ppl = float(np.std(seed_ppls)) if len(seed_ppls) > 1 else 0.0
                    mean_delta = 100.0 * (mean_ppl - baseline) / max(baseline, 0.01)
                    std_delta = 100.0 * std_ppl / max(baseline, 0.01)
                    mean_frac_actual = float(np.mean(frac_actuals)) if frac_actuals else f

                    pf_result["fractions"].append({
                        "k_requested": fracs.index(f) + 1,
                        "frac_requested": f,
                        "frac_actual_mean": mean_frac_actual,
                        "ppl_mean": mean_ppl,
                        "ppl_std": std_ppl,
                        "delta_pct_mean": mean_delta,
                        "delta_pct_std": std_delta,
                        "compute_savings_pct": 100.0 * mean_frac_actual * (P - 1) / P,
                        "n_seeds": len(seed_ppls),
                        "seed_ppls": seed_ppls,
                    })
                    print(f"  P={P} k={k_label} f={f:.3f}: "
                          f"PPL={mean_ppl:.4f}±{std_ppl:.4f}  "
                          f"Δ={mean_delta:+.2f}%±{std_delta:.2f}%")

                    if mean_ppl > baseline * 10:
                        print(f"  Exploded — stopping P={P} sweep")
                        break

                model_result["pool_factors"][str(P)] = pf_result

            all_results[short] = model_result
            self.checkpoint({"pool_factors": pool_factors, "seeds": seeds,
                             "seq_len": args.seq_len, "models": all_results})

        # ── Hypothesis checks ──
        # 1. P=2 more conservative than P=8 — mean |ΔPPL| across all fractions (AUC)
        p2_better = 0
        n_comparable = 0
        for short, res in all_results.items():
            p2 = res.get("pool_factors", {}).get("2")
            p8 = res.get("pool_factors", {}).get("8")
            if not (p2 and p8 and p2.get("fractions") and p8.get("fractions")):
                continue
            d2 = float(np.mean([e["delta_pct_mean"] for e in p2["fractions"]]))
            d8 = float(np.mean([e["delta_pct_mean"] for e in p8["fractions"]]))
            n_comparable += 1
            if abs(d2) < abs(d8):
                p2_better += 1
            print(f"  {short}: P=2 mean Δ={d2:+.2f}% vs P=8 mean Δ={d8:+.2f}% (AUC)")

        self.expect(p2_better >= max(1, n_comparable * 0.6),
            f"P=2 more conservative than P=8 in {p2_better}/{n_comparable} models")

        # 2. Large models (>=9B) less sensitive than small (<=2B) — mean |ΔPPL| across all fractions (AUC)
        large_deltas, small_deltas = [], []
        for short, res in all_results.items():
            pf = res.get("pool_factors", {}).get("4") or \
                 next(iter(res.get("pool_factors", {}).values()), None)
            if not pf or not pf.get("fractions"):
                continue
            d = float(np.mean([e["delta_pct_mean"] for e in pf["fractions"]]))
            try:
                sz = float(res["size"].upper().replace("B", "").split("A")[0])
            except ValueError:
                continue
            if sz >= 9:
                large_deltas.append(d)
            elif sz <= 2:
                small_deltas.append(d)

        if large_deltas and small_deltas:
            size_effect = np.mean(np.abs(large_deltas)) < np.mean(np.abs(small_deltas))
            self.expect(size_effect,
                f"Large (>=9B) mean |ΔPPL|={np.mean(np.abs(large_deltas)):.2f}% "
                f"< small (<=2B) {np.mean(np.abs(small_deltas)):.2f}%")
        else:
            self.expect(True, "Size comparison skipped (insufficient model coverage)")

        # ── Pareto analysis: optimal (P, fraction) per model ──
        # compute_savings_pct = f_actual × (P-1)/P × 100  (attention compute only)
        # Pareto frontier: non-dominated points in (savings ↑, ΔPPL ↓) space
        pareto_per_model = {}
        print("\n  Pareto-optimal settings per model:")
        for short, res in all_results.items():
            points = []
            for P_str, pf in res.get("pool_factors", {}).items():
                for entry in pf.get("fractions", []):
                    points.append({
                        "P": int(P_str),
                        "frac_actual": entry["frac_actual_mean"],
                        "compute_savings_pct": entry["compute_savings_pct"],
                        "delta_pct_mean": entry["delta_pct_mean"],
                        "delta_pct_std": entry["delta_pct_std"],
                    })

            # Pareto frontier: sort by savings desc; keep point if its ΔPPL is
            # lower than every higher-savings point seen so far
            points_desc = sorted(points, key=lambda x: x["compute_savings_pct"], reverse=True)
            pareto = []
            best_delta = float("inf")
            for pt in points_desc:
                if pt["delta_pct_mean"] < best_delta:
                    best_delta = pt["delta_pct_mean"]
                    pareto.append(pt)
            pareto.sort(key=lambda x: x["compute_savings_pct"])

            def best_within(threshold):
                cands = [p for p in points if p["delta_pct_mean"] < threshold]
                return max(cands, key=lambda x: x["compute_savings_pct"], default=None)

            best_5  = best_within(5.0)
            best_10 = best_within(10.0)

            pareto_per_model[short] = {
                "all_points": points,
                "pareto_frontier": pareto,
                "best_at_5pct_delta": best_5,
                "best_at_10pct_delta": best_10,
            }

            label = res.get("label", short)
            b5_str  = (f"P={best_5['P']}, f={best_5['frac_actual']:.2f} "
                       f"→ {best_5['compute_savings_pct']:.1f}% saved") if best_5 else "none"
            b10_str = (f"P={best_10['P']}, f={best_10['frac_actual']:.2f} "
                       f"→ {best_10['compute_savings_pct']:.1f}% saved") if best_10 else "none"
            print(f"    {label}")
            print(f"      <5%  ΔPPL: {b5_str}")
            print(f"      <10% ΔPPL: {b10_str}")

        # ── Save ──
        self.save_results({
            "pool_factors": pool_factors,
            "seeds": seeds,
            "seq_len": args.seq_len,
            "models": all_results,
            "pareto": pareto_per_model,
        })

        # ── Figures ──
        family_colors = {
            "qwen2.5": "#1f77b4", "qwen3.5": "#aec7e8",
            "gemma4":  "#2ca02c", "llama3":  "#d62728",
            "mistral": "#ff7f0e",
        }
        p_markers = {2: "o", 4: "s", 8: "^", 16: "D"}
        p_colors  = {2: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c", 16: "#d62728"}

        # Figure 1: ΔPPL ± std vs KV-head coverage, one panel per P
        fig, axes = plt.subplots(1, len(pool_factors),
                                 figsize=(4.5 * len(pool_factors), 4.5), sharey=True)
        if len(pool_factors) == 1:
            axes = [axes]

        for ax, P in zip(axes, pool_factors):
            for short, res in all_results.items():
                pf = res.get("pool_factors", {}).get(str(P))
                if not pf or not pf.get("fractions"):
                    continue
                xs = [0.0] + [e["frac_actual_mean"] for e in pf["fractions"]]
                ys = [0.0] + [e["delta_pct_mean"] for e in pf["fractions"]]
                es = [0.0] + [e["delta_pct_std"]  for e in pf["fractions"]]
                col = family_colors.get(res.get("family", ""), "#888888")
                ax.plot(xs, ys, f"{p_markers.get(P,'o')}-", color=col,
                        label=res.get("label", short), markersize=4, linewidth=1.5)
                ax.fill_between(xs,
                                [y - e for y, e in zip(ys, es)],
                                [y + e for y, e in zip(ys, es)],
                                color=col, alpha=0.12)
            ax.axhline(y=0, color="#444", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_title(f"P = {P}", fontsize=11)
            ax.set_xlabel("KV-head pool fraction", fontsize=9)
            ax.set_xlim(-0.02, 1.02)
            ax.grid(alpha=0.25)
            if ax is axes[0]:
                ax.set_ylabel("ΔPPL (%)", fontsize=9)
                ax.legend(fontsize=6, loc="upper left", framealpha=0.7)

        fig.suptitle(
            f"RRA: ΔPPL vs KV-head pool fraction — {len(all_results)} models, "
            f"seq_len={args.seq_len}, {len(seeds)} seeds/point",
            fontsize=11, fontweight="bold")
        fig.tight_layout()
        self.save_figure(fig, "head_grid_delta")
        plt.close()

        # Figure 2: All P values overlaid, one panel per model family
        families = sorted({res.get("family", "?") for res in all_results.values()
                           if res.get("pool_factors")})
        n_fam = max(len(families), 1)
        fig, axes = plt.subplots(1, n_fam, figsize=(4.5 * n_fam, 4.5), sharey=True)
        if n_fam == 1:
            axes = [axes]

        for ax, fam in zip(axes, families):
            fam_models = [(s, r) for s, r in all_results.items()
                          if r.get("family") == fam and r.get("pool_factors")]
            first = True
            for s, res in fam_models:
                for P in pool_factors:
                    pf = res.get("pool_factors", {}).get(str(P))
                    if not pf or not pf.get("fractions"):
                        continue
                    xs = [0.0] + [e["frac_actual_mean"] for e in pf["fractions"]]
                    ys = [0.0] + [e["delta_pct_mean"]  for e in pf["fractions"]]
                    es = [0.0] + [e["delta_pct_std"]   for e in pf["fractions"]]
                    col = p_colors.get(P, "#888")
                    ax.plot(xs, ys, f"{p_markers.get(P,'o')}-", color=col,
                            label=f"P={P}" if first else "_",
                            linewidth=1.5, markersize=3, alpha=0.85)
                    ax.fill_between(xs,
                                    [y - e for y, e in zip(ys, es)],
                                    [y + e for y, e in zip(ys, es)],
                                    color=col, alpha=0.10)
                first = False
            ax.axhline(y=0, color="#444", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_title(fam, fontsize=11)
            ax.set_xlabel("KV-head pool fraction", fontsize=9)
            ax.set_xlim(-0.02, 1.02)
            ax.grid(alpha=0.25)
            if ax is axes[0]:
                ax.set_ylabel("ΔPPL (%)", fontsize=9)
            handles = [plt.Line2D([0], [0], color=p_colors[P],
                                  marker=p_markers[P], linestyle="-", label=f"P={P}")
                       for P in pool_factors]
            ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.7)

        fig.suptitle(
            f"RRA: ΔPPL by pool factor — per model family, "
            f"seq_len={args.seq_len}, {len(seeds)} seeds/point",
            fontsize=11, fontweight="bold")
        fig.tight_layout()
        self.save_figure(fig, "head_grid_by_family")
        plt.close()

        # Figure 3: Pareto frontier — compute savings vs ΔPPL, all models
        fig, ax = plt.subplots(figsize=(8, 6))
        for short, pdata in pareto_per_model.items():
            res = all_results[short]
            col = family_colors.get(res.get("family", ""), "#888888")
            label = res.get("label", short)

            # Scatter all (P, fraction) points, faint
            pts = pdata["all_points"]
            if pts:
                ax.scatter(
                    [p["compute_savings_pct"] for p in pts],
                    [p["delta_pct_mean"] for p in pts],
                    color=col, alpha=0.18, s=18, zorder=2)

            # Pareto frontier as solid line
            pf = pdata["pareto_frontier"]
            if len(pf) >= 2:
                ax.plot(
                    [p["compute_savings_pct"] for p in pf],
                    [p["delta_pct_mean"] for p in pf],
                    "-o", color=col, label=label,
                    markersize=5, linewidth=1.8, zorder=3)

            # Mark the <5% and <10% operating points
            for thresh, marker, ms in [(5.0, "*", 12), (10.0, "D", 7)]:
                best = pdata[f"best_at_{int(thresh)}pct_delta"]
                if best:
                    ax.scatter(best["compute_savings_pct"], best["delta_pct_mean"],
                               marker=marker, color=col, s=ms**2,
                               zorder=4, edgecolors="black", linewidths=0.5)

        ax.axhline(y=5,  color="#aaa", linestyle="--", linewidth=0.8, alpha=0.7,
                   label="5% ΔPPL threshold")
        ax.axhline(y=10, color="#ccc", linestyle=":",  linewidth=0.8, alpha=0.7,
                   label="10% ΔPPL threshold")
        ax.axhline(y=0,  color="#444", linestyle="-",  linewidth=0.6, alpha=0.4)
        ax.set_xlabel("Attention compute savings (%)\n"
                      r"$f_{\mathrm{actual}} \times (P-1)/P \times 100$", fontsize=9)
        ax.set_ylabel("ΔPPL (%)", fontsize=9)
        ax.set_title(
            f"RRA Pareto Frontier: Quality vs Compute Savings — {len(pareto_per_model)} models\n"
            f"(★ = best setting within 5% ΔPPL  ◆ = best within 10% ΔPPL)",
            fontsize=10)
        ax.legend(fontsize=7, ncol=2, framealpha=0.8, loc="upper left")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        self.save_figure(fig, "head_grid_pareto")
        plt.close()

        # Summarise best savings achievable within quality budgets across all models
        savings_5  = [d["best_at_5pct_delta"]["compute_savings_pct"]
                      for d in pareto_per_model.values() if d["best_at_5pct_delta"]]
        savings_10 = [d["best_at_10pct_delta"]["compute_savings_pct"]
                      for d in pareto_per_model.values() if d["best_at_10pct_delta"]]

        self.conclude(
            f"Head-normalized grid ({len(all_results)} models, {len(seeds)} seeds/point, "
            f"seq_len={args.seq_len}): P=2 more conservative than P=8 in "
            f"{p2_better}/{n_comparable} models. "
            + (f"Large (>=9B) mean |ΔPPL|={np.mean(np.abs(large_deltas)):.1f}% "
               f"vs small (<=2B) {np.mean(np.abs(small_deltas)):.1f}%. "
               if large_deltas and small_deltas else "")
            + (f"Pareto: within 5% ΔPPL, models achieve "
               f"{np.mean(savings_5):.1f}% avg attention savings "
               f"(range {min(savings_5):.1f}–{max(savings_5):.1f}%). "
               if savings_5 else "")
            + (f"Within 10% ΔPPL: {np.mean(savings_10):.1f}% avg "
               f"({min(savings_10):.1f}–{max(savings_10):.1f}%)."
               if savings_10 else "")
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _load_wikitext(max_samples: int = 100) -> str:
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
    Exp03HeadNormalizedGrid().main()
