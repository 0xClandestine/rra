"""
Experiment harness — enforces process rules at runtime.

Every experiment inherits from Experiment and calls:
  self.expect(condition, label)     — required (≥1), machine-checks hypothesis criteria
  self.save_results(data)           — required, saves JSON to results/
  self.save_figure(fig, name)       — required (at least once), saves PNG to figures/
  self.conclude(reason)             — required, records verdict derived from expect() calls

Missing any required call → non-zero exit.
Uncommitted changes → non-zero exit (untracked files are ignored).
Re-runs are auto-numbered: exp01_name_r2, exp01_name_r3, etc.
"""

import hashlib
import inspect
import json
import platform
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
REQUIRED_FIELDS = ["Null hypothesis", "Success", "Failure", "Model", "Date"]

class Experiment(ABC):
    def __init__(self):
        self._module = sys.modules[self.__class__.__module__]
        self._doc = inspect.getdoc(self._module) or ""
        self._base = self._parse_base()
        self._start = time.time()
        self._expectations = []
        self._results_saved = False
        self._figures_saved = []
        self._conclusion = None

        self._validate_docstring()
        self._check_clean_tree()

        REPO_ROOT.joinpath("results").mkdir(exist_ok=True)
        REPO_ROOT.joinpath("figures").mkdir(exist_ok=True)
        REPO_ROOT.joinpath("findings").mkdir(exist_ok=True)

        self._prefix = self._next_prefix()

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self):
        """Implement the experiment here."""

    def expect(self, condition: bool, label: str) -> None:
        """
        Assert a measurable condition against the hypothesis.

        All expects are evaluated and recorded. conclude() derives its
        verdict from them — the agent cannot self-report a pass when
        numbers say otherwise.

        Example:
            self.expect(diffuse_frac > 0.5, f"diffuse_frac={diffuse_frac:.3f} > 0.5")
        """
        status = "PASS" if condition else "FAIL"
        self._expectations.append({"condition": condition, "label": label})
        print(f"  expect [{status}] {label}")

    def checkpoint(self, data: dict) -> None:
        """Persist current progress to results/expXX_checkpoint.json (overwritten each call).

        Call at the end of each model's loop iteration so a crash mid-run
        doesn't lose completed work. The checkpoint is separate from the
        final save_results() output and is cleaned up when that call succeeds.
        """
        path = REPO_ROOT / "results" / f"{self._prefix}_checkpoint.json"
        with open(path, "w") as f:
            json.dump({"meta": self._meta(), "checkpoint": True, **data}, f, indent=2)

    def save_results(self, data: dict, suffix: str = "") -> Path:
        """Save data as JSON to results/expXX_<suffix>.json."""
        name = f"{self._prefix}_{suffix}.json" if suffix else f"{self._prefix}.json"
        path = REPO_ROOT / "results" / name
        with open(path, "w") as f:
            json.dump({"meta": self._meta(), **data}, f, indent=2)
        self._results_saved = True
        print(f"  results → {path.relative_to(REPO_ROOT)}")
        # Remove checkpoint now that final results are written
        ckpt = REPO_ROOT / "results" / f"{self._prefix}_checkpoint.json"
        if ckpt.exists():
            ckpt.unlink()
        return path

    def save_figure(self, fig, name: str) -> Path:
        """Save a matplotlib figure to figures/expXX_<name>.png."""
        fname = f"{self._prefix}_{name}.png"
        path = REPO_ROOT / "figures" / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        self._figures_saved.append(path)
        print(f"  figure  → {path.relative_to(REPO_ROOT)}")
        return path

    def conclude(self, reason: str) -> None:
        """
        Record the hypothesis verdict.

        passed/failed is derived from all prior expect() calls — every
        condition must pass for the verdict to be CONFIRMED.

        reason: one or two sentences citing the key numbers.
        """
        if not self._expectations:
            raise RuntimeError(
                "conclude() requires at least one expect() call first — "
                "encode your success criterion as machine-checkable conditions."
            )
        passed = all(e["condition"] for e in self._expectations)
        self._conclusion = {
            "passed": passed,
            "reason": reason,
            "expectations": self._expectations,
        }
        verdict = "CONFIRMED" if passed else "REFUTED"
        n_pass = sum(e["condition"] for e in self._expectations)
        n_total = len(self._expectations)
        print(f"  verdict → {verdict} ({n_pass}/{n_total} conditions met): {reason}")
        self._write_finding_stub(passed, reason)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def main(self):
        print(f"\n=== {self._prefix} ===")
        try:
            self.run()
        except Exception:
            import traceback
            traceback.print_exc()
            sys.exit(1)

        elapsed = time.time() - self._start

        errors = []
        if not self._expectations:
            errors.append("expect() was never called — hypothesis criteria not encoded")
        if not self._results_saved:
            errors.append("save_results() was never called — no JSON output produced")
        if not self._figures_saved:
            errors.append("save_figure() was never called — no figure produced")
        if self._conclusion is None:
            errors.append("conclude() was never called — hypothesis verdict not recorded")

        if errors:
            print("\nPROCESS VIOLATION:")
            for e in errors:
                print(f"  ✗ {e}")
            sys.exit(1)

        print(f"\nDone in {elapsed:.1f}s. {len(self._figures_saved)} figure(s) saved.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_base(self) -> str:
        mod_name = self.__class__.__module__
        # When run via python -m, __module__ is __main__; derive from __spec__
        if mod_name == "__main__":
            spec = getattr(sys.modules.get("__main__"), "__spec__", None)
            if spec is not None and spec.name != "__main__":
                mod_name = spec.name
            else:
                # Fallback: derive from file path
                f = inspect.getfile(self.__class__)
                p = Path(f).relative_to(REPO_ROOT)
                mod_name = str(p.with_suffix("")).replace("/", ".")
        name = mod_name.split(".")[-1]  # e.g. exp01_head_skew
        if not re.match(r"exp\d+", name):
            raise ValueError(
                f"Module name '{name}' must start with 'expNN_' (e.g. exp01_foo)\n"
                f"  (derived from: {mod_name})"
            )
        return name

    def _next_prefix(self) -> str:
        """Return expXX_name for the first run, expXX_name_r2 for the second, etc."""
        existing = list((REPO_ROOT / "results").glob(f"{self._base}*.json"))
        if not existing:
            return self._base
        run = 1
        for p in existing:
            m = re.search(rf"{re.escape(self._base)}_r(\d+)", p.stem)
            if m:
                run = max(run, int(m.group(1)))
        prefix = f"{self._base}_r{run + 1}"
        print(f"  (prior results exist — this run is {prefix})")
        return prefix

    def _validate_docstring(self):
        missing = [f for f in REQUIRED_FIELDS if f not in self._doc]
        if missing:
            raise ValueError(
                f"Experiment docstring is missing required fields: {missing}\n"
                f"Required: {REQUIRED_FIELDS}"
            )

    def _check_clean_tree(self):
        try:
            result = subprocess.run(
                # --untracked-files=no: ignore untracked files (.DS_Store, __pycache__, etc.)
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            )
            dirty = result.stdout.strip()
        except Exception:
            return  # can't check, don't block
        if dirty:
            raise RuntimeError(
                "Uncommitted changes to tracked files — commit before running so "
                "the git hash in results is meaningful.\n\n"
                f"{dirty}\n\n"
                "Commit your changes first."
            )

    def _meta(self) -> dict:
        return {
            "experiment": self._prefix,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "elapsed_s": round(time.time() - self._start, 2),
            "argv": sys.argv[1:],
            "git_commit": self._git_commit(),
            "script_hash": self._script_hash(),
            "env": self._env(),
        }

    def _git_commit(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _script_hash(self) -> str:
        try:
            src = Path(inspect.getfile(self._module)).read_bytes()
            return hashlib.sha256(src).hexdigest()[:12]
        except Exception:
            return "unknown"

    def _env(self) -> dict:
        env = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        }
        for pkg in ("mlx", "numpy", "mlx_lm"):
            try:
                env[pkg] = __import__(pkg).__version__
            except Exception:
                pass
        return env

    def _write_finding_stub(self, passed: bool, reason: str) -> None:
        slug = self._base.replace("_", "-")
        path = REPO_ROOT / "findings" / f"{slug}.md"
        if path.exists():
            return  # don't overwrite an existing finding
        status = "confirmed" if passed else "uncertain"
        null_hyp = self._extract_field("Null hypothesis")
        success = self._extract_field("Success")
        failure = self._extract_field("Failure")
        expects = "\n".join(
            f"- [{'PASS' if e['condition'] else 'FAIL'}] {e['label']}"
            for e in self._expectations
        )
        stub = f"""# {self._base}: <fill in one-line finding>

Status: {status}

## What we measured

<!-- describe what the experiment measured -->

## What the numbers were

<!-- paste key numbers from results/{self._base}*.json -->

## Hypothesis evaluation

{expects}

## What we conclude

{reason}

## What we're uncertain about

<!-- list open questions -->

## Next experiment

<!-- what should we run next? -->

## Source
- Experiment: experiments/{self._base}.py
- Data: results/{self._base}*.json
- Null hypothesis: {null_hyp}
- Success criterion: {success}
- Failure criterion: {failure}
"""
        path.write_text(stub)
        print(f"  finding → {path.relative_to(REPO_ROOT)} (stub — fill in details)")

    def _extract_field(self, field: str) -> str:
        m = re.search(rf"{field}:\s*(.+)", self._doc)
        return m.group(1).strip() if m else ""
