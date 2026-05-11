#!/usr/bin/env bash
# run_suite.sh — Run the full RRA experiment suite on a remote machine.
#
# Usage:
#   bash experiments/run_suite.sh [--skip N ...] [--only N ...] [--model-set <set>]
#
# Flags:
#   --skip N ...         Skip experiment numbers (e.g. --skip 4 9)
#   --only N ...         Run only these experiment numbers
#   --model-set <set>    'all' (default) or a family name e.g. 'qwen3.5'
#
# All experiments test every model in experiments/models.py.
# A model that fails to load is recorded as an error, not silently skipped.
#
# Prerequisites:
#   1. Python venv at .venv (uv venv .venv && source .venv/bin/activate && uv pip install -e .)
#   2. Git working tree must be clean (harness enforces this)
#   3. Sufficient disk space for model weights
#
# Results:  results/  figures/
# Logs:     logs/run_<timestamp>/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs/run_$TIMESTAMP"
mkdir -p "$LOG_DIR"

# ── Parse flags ──────────────────────────────────────────────────────────────
SKIP_EXPS=()
ONLY_EXPS=()
MODEL_SET="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip)      shift; while [[ $# -gt 0 && "$1" != --* ]]; do SKIP_EXPS+=("$1"); shift; done ;;
        --only)      shift; while [[ $# -gt 0 && "$1" != --* ]]; do ONLY_EXPS+=("$1"); shift; done ;;
        --model-set) shift; MODEL_SET="$1"; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

should_run() {
    local n="$1"
    for s in "${SKIP_EXPS[@]+"${SKIP_EXPS[@]}"}"; do [[ "$s" == "$n" ]] && return 1; done
    if [[ ${#ONLY_EXPS[@]} -gt 0 ]]; then
        for o in "${ONLY_EXPS[@]}"; do [[ "$o" == "$n" ]] && return 0; done
        return 1
    fi
    return 0
}

# ── Activation & sanity checks ───────────────────────────────────────────────
if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ERROR: .venv not found. Run: uv venv .venv && source .venv/bin/activate && uv pip install -e ."
    exit 1
fi
source .venv/bin/activate

echo "=== RRA Experiment Suite ==="
echo "  Repo:      $REPO_ROOT"
echo "  Logs:      $LOG_DIR"
echo "  Model set: $MODEL_SET"
echo "  Python:    $(python --version)"
echo "  Date:      $(date)"
echo ""

DIRTY="$(git status --porcelain --untracked-files=no)"
if [[ -n "$DIRTY" ]]; then
    echo "ERROR: Working tree has uncommitted changes. Commit first."
    echo "$DIRTY"
    exit 1
fi
echo "  Git: $(git rev-parse --short HEAD) (clean)"
echo ""

# ── Experiment runner ─────────────────────────────────────────────────────────
run_exp() {
    local num="$1"; shift
    local module="$1"; shift
    local extra_args=("$@")

    if ! should_run "$num"; then
        echo "--- exp${num}: SKIPPED ---"
        return
    fi

    local log="$LOG_DIR/exp${num}.log"
    echo "--- exp${num}: $module ---"
    [[ ${#extra_args[@]} -gt 0 ]] && echo "    args: ${extra_args[*]}"
    echo "    log:  $log"
    echo ""

    local start; start=$(date +%s)
    if python -m "$module" --model-set "$MODEL_SET" "${extra_args[@]+"${extra_args[@]}"}" 2>&1 | tee "$log"; then
        echo "    DONE in $(( $(date +%s) - start ))s"
    else
        echo "    FAILED after $(( $(date +%s) - start ))s (see $log)"
    fi
    echo ""
}

# ── Run all experiments (auto-detected from experiments/exp*.py) ──────────────
while IFS= read -r f; do
    base="$(basename "$f" .py)"
    num="$(echo "$base" | grep -oE '^exp[0-9]+' | grep -oE '[0-9]+')"
    module="experiments.${base}"
    run_exp "$num" "$module"
done < <(ls experiments/exp[0-9]*.py 2>/dev/null | sort)

# ── Summary ──────────────────────────────────────────────────────────────────
echo "=== Suite complete ==="
echo ""
echo "Results:"
ls -lh results/*.json 2>/dev/null | tail -20 || echo "  (none)"
echo ""
echo "Figures:"
ls -lh figures/*.png 2>/dev/null | tail -20 || echo "  (none)"
