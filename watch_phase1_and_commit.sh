#!/usr/bin/env bash
# watch_phase1_and_commit.sh
# Polls every 5 minutes for Phase-1 confirmation metrics.json files.
# When all 3 are present, stages and commits all pending changes, then exits.
#
# Usage:
#   chmod +x watch_phase1_and_commit.sh
#   nohup ./watch_phase1_and_commit.sh &
#   tail -f watch_phase1.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$REPO_ROOT/watch_phase1.log"
POLL_INTERVAL=300   # seconds

RUNS_DIR="$REPO_ROOT/1_model_training/experiments/phase1_runs"

# The 3 metrics.json files that signal Phase-1 confirmation is complete.
SENTINEL_1="$RUNS_DIR/eff_d1_res768__confirm_r2/metrics.json"
SENTINEL_2="$RUNS_DIR/eff_d1_anchor_scale2p0_dense__confirm_r1/metrics.json"
SENTINEL_3="$RUNS_DIR/eff_d1_anchor_scale2p0_dense__confirm_r2/metrics.json"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "Watcher started. Polling every ${POLL_INTERVAL}s."
log "Watching for:"
log "  $SENTINEL_1"
log "  $SENTINEL_2"
log "  $SENTINEL_3"

while true; do
    if [[ -f "$SENTINEL_1" && -f "$SENTINEL_2" && -f "$SENTINEL_3" ]]; then
        log "All 3 sentinel files found. Committing..."
        cd "$REPO_ROOT"

        # Stage all relevant files
        git add 1_model_training/1_train_model.ipynb
        git add 1_model_training/EXPERIMENT_STRATEGY_WRITEUP.md
        git add CLAUDE.md

        # Stage phase1_summary artifacts if they exist
        if [[ -d 1_model_training/experiments/phase1_summary ]]; then
            git add 1_model_training/experiments/phase1_summary/
        fi

        # Stage any SSD pipeline configs generated during Phase 1
        if ls 1_model_training/source_dir/pipeline.phase1_ssd_*.config 2>/dev/null | head -1 | grep -q .; then
            git add 1_model_training/source_dir/pipeline.phase1_ssd_*.config
        fi

        # Check if there is anything staged before committing
        if git diff --cached --quiet; then
            log "Nothing staged to commit. All changes may already be committed."
        else
            git commit -m "$(cat <<'EOF'
Complete Phase-1 confirmation runs and add Phase-2 implementation

Phase-1 confirmation runs complete:
- eff_d1_res768 x2 at 15,000 steps
- eff_d1_anchor_scale2p0_dense x2 at 15,000 steps

Add Phase-2 Generalization and Configuration Optimization cells:
- 4x2 aug/reg grid (baseline/geometry/appearance/mixed x low/high-decay)
- Round 1 screen (8 combos x 5,000 steps, cache-aware)
- Round 2 (top 4 x 10,000 steps, cache-aware)
- Confirmation (top 2 x 2 repeats x 10,000 steps)
- Config builder and phase2_winner.config output

Update EXPERIMENT_STRATEGY_WRITEUP.md and add CLAUDE.md.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
            log "Commit done: $(git log --oneline -1)"
        fi

        log "Watcher exiting."
        exit 0
    else
        MISSING=()
        [[ ! -f "$SENTINEL_1" ]] && MISSING+=("eff_d1_res768__confirm_r2")
        [[ ! -f "$SENTINEL_2" ]] && MISSING+=("eff_d1_anchor_scale2p0_dense__confirm_r1")
        [[ ! -f "$SENTINEL_3" ]] && MISSING+=("eff_d1_anchor_scale2p0_dense__confirm_r2")
        log "Waiting... still missing: ${MISSING[*]}"
        sleep "$POLL_INTERVAL"
    fi
done
