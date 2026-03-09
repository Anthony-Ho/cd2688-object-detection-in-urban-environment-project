#!/usr/bin/env bash
# watch_phase1_and_commit.sh
# Polls every 5 minutes for Phase-2 Round-1 completion.
# Triggers only when all 8 combo metrics.json files exist with status=ok.
#
# Usage:
#   chmod +x watch_phase1_and_commit.sh
#   nohup ./watch_phase1_and_commit.sh &
#   tail -f watch_phase1.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$REPO_ROOT/watch_phase1.log"
POLL_INTERVAL=300   # seconds

P2_RUNS="$REPO_ROOT/1_model_training/experiments/phase2_runs"

COMBO_IDS=(
    baseline_low
    baseline_high
    geometry_low
    geometry_high
    appearance_low
    appearance_high
    mixed_low
    mixed_high
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# Returns "ok", "failed", or "missing" for a combo's r1 metrics.json
check_combo() {
    local combo="$1"
    local mj="$P2_RUNS/${combo}__r1/metrics.json"
    if [[ ! -f "$mj" ]]; then echo "missing"; return; fi
    python3 -c "import json; print(json.load(open('$mj')).get('status','?'))" 2>/dev/null || echo "?"
}

log "Watcher started. Polling every ${POLL_INTERVAL}s — waiting for all 8 Phase-2 R1 combos to reach status=ok."

while true; do
    NOT_OK=()
    for combo in "${COMBO_IDS[@]}"; do
        status=$(check_combo "$combo")
        [[ "$status" != "ok" ]] && NOT_OK+=("${combo}=${status}")
    done

    if [[ ${#NOT_OK[@]} -eq 0 ]]; then
        log "All 8 Phase-2 Round-1 runs completed with status=ok. Committing..."
        cd "$REPO_ROOT"

        # Stage notebook (bug fixes applied during Phase-2 R1)
        git add 1_model_training/1_train_model.ipynb

        # Stage Phase-1 summary (architecture_selection.json + winner_config_path)
        if [[ -d 1_model_training/experiments/phase1_summary ]]; then
            git add 1_model_training/experiments/phase1_summary/
        fi

        # Stage Phase-2 generated configs
        if ls 1_model_training/source_dir/pipeline.phase2_*.config 2>/dev/null | head -1 | grep -q .; then
            git add 1_model_training/source_dir/pipeline.phase2_*.config
        fi

        # Stage Phase-2 summary artifacts (combo_manifest, p2_r1_results, etc.)
        if [[ -d 1_model_training/experiments/phase2_summary ]]; then
            git add 1_model_training/experiments/phase2_summary/
        fi

        # Stage .gitignore if modified
        git add .gitignore

        if git diff --cached --quiet; then
            log "Nothing new to commit. All changes may already be committed."
        else
            git commit -m "$(cat <<'EOF'
Complete Phase-2 Round-1 screen runs (8/8 ok)

Phase-2 Round-1 results (8 combos x 5,000 steps):
- baseline_low / baseline_high: h-flip + scale-crop aug
- geometry_low / geometry_high: + random_rotation90 + random_pad_image
- appearance_low / appearance_high: + random_distort_color + jpeg aug
- mixed_low / mixed_high: geometry + appearance combined

Bug fixes applied to notebook during this session:
- TypeError on None score in format strings (cells P2-D/E/F/G/H)
- P2-H winner filter: wrong key 'status' -> 'score_mean'
- Cell e38ri4v6ly: write winner_config_path to architecture_selection.json
- AUG_GEOMETRY: random_rotation -> random_rotation90 (correct proto field)
- AUG_APPEARANCE: min/max_quality -> min/max_jpeg_quality (correct proto field)

Phase-1 summary updated: architecture_selection.json includes
winner_config_path (eff_d1_anchor_scale2p0_dense).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
            log "Commit done: $(git log --oneline -1)"
        fi

        log "Watcher exiting."
        exit 0
    else
        log "Waiting... not ready (${#NOT_OK[@]}): ${NOT_OK[*]}"
        sleep "$POLL_INTERVAL"
    fi
done
