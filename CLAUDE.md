# CLAUDE.md — Project Guide

## Project Overview

Urban-environment object detection using the **TensorFlow 2 Object Detection API** on **AWS SageMaker**, with a local GPU (RTX 3090) screening loop to minimise cloud spend.

Three classes: **vehicle**, **pedestrian**, **cyclist** (Waymo Open Dataset).

---

## Repository Layout

```
1_model_training/
│   1_train_model.ipynb          # Main notebook — all experiment phases live here
│   EXPERIMENT_STRATEGY_WRITEUP.md  # Full experiment strategy & findings document
│   TRAINING_NOTEBOOK_WALKTHROUGH.md
│   framework.py                 # CustomFramework: SageMaker Estimator subclass
│   source_dir/
│     pipeline.config            # EfficientDet-D1 baseline config (production)
│     pipeline.phase0_*.config   # Phase-0 probe configs (EfficientDet variants)
│     pipeline.phase1_*.config   # Phase-1 arch configs (SSD/ResNet; generated)
│     run_training.sh            # Docker/SageMaker training entrypoint
│     model_main_tf2.py          # TF2 OD API train/eval script (from TF Model Garden)
│     exporter_main_v2.py        # SavedModel exporter
│     checkpoints/               # Pretrained checkpoints (not committed)
│   docker/
│     Dockerfile                 # ROCm base (AMD GPU — legacy)
│     Dockerfile.nvidia          # NVIDIA base (RTX 3090 — active)
│     Dockerfile.nvidiaV2
│     build_and_push.sh          # Build + ECR push helper
│     models/                    # TF Model Garden clone (not committed to git)
│   experiments/
│     phase0_runs/               # Per-run output dirs from Phase-0 screen/confirm
│     phase0_summary/            # Aggregated Phase-0 JSON/CSV artifacts
│     phase1_runs/               # Per-run output dirs from Phase-1 screen/confirm
│     phase1_summary/            # Aggregated Phase-1 JSON/CSV artifacts
│   data/train/ data/val/        # Local TFRecord shards + label_map.pbtxt

2_run_inference/
│   2_deploy_model.ipynb         # SageMaker endpoint deploy + inference
│   visualization_utils.py       # Bounding-box drawing helpers

aws/install                      # AWS CLI / SDK setup scripts
data/                            # Symlinks / raw data (Waymo)
pipeline.config                  # Top-level alias (points to source_dir config)
```

---

## Experiment Strategy (Summary)

The project uses a **three-phase funnel** to select the best architecture cheaply on local GPU before paying for SageMaker:

| Phase | Goal | Budget |
|---|---|---|
| **Phase 0** | Screen 10 EfficientDet-D1 hyperparameter probes; select top families | 3,500 steps screen + 10,000 steps confirm (top 3 × 2 repeats) |
| **Phase 1** | Compare 4 architecture candidates (2 EfficientDet + 2 SSD/ResNet FPN); select one | 5,000 steps screen + 15,000 steps confirm (top 2 × 2 repeats) |
| **Phase 2** | Full SageMaker training of the winning architecture | ~25,000–50,000 steps |

**Ranking metric:** `score = 0.7 × mAP(small) + 0.3 × mAP` — weighted towards small objects because the Waymo dataset is small-object-heavy.

**Phase-0 winners (promoted to Phase-1):**
- `eff_d1_res768` — best mAP (0.164) and AR@100
- `eff_d1_anchor_scale2p0_dense` — best mAP(small) (0.095) and composite score

---

## Key Notebook Cells (1_train_model.ipynb)

All cells are identified by their Jupyter cell `id` field:

| Cell ID | Purpose |
|---|---|
| `ef0def9f` | Phase-0 constants, path setup, `extract_eval_metrics`, `_read_last_metric_from_event` |
| `6c72b6e7` | Dataset profiling helpers |
| `3c7c3570` | Config manipulation helpers: `_replace_first`, `_set_numeric_field`, `_set_bool_field`, `_set_image_size`, `_set_fine_tune_checkpoint`, `_replace_data_augmentation_options`, `apply_probe_overrides`, `adapt_schedule_to_steps`, `_write_rows_csv`, `_score_row`, `_aggregate_probe_runs`, `PHASE0_PROBES` |
| `65af2551` | `run_phase0_probe()`, `recommend_families()`, `run_phase0_sweep()` |
| `dbbb4653` | Phase-0 resume helper (`RUN_PHASE0_RESUME_ONLY` flag) |
| `vo2hial95` | **Phase-1 markdown header** |
| `qsd4b9qsrv` | **Phase-1 constants** (`PHASE1_SCREEN_STEPS=5000`, `PHASE1_RUNS_DIR`, `PHASE1_SUMMARY_DIR`) + `_run_phase1_arch()` wrapper |
| `20025lubcke` | **Bash** — download ResNet-50/101 FPN COCO pretrained checkpoints |
| `tvrahp44a4b` | `make_ssd_resnet_config()` + `PHASE1_ARCH_MANIFEST` + saves `arch_manifest.json` |
| `ke7eytd0mec` | Phase-1 screen runs (cache-aware: skips successful runs) |
| `sbsgp5avpyb` | Phase-1 screen leaderboard |
| `nd3taelwdw` | Phase-1 confirmation runs (top-2 × 2 repeats, 15,000 steps) |
| `e38ri4v6ly` | Phase-1 confirmation summary + saves `architecture_selection.json` |
| `iagm3xc8y1` | Phase-1 resume helper (`RUN_PHASE1_RESUME_ONLY` flag) |

---

## Docker / Training Infrastructure

- **Image name:** `tf2-object-detection` (built from `docker/Dockerfile.nvidia`)
- **Base image:** `tensorflow/tensorflow:2.13.0-gpu`
- **GPU:** NVIDIA RTX 3090 (`--gpus all`)
- **Volume mounts used by `run_phase0_probe`:**
  - `source_dir/` → `/opt/ml/code` (configs + scripts, working dir)
  - `data/train/` → `/opt/ml/input/data/train`
  - `data/val/` → `/opt/ml/input/data/val`
  - `experiments/phase{N}_runs/{run_key}/` → `/opt/training` (outputs)
- **Entrypoint:** `/bin/bash run_training.sh`
- **Pipeline config path** is passed by filename only (relative to `/opt/ml/code`)

`run_training.sh` does three things in sequence: train (`model_main_tf2.py`), eval, export (`exporter_main_v2.py`).

---

## Config File Conventions

- **Base configs** (`pipeline.phase0_*.config`, `pipeline.phase1_*.config`) — hand-written or generated; stored in `source_dir/`
- **Runtime configs** — derived per-run by `adapt_schedule_to_steps()`; named `{base}.steps_{N}.{label}.config`; written to `source_dir/` alongside base configs
- **Important fields:**
  - `fine_tune_checkpoint` — relative to `/opt/ml/code` inside the container
  - `fine_tune_checkpoint_type` — `"detection"` for COCO SSD checkpoints; `"detection"` also for EfficientDet COCO checkpoints
  - Data paths always use `/opt/ml/input/data/train/*.tfrecord` and `/opt/ml/input/data/val/*.tfrecord`
  - `label_map_path` uses `/opt/ml/input/data/train/label_map.pbtxt` (train) and `/opt/ml/input/data/val/label_map.pbtxt` (val)

---

## Known Gotchas

1. **`fine_tune_checkpoint_type: "detection"` for SSD/ResNet COCO checkpoints** — The COCO17 tarballs from TF Model Garden contain full SSD detection model checkpoints, not ImageNet backbone-only checkpoints. Using `"classification"` causes `AssertionError: Found N Python objects not bound to checkpointed values`. Always use `"detection"` when starting from a COCO detection checkpoint.

2. **`_run_phase1_arch` swaps a global** — `run_phase0_probe` uses the module-level `PHASE0_RUNS_DIR` global to determine where to write outputs. The Phase-1 wrapper temporarily replaces it with `PHASE1_RUNS_DIR` inside a `try/finally` block.

3. **Runtime config filenames must be unique per (probe, steps, label)** — `adapt_schedule_to_steps` names them `{base}.steps_{N}.{label}.config`. If two runs share the same name, the second will overwrite the first's config on disk.

4. **`label_map_path` appears twice** in SSD/ResNet template configs (once for train, once for val) with the same placeholder. `make_ssd_resnet_config` handles this by using a two-step intermediate replacement.

5. **Screen runs are cache-aware** — The Phase-1 screen cell (`ke7eytd0mec`) reads `metrics.json` for each arch and skips runs with `"status": "ok"`. Failed runs are always re-run.

---

## SageMaker Integration

- `framework.py` provides `CustomFramework`, a minimal `sagemaker.estimator.Framework` subclass that accepts a custom Docker ECR image URI
- Hyperparameters are passed as `SM_HP_*` env vars; `run_training.sh` reads them with fallback to CLI args for local runs
- Data channels: `train` and `val` (S3 prefixes → `/opt/ml/input/data/{channel}/`)
- Model output: exported SavedModel is placed under `SM_MODEL_DIR/1/saved_model/` for SageMaker hosting

---

## Branches

- `main` — stable, upstream baseline
- `rtx3090-test` — active development: local GPU screening loop, Phase-0, Phase-1
