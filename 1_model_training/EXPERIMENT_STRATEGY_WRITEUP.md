# Experiment Strategy for Cost-Efficient Object Detection Training

## 1. Objective and Constraints

The objective of this training strategy is to identify, before launching cost-incurring SageMaker training jobs, the most promising:

- object detection architecture family,
- data augmentation policy,
- and key hyperparameter settings.

The strategy is constrained by practical project realities:

- local training is performed on a single NVIDIA RTX 3090 GPU,
- dataset class coverage is limited and imbalanced,
- and iteration budget must prioritize fast elimination of weak candidates.

For this reason, local controlled experiment loops are treated as the primary decision mechanism, while SageMaker is reserved for final, higher-confidence training runs.

## 2. Dataset Characteristics Used to Drive Design

The current local TFRecord dataset profile (from `1_model_training/data/`) is:

- 84 training TFRecord shards
- 13 validation TFRecord shards
- 1,679 training frames
- 258 validation frames
- 36,136 training boxes
- 8,119 validation boxes

Class distribution is strongly imbalanced:

- `vehicle`: 27,907 boxes
- `pedestrian`: 8,015 boxes
- `cyclist`: 214 boxes

Object-scale distribution indicates strong small-object dominance and dense scenes:

- Small-object proxy share: 76.1% (train), 82.3% (val)
- Average boxes per image: 21.5 (train), 31.5 (val)

These characteristics favor a screening process that explicitly tests small-object sensitivity and recall behavior before committing to architecture-family expansion.

## 3. Phase-0 Objective

Phase-0 is a D1-centered screening stage whose objective is to identify, from local evidence, which architecture families should move forward to full comparison.

Phase-0 is designed to answer three questions:

- How sensitive performance is to scale and small-object handling.
- Whether dense-scene recall behavior indicates a need for heavier two-stage models.
- Whether lower-compute model families are still competitive enough to keep in scope.

## 4. Revised Phase-0: EfficientDet-D1 Probe Loop

Phase-0 is divided into four steps.

### 4.1 Phase-0.1: Dataset Profiling (No Training)

Compute dataset summary statistics from TFRecords and persist them as structured artifacts for experiment context and reporting.

### 4.2 Phase-0.2: Short D1 Probe Runs

Run short-budget EfficientDet-D1 experiments to test sensitivity to key configuration dimensions:

- input resolution probe (`512`, `640`, `768` variants),
- anchor density and anchor aspect-ratio probes,
- augmentation-strength probes (light vs. strong),
- focal-loss rebalance probe for class-imbalance stress testing.

These are intentionally short runs designed for directional signal, not final model quality.

### 4.3 Phase-0.3: Confirmation Runs

Select the top probe candidates and rerun them with a larger step budget to reduce ranking noise and verify consistency.

### 4.4 Phase-0.4: Family Recommendation

Convert probe outcomes into explicit architecture-family recommendations for Phase-1 using deterministic rules (documented below), rather than subjective interpretation.

### 4.5 Metrics Used in Phase-0

Primary ranking metric:

- validation `DetectionBoxes_Precision/mAP`

Supporting decision metrics:

- `DetectionBoxes_Precision/mAP (small)`
- `DetectionBoxes_Recall/AR@100`
- `Loss/total_loss`

## 5. Family Shortlisting Rules

Phase-0 outputs a deterministic family shortlist according to the following rules.

### 5.1 EfficientDet Family

Always include EfficientDet as the anchor family because Phase-0 probes are D1-based and provide direct evidence in this family.

### 5.2 SSD/RetinaNet-Style Family

Include this family when small-object sensitivity is strong under D1 probes, for example when either:

- `mAP_small(d1_res_768) - mAP_small(d1_res_512) >= 0.02`, or
- `mAP_small(d1_anchor_dense) - mAP_small(d1_baseline_640) >= 0.01`.

### 5.3 Faster R-CNN Family

Include this family when crowd-recall evidence suggests one-stage localization saturation, indicated by:

- `AR@100 - mAP >= 0.20` on the best confirmed D1 probe.

### 5.4 Mobile Family

Include mobile-family candidates only when low-resolution degradation is limited:

- `mAP(d1_res_512) >= mAP(d1_baseline_640) - 0.015`.

### 5.5 Fallback Rules

- If more than three families pass, keep the top three by evidence-strength score.
- If fewer than two families pass, force-add `EfficientDet` and `SSD/RetinaNet-style` to ensure minimum comparative coverage.

## 6. Phase-1: Architecture Selection Within Shortlisted Families

Phase-1 starts from the Phase-0 family shortlist and performs direct architecture-level comparison.

Core work in Phase-1:

- Define the candidate architecture set for each shortlisted family (for example, lightweight vs. standard variant within a family).
- Align each candidate to its matching TF2 Object Detection template config and pretrained checkpoint.
- Apply a standardized training/evaluation budget so model comparisons are fair.
- Run local training loops, collect validation metrics, and compare `mAP`, `mAP (small)`, `AR@100`, and stability of loss.
- Select a single winning architecture (or a top-two tie list if results are statistically close).

Phase-1 output:

- ranked architecture leaderboard,
- selected architecture for Phase-2.

## 7. Phase-2: Data Augmentation Policy Selection

Phase-2 fixes the architecture chosen in Phase-1 and searches augmentation policies.

Core work in Phase-2:

- Build a small augmentation policy library:
  - baseline/minimal policy,
  - geometry-focused policy (for scale/position robustness),
  - appearance-focused policy (color/quality disturbance),
  - mixed policy (geometry + appearance).
- Keep architecture and base optimizer settings fixed so augmentation impact is isolated.
- Run controlled local loops for each policy using equal step budgets.
- Evaluate policy impact primarily on validation `mAP`, with emphasis on `mAP (small)` and generalization stability.
- Select the augmentation policy that gives the best accuracy/robustness tradeoff without introducing unstable convergence.

Phase-2 output:

- augmentation policy leaderboard,
- selected augmentation policy for Phase-3.

## 8. Phase-3: Hyperparameter Optimization (Successive Halving)

Phase-3 fixes both architecture and augmentation from prior phases and tunes optimization settings.

Core work in Phase-3:

- Define a bounded hyperparameter search space (learning rate, warmup, momentum, weight decay, and batch size within GPU memory limits).
- Run a successive-halving schedule:
  - short initial runs for all candidates,
  - promote top performers to larger budgets,
  - repeat until final survivors remain.
- Track both quality and training behavior (`mAP`, `mAP (small)`, `AR@100`, `Loss/total_loss`, convergence smoothness).
- Perform one confirmation run on the final selected configuration.

Phase-3 output:

- final hyperparameter set,
- production-ready local training configuration.

The final artifact for SageMaker launch is:

- `1_model_training/experiments/final/final_pipeline.config`

This final config is then used as the launch input for SageMaker training.

## 9. Reproducibility and Cost Controls

To keep experiments reproducible and budget-aware:

- execution is Docker-only and aligned with the existing local training path,
- each phase uses fixed training budgets,
- run artifacts are recorded in a structured run registry,
- interrupted work can be resumed without repeating completed runs,
- and failed runs are explicitly marked and excluded from ranking unless rerun successfully.

These controls reduce unnecessary compute spend while preserving comparability across experiments.

## 10. Expected Outcome

This strategy is designed to produce:

- a defensible, data-driven architecture-family shortlist,
- a selected augmentation policy and tuned hyperparameters,
- and a final training config ready for SageMaker execution.

The expected net effect is lower cloud cost, better decision quality, and clearer technical justification in the project writeup.
