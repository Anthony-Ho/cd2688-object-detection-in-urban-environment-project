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

## 6. Phase-0 Findings and Conclusions

All 10 screening runs and 6 confirmation runs completed successfully. Artifacts are recorded in `1_model_training/experiments/phase0_summary/`. Screening runs used 3,500 steps; confirmation runs used 10,000 steps with 2 independent repeats per probe. The composite score used for screening ranking was `score = 0.7 × mAP_small + 0.3 × mAP`, weighted toward small-object performance given the dataset profile.

### 6.1 Screening Results (3,500 steps, all 10 probes)

| Probe | mAP | mAP (small) | AR@100 | Score |
|---|---|---|---|---|
| `d1_anchor_scale_2p0_dense` | 0.1456 | 0.0872 | 0.2017 | 0.1048 |
| `d1_anchor_scale_2p5` | 0.1409 | 0.0805 | 0.1925 | 0.0986 |
| `d1_res_768` | 0.1292 | 0.0659 | 0.1804 | 0.0849 |
| `d1_aug_light` | 0.1332 | 0.0566 | 0.1809 | 0.0796 |
| `d1_aug_mild_color` | 0.1290 | 0.0538 | 0.1807 | 0.0763 |
| `d1_anchor_wide_ar` | 0.1267 | 0.0559 | 0.1764 | 0.0772 |
| `d1_anchor_dense` | 0.1257 | 0.0533 | 0.1760 | 0.0750 |
| `d1_focal_rebalanced` | 0.1033 | 0.0374 | 0.1550 | 0.0572 |
| `d1_res_512` | 0.0980 | 0.0243 | 0.1438 | 0.0464 |
| `d1_baseline_640` | 0.0843 | 0.0266 | 0.1325 | 0.0439 |

The top three probes (`d1_anchor_scale_2p0_dense`, `d1_anchor_scale_2p5`, `d1_res_768`) were promoted to confirmation. Compared to the baseline, the best screen probe improved mAP by +0.061, mAP (small) by +0.061, and AR@100 by +0.069.

### 6.2 Confirmation Results (10,000 steps, 2 runs each)

| Probe | mAP (mean ± std) | mAP small (mean ± std) | AR@100 (mean) | Score (mean ± std) |
|---|---|---|---|---|
| `d1_res_768` | **0.1641 ± 0.0016** | 0.0872 ± 0.0040 | **0.2130** | 0.1103 ± 0.0033 |
| `d1_anchor_scale_2p0_dense` | 0.1570 ± 0.0016 | **0.0953 ± 0.0020** | 0.2123 | **0.1138 ± 0.0019** |
| `d1_anchor_scale_2p5` | 0.1569 ± 0.0001 | 0.0897 ± 0.0006 | 0.2099 | 0.1099 ± 0.0004 |

All six confirmation runs completed without failure. Variance across repeats was low for all three probes, confirming the screening ranking was not noise-driven.

### 6.3 Key Findings

**Resolution matters more than anchor tuning for overall mAP.** `d1_res_768` leads on mAP (0.164) and AR@100 (0.213) and has the lowest eval loss (0.290). The higher resolution allows the model to resolve small objects that 640-input crops miss, which contributes to better recall.

**Anchor scale matters more for small-object mAP.** `d1_anchor_scale_2p0_dense` leads on mAP (small) (0.095) and composite score (0.114). Given that 76–82% of boxes in this dataset are small, this is a meaningful advantage for the primary failure mode.

**`d1_anchor_scale_2p5` adds no value over `d1_anchor_scale_2p0_dense`.** It is weaker on every metric while nearly identical in compute cost. It is excluded from Phase-1.

**Augmentation probes (`d1_aug_light`, `d1_aug_mild_color`) underperformed.** Both scored below resolution and anchor probes, suggesting the pretrained backbone generalises adequately with standard augmentation at short budgets. Augmentation tuning is deferred to Phase-2 where it is isolated properly.

**Focal loss rebalancing hurt at short budgets.** `d1_focal_rebalanced` ranked second-to-last on both mAP and score. The modified focal parameters likely destabilise early gradient flow at 3,500 steps. This probe is not promoted.

**Classification loss dominates eval loss across all top probes.** Across confirmation runs, classification loss accounts for approximately 85–87% of total eval loss, while localization loss accounts for 4–5%. This indicates that class prediction — not box regression — is the primary bottleneck, and that improving class discrimination (via augmentation, label smoothing, or better architecture) is the highest-value direction for Phase-2 and Phase-3.

**Faster R-CNN trigger did not activate.** The best confirmed `AR@100 - mAP` gap was approximately 0.049–0.055, well below the 0.20 threshold. Two-stage models are not justified by the current evidence.

**Mobile family is excluded.** `d1_res_512` scored 0.046 vs 0.044 for `d1_baseline_640`, a difference of only 0.002, which is below the 0.015 tolerance threshold in Rule 5.4. Low-resolution degradation is too large to justify mobile-family inclusion.

### 6.4 Conclusion and Family Recommendation

Phase-0 produces the following deterministic family shortlist for Phase-1:

- **EfficientDet** — always included; direct D1 evidence available.
- **SSD/RetinaNet-style** — included because `mAP_small(d1_anchor_dense) - mAP_small(d1_baseline_640) = 0.0267 >= 0.01`, triggering Rule 5.2.

The two configurations carried forward into Phase-1 are:

- `d1_res_768` — strongest overall mAP and AR@100; preferred for general accuracy.
- `d1_anchor_scale_2p0_dense` — strongest mAP (small) and composite score; preferred for the small-object-dominated evaluation regime of this dataset.

These two configurations will serve as the EfficientDet representatives in Phase-1 architecture comparison.

## 7. Phase-1: Architecture Selection Within Shortlisted Families

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

## 8. Phase-2: Data Augmentation Policy Selection

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

## 9. Phase-3: Hyperparameter Optimization (Successive Halving)

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

## 10. Reproducibility and Cost Controls

To keep experiments reproducible and budget-aware:

- execution is Docker-only and aligned with the existing local training path,
- each phase uses fixed training budgets,
- run artifacts are recorded in a structured run registry,
- interrupted work can be resumed without repeating completed runs,
- and failed runs are explicitly marked and excluded from ranking unless rerun successfully.

These controls reduce unnecessary compute spend while preserving comparability across experiments.

## 11. Expected Outcome

This strategy is designed to produce:

- a defensible, data-driven architecture-family shortlist,
- a selected augmentation policy and tuned hyperparameters,
- and a final training config ready for SageMaker execution.

The expected net effect is lower cloud cost, better decision quality, and clearer technical justification in the project writeup.
