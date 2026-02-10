# Training Notebook Walkthrough (`1_train_model.ipynb`)

This guide explains the intent of each section of the training notebook and what to verify before moving on.

## 1) Environment setup and imports

The notebook first installs core dependencies (`tensorflow_io`, `sagemaker<3`, `numpy<2.0`, `python-dotenv`, `awscli`) and then validates imports for SageMaker and the local `CustomFramework` class from `framework.py`.

**What to check:**
- You are running in the expected Python environment.
- `sagemaker` imports successfully.
- `framework.py` is on your path and imports cleanly.

## 2) AWS role and region configuration

The next cells load `.env`, set `AWS_DEFAULT_REGION` if missing, and determine the execution role from `SAGEMAKER_ROLE_ARN` (with fallback to `sagemaker.get_execution_role()`).

**What to check:**
- `role` prints as a valid IAM role ARN.
- Region is set correctly for your account resources.

## 3) Input data and TensorBoard output location

The notebook defines S3 input channels for train/val and a TensorBoard S3 prefix.

**What to check:**
- S3 train/val buckets are readable.
- TensorBoard output bucket/prefix is writable.

## 4) Build training container inputs

It clones TensorFlow Model Garden into `docker/models` and copies `model_main_tf2.py` and `exporter_main_v2.py` into `source_dir`.

**What to check:**
- `source_dir/model_main_tf2.py` and `source_dir/exporter_main_v2.py` exist after running.
- If rerunning, remove stale clone content if needed.

## 5) Build (and optionally push) Docker image

The notebook builds `tf2-object-detection` from `docker/Dockerfile.rocm`. It then determines the image name/URI and runs a quick container import/GPU visibility check.

**What to check:**
- Docker build completes successfully.
- Test container run reports expected GPU visibility.
- Object Detection API imports in-container.

## 6) Download pretrained checkpoint

The notebook downloads an EfficientDet-D1 checkpoint tarball and extracts checkpoint files into `source_dir/checkpoint`.

**What to check:**
- `source_dir/checkpoint/ckpt-0.*` exists.
- `pipeline.config` points to the checkpoint path and expected dataset/label settings.

## 7) Manual local GPU dry run (recommended)

Before launching SageMaker training, the notebook downloads train/val data locally and runs `run_training.sh` inside Docker with mounted volumes.

**What to check:**
- Data sync from S3 succeeds.
- `local_training_output/` receives checkpoints/events.
- This phase helps catch config errors early.

## 8) Launch SageMaker training job

The notebook configures SageMaker TensorBoard output, constructs `CustomFramework`, and calls `fit(inputs)` with the train/val channels.

Key job parameters in this starter version:
- `entry_point='run_training.sh'`
- `source_dir='source_dir/'`
- `instance_type='local'`
- hyperparameters for model dir, pipeline config, train steps, eval sampling

**What to check:**
- Job launches and enters training.
- TensorBoard logs appear at the configured S3 prefix.
- Output model artifacts are produced on completion.

## 9) Iteration loop

The final section prompts you to improve quality by editing `pipeline.config`, increasing train steps, trying different pretrained models, and rerunning experiments.

---

## Practical newcomer tips

1. Keep your first run tiny (low train steps) to verify end-to-end wiring.
2. Treat `run_training.sh` as the single source of truth for train/eval/export behavior.
3. Version your `pipeline.config` edits so you can compare experiments reliably.
4. Only switch to non-local instance types after local flow is stable.
