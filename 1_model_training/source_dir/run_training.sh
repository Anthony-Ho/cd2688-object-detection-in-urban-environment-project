#!/bin/bash

MODEL_DIR=${SM_HP_MODEL_DIR}
PIPELINE_CONFIG_PATH=${SM_HP_PIPELINE_CONFIG_PATH}
NUM_TRAIN_STEPS=${SM_HP_NUM_TRAIN_STEPS}
SAMPLE_1_OF_N_EVAL_EXAMPLES=${SM_HP_SAMPLE_1_OF_N_EVAL_EXAMPLES}

# Parse command line arguments to support local execution
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_dir) MODEL_DIR="$2"; shift ;;
        --pipeline_config_path) PIPELINE_CONFIG_PATH="$2"; shift ;;
        --num_train_steps) NUM_TRAIN_STEPS="$2"; shift ;;
        --sample_1_of_n_eval_examples) SAMPLE_1_OF_N_EVAL_EXAMPLES="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; shift ;;
    esac
    shift
done

# Resolve fine_tune checkpoint from pipeline.config and verify it exists
FINE_TUNE_CKPT=$(python - "${PIPELINE_CONFIG_PATH:-pipeline.config}" <<'PY'
import os, re, sys
path = sys.argv[1]
if not os.path.exists(path):
    print("")
    sys.exit(0)
with open(path) as f:
    for line in f:
        m = re.match(r'\s*fine_tune_checkpoint:\s*"([^"]+)"', line)
        if m:
            ckpt = m.group(1)
            if not os.path.isabs(ckpt):
                ckpt = os.path.normpath(os.path.join(os.path.dirname(path), ckpt))
            print(ckpt)
            sys.exit(0)
print("")
PY
)

if [ -n "${FINE_TUNE_CKPT}" ]; then
  if [ ! -f "${FINE_TUNE_CKPT}.index" ]; then
    echo "ERROR: fine_tune_checkpoint not found: ${FINE_TUNE_CKPT}(.index/.data)" >&2
    exit 1
  fi
  echo "Using fine_tune_checkpoint: ${FINE_TUNE_CKPT}"
fi

if [ "${SM_NUM_GPUS:-0}" -gt 0 ]
then
   NUM_WORKERS=${SM_NUM_GPUS}
else
   NUM_WORKERS=1
fi

echo "===TRAINING THE MODEL=="
python model_main_tf2.py \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --model_dir ${MODEL_DIR} \
    --num_train_steps ${NUM_TRAIN_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --sample_1_of_n_eval_examples ${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
    --alsologtostderr

echo "==EVALUATING THE MODEL=="
python model_main_tf2.py \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --model_dir ${MODEL_DIR} \
    --checkpoint_dir ${MODEL_DIR} \
    --eval_timeout 10

echo "==EXPORTING THE MODEL=="
EXPORT_BASE_DIR="${SM_MODEL_DIR:-${MODEL_DIR}}"
EXPORT_DIR="${EXPORT_BASE_DIR}/exported"
python exporter_main_v2.py \
    --trained_checkpoint_dir ${MODEL_DIR} \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --output_directory ${EXPORT_DIR}

if [ -n "${SM_MODEL_DIR}" ]; then
  mkdir -p "${SM_MODEL_DIR}/1"
  mv "${EXPORT_DIR}/saved_model" "${SM_MODEL_DIR}/1"
else
  mv "${EXPORT_DIR}/saved_model" "${MODEL_DIR}/saved_model"
fi
