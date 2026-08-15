#!/usr/bin/env bash
# Model-owned cluster settings. This file is the only SLURM file expected to need routine edits.
PARTITION_CPU="cpu"
PARTITION_GPU="gpu"
TIME_CPU="01:00:00"
TIME_GPU="04:00:00"
CPUS="4"
MEMORY="16G"
GPUS="1"
RUN_ROOT="${RUN_ROOT:-runs}"

# Print the implementation-specific key=value arguments for one command. Keep secrets out of this file;
# obtain them from the submit environment. Paths below are examples and must be finalized with tests/e2e.py.
stage_args() {
  case "$1" in
    preprocess_data) printf '%s\n' "external_meds_dir=${MEDS_ROOT:?set MEDS_ROOT}" "output_data_dir=${RUN_DIR:?set RUN_DIR}/data" ;;
    pretrain) printf '%s\n' "input_data_dir=$RUN_DIR/data" "output_pretrained_model_dir=$RUN_DIR/pretrained" ;;
    infer) printf '%s\n' "input_data_dir=$RUN_DIR/data" "external_labels_dir=${LABELS_DIR:?set LABELS_DIR}" "output_inference_subdir=inference/default" ;;
    supervised_train) printf '%s\n' "input_data_dir=$RUN_DIR/data" "external_labels_dir=${LABELS_DIR:?set LABELS_DIR}" "output_supervised_model_dir=$RUN_DIR/model" ;;
    predict) printf '%s\n' "external_labels_dir=${LABELS_DIR:?set LABELS_DIR}" "output_predictions_dir=$RUN_DIR/predict" ;;
    *) return 2 ;;
  esac
}
