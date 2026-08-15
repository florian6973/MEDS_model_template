#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi

mapfile -t commands < <(meds-model commands)
if [[ -z "${commands[0]:-}" ]]; then
  echo "No commands declared" >&2
  exit 1
fi

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${GITHUB_SHA:-local}}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$run_id/artifacts}"
export RUN_DIR
dependency=""
for command in "${commands[@]}"; do
  args_dir="$RUN_ROOT/$run_id/command-args"
  args_file="$args_dir/$command.args"
  if ! $dry_run; then
    mkdir -p "$args_dir"
    stage_args "$command" > "$args_file"
  else
    args_file="$RUN_ROOT/$run_id/command-args/$command.args"
  fi
  partition="$PARTITION_GPU"
  time_limit="$TIME_GPU"
  gres=(--gres="gpu:${GPUS}")
  if [[ "$command" == "preprocess_data" ]]; then
    partition="$PARTITION_CPU"
    time_limit="$TIME_CPU"
    gres=()
  fi
  submit=(sbatch --parsable --partition="$partition" --time="$time_limit" --cpus-per-task="$CPUS" --mem="$MEMORY")
  if [[ -n "$dependency" ]]; then
    submit+=(--dependency="afterok:$dependency")
  fi
  submit+=("${gres[@]}" --export="ALL,MEDS_MODEL_COMMAND=$command,MEDS_MODEL_ARGS_FILE=$args_file,RUN_ID=$run_id" "$SCRIPT_DIR/job.sbatch")
  if $dry_run; then
    printf '%q ' "${submit[@]}"
    printf '\n'
    dependency="DRY_RUN_${command}"
  else
    dependency="$("${submit[@]}")"
    echo "$command: $dependency"
  fi
done
