#!/usr/bin/env bash
#SBATCH --job-name=gap8_sequential_student
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
cd "$REPO"
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m gap8_perception.profile_sequential \
  --output "${PROFILE_OUTPUT:-/home/cchen/isaacsim-workspace/workspace/gap8_sequential_student_v2/architecture_report.json}"

if [[ "${RUN_TRAIN:-0}" == 1 ]]; then
  if [[ "${CORPUS_AUDIT:-0}" == 1 ]]; then
    PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m gap8_perception.audit_75k_corpus \
      --dataset "${DATASET:?set DATASET}" \
      --targets "${TARGETS:?set TARGETS}" \
      --split-file "${SPLIT_FILE:-gap8_perception/configs/split_shards_75k.json}" \
      --output "${CORPUS_AUDIT_OUTPUT:?set CORPUS_AUDIT_OUTPUT}"
  fi
  TRAIN_ARGS=()
  if [[ -n "${TRAIN_LIMIT:-}" ]]; then
    TRAIN_ARGS+=(--train-limit "$TRAIN_LIMIT")
  fi
  if [[ -n "${VAL_LIMIT:-}" ]]; then
    TRAIN_ARGS+=(--val-limit "$VAL_LIMIT")
  fi
  TRAIN_ARGS+=(--real-root "${REAL_ROOT:-/home/cchen/real_flight_data}")
  TRAIN_ARGS+=(--real-train-flights "${REAL_TRAIN_FLIGHTS:-flight_06,flight_07}")
  TRAIN_ARGS+=(--real-weight "${REAL_WEIGHT:-0.35}")
  PYTHONPATH=. /home/cchen/isaacsim-env/bin/python \
    -m gap8_perception.train_sequential \
    --dataset "${DATASET:?set DATASET}" \
    --targets "${TARGETS:?set TARGETS}" \
    --split-file "${SPLIT_FILE:-gap8_perception/configs/split_shards_75k.json}" \
    --output "${TRAIN_OUTPUT:?set TRAIN_OUTPUT}" \
    --epochs "${EPOCHS:-80}" --batch-size "${BATCH_SIZE:-32}" --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --observation-dropout "${OBSERVATION_DROPOUT:-0.10}" \
    --sensor-augmentation "${SENSOR_AUGMENTATION:-1.0}" \
    "${TRAIN_ARGS[@]}"
fi
