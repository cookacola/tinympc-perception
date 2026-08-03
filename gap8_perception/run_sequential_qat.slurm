#!/usr/bin/env bash
#SBATCH --job-name=gap8_sequential_qat
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

: "${FLOAT_CHECKPOINT:?set FLOAT_CHECKPOINT}"
: "${OUTPUT:?set OUTPUT}"
: "${DATASET:?set DATASET to the canonical 160x160 source corpus}"
: "${TARGETS:?set TARGETS to regenerated sequential_fixed_normal_v1 archives}"
REPO=/home/cchen/gap8-drone-racing-perception
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
cd "$REPO"
QAT_ARGS=(
  --real-root "${REAL_ROOT:-/home/cchen/real_flight_data}"
  --real-train-flights "${REAL_TRAIN_FLIGHTS:-flight_06,flight_07}"
  --real-weight "${REAL_WEIGHT:-0.35}"
  --seed "${SEED:-2026}"
)
PYTHONPATH=. /home/cchen/isaacsim-env/bin/python -m gap8_perception.train_sequential_qat \
  --dataset "$DATASET" \
  --targets "$TARGETS" \
  --split-file "${SPLIT_FILE:-gap8_perception/configs/split_shards_75k.json}" \
  --float-checkpoint "$FLOAT_CHECKPOINT" --output "$OUTPUT" \
  --epochs "${EPOCHS:-12}" --batch-size "${BATCH_SIZE:-64}" --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --observation-dropout "${OBSERVATION_DROPOUT:-0.10}" \
  --sensor-augmentation "${SENSOR_AUGMENTATION:-1.0}" \
  "${QAT_ARGS[@]}"
