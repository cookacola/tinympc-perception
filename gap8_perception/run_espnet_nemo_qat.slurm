#!/usr/bin/env bash
#SBATCH --job-name=espnet_nemo_qat
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=a2r-main
#SBATCH --time=12:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/gap8-drone-racing-perception
WORKSPACE=/home/cchen/isaacsim-workspace/workspace
BRIDGE=${BRIDGE:-$WORKSPACE/espnet_dory_student_v2_deployment/bridge}
DATASET=$WORKSPACE/tinympc_event_aligned_dataset_20260816
OUTPUT=${OUTPUT:-$WORKSPACE/espnet_dory_student_v2_qat}
GATE_DATASET=$WORKSPACE/newbeedrone_multi_course_gate_v2_20k
GATE_SPLIT=$WORKSPACE/newbeedrone_multi_course_gate_v2_targets/split_shards.json
NO_GATE=$WORKSPACE/newbeedrone_multi_course_no_gate_v2_5k
PAIRED_NO_GATE=$WORKSPACE/newbeedrone_multi_course_paired_no_gate_v3_20k
mkdir -p "$OUTPUT"
cd "$REPO"
read -r -a graphs <<< "${GRAPHS:-encoder corner_head gate_head danger_head}"
for graph in "${graphs[@]}"; do
  representative=()
  if [[ "$graph" != danger_head ]]; then
    representative+=(
      --gate-dataset "$GATE_DATASET" --gate-split-file "$GATE_SPLIT"
      --no-gate-dataset "$NO_GATE"
      --paired-no-gate-dataset "$PAIRED_NO_GATE"
      --real-root /home/cchen/real_flight_data
    )
  fi
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
    /home/cchen/miniconda3/envs/frontnet/bin/python \
    gap8_perception/train_nemo_espnet_dory_qat.py \
    --graph "$graph" --bridge "$BRIDGE" --dataset "$DATASET" \
    --output "$OUTPUT" --epochs "${EPOCHS:-4}" \
    --train-limit "${TRAIN_LIMIT:-8192}" \
    --calibration-images "${CALIBRATION_IMAGES:-1024}" --batch-size 32 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" --learning-rate "${LEARNING_RATE:-1e-3}" \
    "${representative[@]}"
done
