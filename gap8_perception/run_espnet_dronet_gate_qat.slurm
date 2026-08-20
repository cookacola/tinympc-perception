#!/usr/bin/env bash
#SBATCH --job-name=esp-gate-qat
#SBATCH --nodelist=a2r-main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

repo=/home/cchen/gap8-drone-racing-perception
workspace=/home/cchen/isaacsim-workspace/workspace
bridge=${BRIDGE:?set BRIDGE to the selected float bridge}
output=${OUTPUT:?set OUTPUT to a new QAT directory}
dataset=${DATASET:-$workspace/tinympc_event_aligned_dataset_20260816}

mkdir -p "$output"
cd "$repo"
source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate frontnet
echo "Job: $SLURM_JOB_ID node: $(hostname) conda: $CONDA_PREFIX python: $(which python)"

read -r -a graphs <<< "${GRAPHS:-encoder corner_head gate_head presence_head navigation_head}"
canonical_args=()
if [[ "${CANONICALIZE_BATCHNORM:-0}" == 1 ]]; then
  canonical_args+=(--canonicalize-batchnorm)
fi
if [[ "${CONDITION_ON_INTEGER_ENCODER:-0}" == 1 ]]; then
  canonical_args+=(--condition-on-integer-encoder)
fi
for graph in "${graphs[@]}"; do
  PYTHONPATH=. CUDA_VISIBLE_DEVICES='' python \
    gap8_perception/train_nemo_espnet_dronet_gate_qat.py \
    --graph "$graph" --bridge "$bridge" --dataset "$dataset" --output "$output" \
    --epochs "${EPOCHS:-4}" --train-limit "${TRAIN_LIMIT:-8192}" \
    --calibration-images "${CALIBRATION_IMAGES:-1024}" --batch-size 32 \
    --workers "$SLURM_CPUS_PER_TASK" --learning-rate "${LEARNING_RATE:-1e-3}" \
    "${canonical_args[@]}" \
    --gate-dataset "$workspace/newbeedrone_multi_course_gate_v2_20k" \
    --gate-split-file "$workspace/newbeedrone_multi_course_gate_v2_targets/split_shards.json" \
    --no-gate-dataset "$workspace/newbeedrone_multi_course_no_gate_v2_5k" \
    --paired-no-gate-dataset "$workspace/newbeedrone_multi_course_paired_no_gate_v3_20k" \
    --real-root /home/cchen/real_flight_data
done
