#!/usr/bin/env bash
#SBATCH --job-name=gate_targets
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=a2r-main
#SBATCH --time=02:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env
SOURCE_DIR=${SOURCE_DIR:-/home/cchen/tinympc-gate-texture}
DATASET=${DATASET:-/home/cchen/isaacsim-workspace/workspace/newbeedrone_photo_v1_corners_20k}
TARGETS=${TARGETS:-/home/cchen/isaacsim-workspace/workspace/newbeedrone_photo_v1_corner_targets}
SPLIT_FILE=${SPLIT_FILE:-$TARGETS/split_shards.json}

echo "Job ID: $SLURM_JOB_ID node: $(hostname) conda: $CONDA_PREFIX python: $(which python)"
cd "$SOURCE_DIR"
mkdir -p "$TARGETS"
for shard in "$DATASET"/shard_*; do
  name=$(basename "$shard")
  output="$TARGETS/$name.npz"
  if [[ -e "$output" ]]; then echo "refusing to overwrite $output" >&2; exit 2; fi
  python -m gap8_perception.generate_targets "$shard" --output "$output"
done
python -m gap8_perception.make_split "$DATASET" --output "$SPLIT_FILE" \
  --train-count 16 --validation-count 2 --test-count 2
echo "Targets complete: $TARGETS"
