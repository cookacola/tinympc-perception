#!/usr/bin/env bash
#SBATCH --job-name=esp-gate-int-eval
#SBATCH --nodelist=a2r-main
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

repo=/home/cchen/gap8-drone-racing-perception
workspace=/home/cchen/isaacsim-workspace/workspace
integer_dir=${INTEGER_DIR:?set INTEGER_DIR to the NeMO integer graph directory}
training_summary=${TRAINING_SUMMARY:?set TRAINING_SUMMARY to the selected float summary}
output=${OUTPUT:?set OUTPUT to a new JSON report path}

cd "$repo"
source /home/cchen/miniconda3/etc/profile.d/conda.sh
conda activate /home/cchen/isaacsim-env
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "Job: $SLURM_JOB_ID node: $(hostname) conda: $CONDA_PREFIX python: $(which python)"

python gap8_perception/evaluate_espnet_dronet_gate_integer.py \
  --integer-dir "$integer_dir" \
  --nemo-report "$integer_dir/nemo_espnet_dronet_gate_report.json" \
  --training-summary "$training_summary" \
  --navigation-data /home/cchen/pulp-dronet/tiny-pulp-dronet-v3/dataset_original/Dataset_PULP_Dronet_v3 \
  --gate-dataset "$workspace/newbeedrone_multi_course_gate_v2_20k" \
  --gate-targets "$workspace/newbeedrone_multi_course_gate_v2_targets" \
  --gate-split-file "$workspace/newbeedrone_multi_course_gate_v2_targets/split_shards.json" \
  --no-gate-dataset "$workspace/newbeedrone_multi_course_no_gate_v2_5k" \
  --paired-no-gate-dataset "$workspace/newbeedrone_multi_course_paired_no_gate_v3_20k" \
  --real-root /home/cchen/real_flight_data \
  --output "$output" "${@}"
