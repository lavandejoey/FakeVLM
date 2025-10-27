#!/bin/bash
#SBATCH --job-name=FakeVLMEval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=H100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# -------- shell hygiene --------
# Exit immediately if a command exits with a non-zero status.
set -euo pipefail
# Enable debugging output
#set -x
umask 077
mkdir -p logs

# -------- print job header --------
echo "================= SLURM JOB START ================="
echo "Job:    $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node:   ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-unknown}  (${SLURM_JOB_GPUS:-not-set})"
echo "Start:  $(date)"
echo "==================================================="

datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_fakeVLM"
mkdir -p "${result_dir}"
#data_root="/home/infres/ziyliu-24/data/FakeParts2DataMock"
data_root="/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"
model_path="llava-hf/llava-1.5-7b-hf" # "llava-hf/llava-1.5-7b-hf", "lingcco/fakeVLM"
#model_path="lingcco/fakeVLM"
data_entry_csv="/projects/hi-paris/DeepFakeDataset/frames_index.csv"
done_csv_list=("results")

source /home/infres/ziyliu-24/miniconda3/etc/profile.d/conda.sh
conda activate fakevlm310

srun python3 -Wignore FakeVLMEval.py \
--data_root "${data_root}" \
--model_path "${model_path}" \
--pred_csv "${result_dir}/predictions.csv" \
--data_csv ${data_entry_csv} \
--done_csv_list "${done_csv_list[@]}"

#N_GPUS=${SLURM_GPUS_ON_NODE:-1}
#VAL_BATCH_SIZE=$((12 * N_GPUS))
#torchrun --standalone --nproc_per_node="${N_GPUS}" FakeVLMEval.py \
#--data_root "${data_root}" \
#--model_path "${model_path}" \
#--pred_csv "${result_dir}/predictions.csv" \
#--data_csv ${data_entry_csv} \
#--val_batch_size "${VAL_BATCH_SIZE}"

EXIT_CODE=$?

echo "================== SLURM JOB END =================="
echo "End:   $(date)"
echo "Exit:  ${EXIT_CODE}"
echo "==================================================="
exit "${EXIT_CODE}"
