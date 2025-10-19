#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -euo pipefail
# Enable debugging output
#set -x

export CUDA_VISIBLE_DEVICES=0
datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_fakeVLM"
mkdir -p "${result_dir}"
#data_root="/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"
data_root="/home/infres/ziyliu-24/data/FakeParts2DataMock"
model_path="llava-hf/llava-1.5-7b-hf" # "llava-hf/llava-1.5-7b-hf", "lingcco/fakeVLM"
#model_path="lingcco/fakeVLM"

source /home/infres/ziyliu-24/miniconda3/etc/profile.d/conda.sh
conda activate fakevlm310
module load cuda/12.4.1 || module load cuda/12.1

python3 "FakeVlmEval.py" \
    --data_root "${data_root}" \
    --model_path "${model_path}" \
    --pred_csv "${result_dir}/predictions.csv"
