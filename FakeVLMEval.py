"""
llava-hf/llava-1.5-7b-hf:
 - A100 0.3s / frame
lingcco/fakeVLM:
 - A100 12s / frame

Usage:
datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_fakeVLM"
data_root="/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"
model_path="llava-hf/llava-1.5-7b-hf" # "llava-hf/llava-1.5-7b-hf", "lingcco/fakeVLM"
data_entry_csv="/projects/hi-paris/DeepFakeDataset/frames_index.csv"

conda activate fakevlm310

srun python3 -Wignore "FakeVlmEval.py" \
    --data_root "${data_root}" \
    --model_path "${model_path}" \
    --pred_csv "${result_dir}/predictions.csv" \
    --data_csv ${data_entry_csv}
"""
import argparse
import logging
import pandas as pd
import random
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from DataUtils import standardise_predictions, FakePartsV2DatasetBase

# Log config for streaming outputs to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="FakeVlmEval")
    # keep flags minimal
    parser.add_argument("--model_path", default="llava-hf/llava-1.5-7b-hf", type=str)
    parser.add_argument("--val_batch_size", default=50, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--data_root", default="/home/infres/ziyliu-24/data/FakeParts2DataMock", type=str,
                        help="Dataset root; if empty, use DataUtils default")
    parser.add_argument('--data_csv', type=str, default=None, help='csv file indexing the dataset')
    parser.add_argument("--done_csv_list", type=str, nargs='*', default=[], help="List of done CSVs to skip samples")
    parser.add_argument("--pred_csv", default="results/predictions.csv", type=str)
    parser.add_argument("--mode", default="frame", choices=["frame", "video"])
    return parser.parse_args()


def _ensure_llava_processing_args(processor, default_patch=14, default_strategy="patch"):
    """
    Minimal fix for the deprecation: attach the two attributes to the processor
    if they are missing in the processing config. Defaults suit LLaVA-1.5 (ViT-L/14).
    """
    if not hasattr(processor, "patch_size"):
        processor.patch_size = default_patch
    if not hasattr(processor, "vision_feature_select_strategy"):
        processor.vision_feature_select_strategy = default_strategy
    return processor


class FakePartsV2Dataset(FakePartsV2DatasetBase):
    def __init__(self, args, processor, **kwargs, ):
        # Wire CLI args into the base dataset (no silent fallbacks).
        super().__init__(
            data_root=args.data_root,
            mode=args.mode,
            csv_path=args.data_csv,
            model_name=args.model_path,
            done_csv_list=args.done_csv_list,
            **kwargs,
        )
        self.args = args
        self.processor = processor
        # Simple prompt words (keep the conversation template unchanged)
        self.prompt_text = "Is this image real or fake? Answer with 'real' or 'fake' only."

    def __getitem__(self, idx):
        image, label, meta = super().__getitem__(idx)
        chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt_text}, ], }]
        prompt = self.processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True,
        )
        return inputs, meta


def _collate_inputs_and_meta(batch):
    """
    Keep metas as a list of dicts (one per sample) to avoid default_collate
    turning them into dict of lists/tensors.
    Also stack input tensors across the batch dimension.
    """
    # Filter Nones defensively (mirrors collate_skip_none behaviour)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    inputs_list, metas = zip(*batch)  # each inputs is a dict of tensors with a leading [1, ...] dim
    keys = inputs_list[0].keys()
    stacked = {}
    for k in keys:
        # each inputs[k] is shape [1, ...]; cat along dim=0 -> [B, ...]
        stacked[k] = torch.cat([inp[k] for inp in inputs_list], dim=0)
    return stacked, list(metas)


def validate(args, model, cls_test_dataloader, processor):
    processor = _ensure_llava_processing_args(processor)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []  # Used to build DataUtils.required schema
    # Ready for csv
    out_path = Path(args.pred_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for collated in tqdm(cls_test_dataloader, desc="Evaluating"):
            if collated is None:
                continue
            inputs, metas = collated
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Force to CPU to avoid CUDA OOM
            # inputs = {k: v for k, v in inputs.items()}  # Keep on CPU to avoid CUDA OOM
            output = model.generate(**inputs, max_new_tokens=32)
            # Single sample batch (val_batch_size can be >1, but the current data pipeline processes it one at a time)
            for i in range(output.shape[0]):
                response = processor.decode(output[i], skip_special_tokens=True).split('?')[-1]
                # log.info(response)
                if "ASSISTANT: Real" in response:
                    pred = 0
                elif "ASSISTANT: Fake" in response:
                    pred = 1
                # compare the count of 'real' and 'fake' in the response
                elif response.lower().count('real') > response.lower().count('fake'):
                    pred = 0  # real
                elif response.lower().count('fake') > response.lower().count('real'):
                    pred = 1  # fake
                else:
                    log.info(f"no explicit 'fake' or 'real' in response: {response}")
                    pred = random.choice([0, 1])
                # Simple score (use 0/1 as score when there is no confidence); can be replaced by more detailed analysis
                score = float(pred)
                meta_i = metas[i]
                # meta_i fields come from DataUtils._make_meta(): use 'sample_id' (not 'rel_path')
                label = int(meta_i["label"])
                rows.append({
                    "sample_id": str(meta_i["sample_id"]),
                    "task": str(meta_i["task"]),
                    "method": str(meta_i["method"]),
                    "subset": str(meta_i["subset"]),
                    "label": label,
                    "model": str(args.model_path),
                    "mode": str(meta_i["mode"]),
                    "score": score,
                    "pred": int(pred),
                })
                # Append to save batch row into csv
                df = pd.DataFrame(rows)
                df = standardise_predictions(df)
                df.to_csv(args.pred_csv, mode="a", header=not out_path.exists(), index=False)
                rows = []  # reset for next batch
    log.info(f"Saved predictions to {args.pred_csv}")


def load_model(args):
    log.info("Loading model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
        revision='a272c74' if 'llava-hf/llava-1.5-7b-hf' in args.model_path else None,
        # load_in_8bit=True,
        # load_in_4bit=True,
        # quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    ).eval()
    log.info(f"Successfully loaded model from: {args.model_path}")
    return model


def main():
    args = parse_args()
    model = load_model(args)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')
    processor = _ensure_llava_processing_args(processor)
    cls_test_dataset = FakePartsV2Dataset(
        args=args,
        processor=processor,
    )
    cls_test_dataloader = DataLoader(
        cls_test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=_collate_inputs_and_meta,
    )
    validate(args, model, cls_test_dataloader, processor)


def validate_parallel(args, model, cls_test_dataloader, processor, device, local_rank):
    processor = _ensure_llava_processing_args(processor)
    rows = []  # Used to build DataUtils.required schema

    # --- DDP Output File Fix ---
    out_path = Path(args.pred_csv)
    # Make the output path unique for each process
    out_path = out_path.parent / f"{out_path.stem}_rank{local_rank}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # --- DDP Output File Fix End ---

    with torch.no_grad():
        for collated in tqdm(cls_test_dataloader, desc=f"Evaluating Rank {local_rank}"):  # Updated desc
            if collated is None:
                continue
            inputs, metas = collated
            inputs = {k: v.to(device) for k, v in inputs.items()}  # This now uses the correct DDP device

            # model.generate() works fine with the DDP-wrapped model
            output = model.generate(**inputs, max_new_tokens=32)
            # Single sample batch (val_batch_size can be >1, but the current data pipeline processes it one at a time)
            for i in range(output.shape[0]):
                response = processor.decode(output[i], skip_special_tokens=True).split('?')[-1]
                # log.info(response)
                if "ASSISTANT: Real" in response:
                    pred = 0
                elif "ASSISTANT: Fake" in response:
                    pred = 1
                # compare the count of 'real' and 'fake' in the response
                elif response.lower().count('real') > response.lower().count('fake'):
                    pred = 0  # real
                elif response.lower().count('fake') > response.lower().count('real'):
                    pred = 1  # fake
                else:
                    log.info(f"no explicit 'fake' or 'real' in response: {response}")
                    pred = random.choice([0, 1])
                # Simple score (use 0/1 as score when there is no confidence); can be replaced by more detailed analysis
                score = float(pred)
                meta_i = metas[i]
                # meta_i fields come from DataUtils._make_meta(): use 'sample_id' (not 'rel_path')
                label = int(meta_i["label"])
                rows.append({
                    "sample_id": str(meta_i["sample_id"]),
                    "task": str(meta_i["task"]),
                    "method": str(meta_i["method"]),
                    "subset": str(meta_i["subset"]),
                    "label": label,
                    "model": str(args.model_path),
                    "mode": str(meta_i["mode"]),
                    "score": score,
                    "pred": int(pred),
                })
                # Append to save batch row into csv
                df = pd.DataFrame(rows)
                df = standardise_predictions(df)
                # This now writes to the rank-specific file
                df.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
                rows = []  # reset for next batch

    log.info(f"Saved predictions for rank {local_rank} to {out_path}")


def main_parallel():
    args = parse_args()
    # --- DDP Setup Start ---
    # Initialize the distributed process group
    dist.init_process_group(backend="nccl")
    # Get the GPU ID for the current process
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    log.info(f"Initialized DDP on rank {local_rank} on device {device}")
    # --- DDP Setup End ---

    model = load_model(args)
    # --- DDP Model Setup ---
    # Move model to the process-specific GPU
    model = model.to(device)
    # Wrap the model in DDP
    model = DDP(model, device_ids=[local_rank])
    # --- DDP Model Setup End ---

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')
    processor = _ensure_llava_processing_args(processor)
    cls_test_dataset = FakePartsV2Dataset(
        args=args,
        processor=processor,
    )

    # --- DDP Sampler Setup ---
    # Create a sampler to give each process its own slice of data
    cls_test_sampler = DistributedSampler(cls_test_dataset, shuffle=False)
    # --- DDP Sampler Setup End ---

    cls_test_dataloader = DataLoader(
        cls_test_dataset,
        batch_size=args.val_batch_size,
        # shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=_collate_inputs_and_meta,
        sampler=cls_test_sampler
    )

    # Pass the device and local_rank to validate
    validate_parallel(args, model, cls_test_dataloader, processor, device, local_rank)


if __name__ == "__main__":
    main()
    # main_parallel()
