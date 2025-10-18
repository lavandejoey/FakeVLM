"""
llava-hf/llava-1.5-7b-hf:
 - A100 0.3s / frame
lingcco/fakeVLM:
 - A100 12s / frame

Usage:
conda activate fakevlm310
python3 FakeVlmEval.py
"""
import argparse
import logging
import os
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from DataUtils import (
    index_dataframe,
    standardise_predictions,
    quick_report,
)

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
    parser = argparse.ArgumentParser(description="FakeVlmEval (no-json, DataUtils-based)")
    # keep flags minimal
    parser.add_argument("--model_path", default="llava-hf/llava-1.5-7b-hf", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=os.cpu_count(), type=int)
    parser.add_argument("--data_root", default="/home/infres/ziyliu-24/data/FakeParts2DataMock", type=str,
                        help="Dataset root; if empty, use DataUtils default")
    parser.add_argument("--pred_csv", default="results/predictions.csv", type=str)
    parser.add_argument("--report_json", default="results/report.json", type=str)
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


class legion_cls_dataset(Dataset):
    """
    用 DataUtils.index_dataframe() 构建样本清单，不再依赖 json。
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", revision='a272c74'
        )
        self.processor = _ensure_llava_processing_args(self.processor)
        root = Path(args.data_root) if args.data_root else None
        # 用 DataUtils 的索引；如果 root 为空，允许外部用环境变量或默认路径挂载
        df = index_dataframe(root if root else Path("."))
        # 仅保留需要的 mode（默认 frame）
        df = df[df["mode"] == args.mode].copy()
        # 只取真实/伪造帧/视频的两类
        df = df[df["subset"].isin(["real_frames", "fake_frames", "real_videos", "fake_videos"])]
        # 记录必要字段（下游拼装 required cols）
        self.records = df[["task", "method", "subset", "label", "mode", "rel_path", "abs_path"]].reset_index(drop=True)
        # 简单提示词（保持对话模板不变）
        self.prompt_text = "Is this image real or fake? Answer with 'real' or 'fake' only."

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img_path = row["abs_path"]
        label = int(row["label"])  # DataUtils: 0=real, 1=fake
        image = Image.open(img_path).convert("RGB")
        chat = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.prompt_text},
            ],
        }]
        prompt = self.processor.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True
        )
        meta = {
            "task": str(row["task"]),
            "method": str(row["method"]),
            "subset": str(row["subset"]),
            "mode": str(row["mode"]),
            "rel_path": str(row["rel_path"]),
            "abs_path": str(row["abs_path"]),
            "label": label,
        }
        return inputs, meta


def validate(args, model, cls_test_dataloader):
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')
    processor = _ensure_llava_processing_args(processor)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []  # 用于构建 DataUtils.required schema
    with torch.no_grad():
        for inputs, meta in tqdm(cls_test_dataloader):
            # DataLoader will stack each sample [1, ...] into [B, 1, ...]; only remove dim=1 to keep batch dim
            inputs["input_ids"] = inputs["input_ids"].squeeze(1).to(device)
            inputs["attention_mask"] = inputs["attention_mask"].squeeze(1).to(device)
            inputs["pixel_values"] = inputs["pixel_values"].squeeze(1).to(device)
            output = model.generate(**inputs, max_new_tokens=256)
            # 单样本 batch（val_batch_size 可>1，但当前数据管线按一条/次处理）
            for i in range(output.shape[0]):
                response = processor.decode(output[i], skip_special_tokens=True).split('?')[-1]
                log.info(response)
                # 与 DataUtils 统一: label 0=real, 1=fake；pred 同样 0/1
                if "ASSISTANT: Real" in response:
                    pred = 0
                elif "ASSISTANT: Fake" in response:
                    pred = 1
                # compare the count of 'real' and 'fake' in the response
                elif response.lower().count('real') > response.lower().count('fake'):
                    pred = 1
                elif response.lower().count('fake') > response.lower().count('real'):
                    pred = 0
                else:
                    log.info(f"no explicit 'fake' or 'real' in response: {response}")
                    pred = random.choice([0, 1])
                # 简单分数（无置信度时用 0/1 作为 score）；可替换为更细致的解析
                score = float(pred)
                meta_i = meta[i] if isinstance(meta, (list, tuple)) else meta
                label = int(meta_i["label"])
                rows.append({
                    "sample_id": str(meta_i["rel_path"]),  # 可稳定 join 的 id
                    "task": str(meta_i["task"]),
                    "method": str(meta_i["method"]),
                    "subset": str(meta_i["subset"]),
                    "label": label,
                    "model": str(args.model_path),
                    "mode": str(meta_i["mode"]),
                    "score": score,
                    "pred": int(pred),
                })

    df_pred = standardise_predictions(pd.DataFrame(rows))
    df_pred.to_csv(args.pred_csv, index=False)
    rep = quick_report(df_pred)
    import json
    with open(args.report_json, "w") as f:
        json.dump({
            "overall": rep["overall"],
            "by_task": {
                "accuracy": rep["by_task"]["accuracy"].to_dict(orient="records"),
                "roc_auc": rep["by_task"]["roc_auc"].to_dict(orient="records"),
                "tpr@1e-2": rep["by_task"]["tpr@1e-2"].to_dict(orient="records"),
            },
            "by_method": {
                "accuracy": rep["by_method"]["accuracy"].to_dict(orient="records"),
                "roc_auc": rep["by_method"]["roc_auc"].to_dict(orient="records"),
                "tpr@1e-2": rep["by_method"]["tpr@1e-2"].to_dict(orient="records"),
            },
        }, f, indent=2)
    log.info(f"Saved predictions to {args.pred_csv}")
    log.info(f"Saved report to {args.report_json}")


def load_model(args):
    log.info("Loading model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
        revision='a272c74' if 'llava-hf/llava-1.5-7b-hf' in args.model_path else None,
    ).eval()
    log.info(f"Successfully loaded model from: {args.model_path}")
    return model


def main():
    args = parse_args()
    model = load_model(args)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    cls_test_dataset = legion_cls_dataset(args)
    cls_test_dataloader = DataLoader(
        cls_test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    validate(args, model, cls_test_dataloader)


if __name__ == "__main__":
    main()
