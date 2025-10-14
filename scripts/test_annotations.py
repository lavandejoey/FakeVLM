import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

DEFAULT_PROMPT = "Is this image real or fake? Answer with one single word 'real' or 'fake'."
DEFAULT_CATE = "deepfake"
DEFAULT_LABEL_MAP = {"real": 1, "fake": 0}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def load_label_map(pairs: List[str]) -> Dict[str, int]:
    lm = {}
    for pair in pairs:
        if ":" not in pair:
            raise ValueError(f"Bad --label-map entry '{pair}', expected form 'name:int'")
        k, v = pair.split(":", 1)
        lm[k.strip().lower()] = int(v.strip())
    return lm


def to_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root).as_posix())
    except Exception:
        return path.as_posix()


def infer_label_from_path(p: Path, label_map: Dict[str, int]) -> Optional[int]:
    name = p.name.lower()
    parts = [seg.lower() for seg in p.parts]
    for key, val in label_map.items():
        key_l = key.lower()
        if any(re.search(rf"(?:^|\\b){re.escape(key_l)}(?:$|\\b)", seg) for seg in parts):
            return val
    for key, val in label_map.items():
        key_l = key.lower()
        if re.search(rf"(?:^|\\b){re.escape(key_l)}(?:$|\\b)", name):
            return val
    return None


def find_media(root: Path) -> Tuple[List[Path], List[Path]]:
    images, videos = [], []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            suf = Path(fn).suffix.lower()
            p = Path(dirpath) / fn
            if suf in IMG_EXTS:
                images.append(p)
            elif suf in VID_EXTS:
                videos.append(p)
    return images, videos


def ensure_pkg(pkg: str):
    try:
        __import__(pkg)
        return True
    except Exception:
        return False


def extract_frames_from_video(video_path: Path, out_dir: Path, img_ext: str, frame_every: int,
                              max_frames: Optional[int], force: bool = False) -> List[Path]:
    """
    Extract frames using OpenCV if available, else try ffmpeg.
    Avoid duplication: if out_dir exists and contains images, skip unless force=True.
    Returns list of frame paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted([p for p in out_dir.glob(f"*{img_ext}")])
    manifest_path = out_dir / ".extracted.json"
    if existing and not force:
        # Already extracted; return existing
        return existing

    # Clean old frames if forcing
    if force and existing:
        for p in existing:
            try:
                p.unlink()
            except Exception:
                pass

    saved = []
    used_opencv = False

    if ensure_pkg("cv2"):
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_every <= 0: frame_every = 1

        indices = list(range(0, total, frame_every))
        if max_frames == -1:
            # Extract all frames (no limit)
            pass
        elif max_frames is not None and max_frames > 0 and len(indices) > max_frames:
            # Evenly sample max_frames indices
            import numpy as np
            indices = list(np.linspace(0, max(0, total - 1), num=max_frames, dtype=int))

        idx_set = set(indices)
        i = 0
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i in idx_set:
                name = f"{frame_id:06d}{img_ext}"
                out_p = out_dir / name
                if img_ext.lower() in [".jpg", ".jpeg"]:
                    cv2.imwrite(str(out_p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    cv2.imwrite(str(out_p), frame)
                saved.append(out_p)
                frame_id += 1
            i += 1
        cap.release()
        used_opencv = True
    else:
        # Fall back to ffmpeg if present in PATH
        import shutil, subprocess, math
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Neither OpenCV nor ffmpeg is available to extract frames.")
        # probe duration/frames via ffprobe?
        # Simple strategy: extract every Nth frame via fps filter if frame_every>1, else default 1 fps
        # We'll extract at 1/frame_every of input fps approximately using the 'fps' filter.
        # If max_frames is given, we still might extract too many; we'll thin after.
        out_pattern = str((out_dir / f"%06d{img_ext}").as_posix())
        cmd = ["ffmpeg", "-i", str(video_path)]
        if frame_every > 1:
            # Can't directly use frame index step; approximate by fps filter if video has typical fps (unknown here).
            # We'll extract everything, then thin by selecting every Nth file. But to keep it simple, use select filter:
            # 'select=not(mod(n\,N))' to keep every Nth frame.
            cmd += ["-vf", f"select='not(mod(n\\,{frame_every}))'", "-vsync", "vfr"]
        cmd += ["-qscale:v", "2", out_pattern, "-y"]
        subprocess.run(cmd, check=True)
        saved = sorted([p for p in out_dir.glob(f"*{img_ext}")])
        if max_frames is not None and max_frames > 0 and len(saved) > max_frames:
            # Keep evenly spaced subset
            indices = [round(i) for i in list(__import__("numpy").linspace(0, len(saved) - 1, num=max_frames))]
            keep = set(indices)
            for j, p in enumerate(saved):
                if j not in keep:
                    try:
                        p.unlink()
                    except Exception:
                        pass
            saved = sorted([p for p in out_dir.glob(f"*{img_ext}")])

    # write manifest
    try:
        info = {"source_video": video_path.as_posix(), "num_frames": len(saved), "used_opencv": used_opencv}
        manifest_path.write_text(json.dumps(info, indent=2))
    except Exception:
        pass

    return saved


def main():
    ap = argparse.ArgumentParser(
        description="Build test_annotations.json from mixed videos and images (no duplicate extraction).")
    ap.add_argument("--root", type=Path, required=True, help="Root directory containing images and/or videos")
    ap.add_argument("--out", type=Path, default=Path("test_annotations.json"), help="Output JSON path")
    ap.add_argument("--label-map", nargs="*", default=None, help="Pairs like real:1 fake:0; default real:1 fake:0")
    ap.add_argument("--cate", type=str, default=DEFAULT_CATE,
                    help=f"Default category for entries (can be inferred or set globally). Default '{DEFAULT_CATE}'")
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Default prompt for conversations[0].value")
    ap.add_argument("--img-ext", type=str, default=".jpg", choices=[".jpg", ".jpeg", ".png"],
                    help="Image extension for extracted frames")
    ap.add_argument("--frame-every", type=int, default=30,
                    help="Keep every Nth frame when extracting (OpenCV index-based; ffmpeg fallback uses select filter)")
    ap.add_argument("--max-frames-per-video", type=int, default=-1,
                    help="Cap number of frames per video (evenly sampled). 0 = no cap, -1 = extract all frames.")
    ap.add_argument("--force", action="store_true",
                    help="Force re-extraction even if a frame folder already exists with images")
    args = ap.parse_args()

    label_map = DEFAULT_LABEL_MAP.copy()
    if args.label_map:
        label_map = load_label_map(args.label_map)

    root: Path = args.root
    images, videos = find_media(root)

    # 1) Extract frames for videos into folder with same stem
    all_image_paths: List[Path] = []
    for v in videos:
        out_dir = v.with_suffix("")  # a/b/1.mp4 -> a/b/1
        saved = extract_frames_from_video(
            video_path=v,
            out_dir=out_dir,
            img_ext=args.img_ext,
            frame_every=max(1, args.frame_every),
            max_frames=(None if args.max_frames_per_video <= 0 else args.max_frames_per_video),
            force=args.force,
        )
        all_image_paths.extend(saved)

    # 2) Add existing images (outside any just-created frame folders, but including inside is fine; we'll de-dup later)
    all_image_paths.extend(images)

    # De-duplicate by absolute path string
    seen = set()
    unique_images = []
    for p in all_image_paths:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            unique_images.append(p)

    # 3) Build annotations
    data = []
    for img_path in unique_images:
        rel = to_rel(img_path, root)
        label = infer_label_from_path(img_path, label_map)
        if label is None:
            # default to 'fake' if available else smallest label
            label = label_map.get("fake", min(label_map.values()))
        item = {
            "image": str(img_path.resolve()),
            "label": int(label),
            "cate": args.cate,
            "conversations": [{"from": "human", "value": args.prompt}],
        }
        data.append(item)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} entries to {args.out}")


if __name__ == "__main__":
    main()
