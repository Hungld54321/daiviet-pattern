# -*- coding: utf-8 -*-
"""
benchmark/evaluate_benchmark.py
Compute evaluation metrics for all generated models.

Metrics:
  FID    — Frechet Inception Distance (cleanfid)     ↓ lower is better
  CLIP   — CLIP ViT-L/14 image-text cosine sim       ↑ higher is better
  SSIM   — Structural Similarity (scikit-image)      ↑ higher is better
  LPIPS  — Learned Perceptual Image Patch Similarity  ↓ lower is better
  PSNR   — Peak Signal-to-Noise Ratio                ↑ higher is better

Usage:
  python benchmark/evaluate_benchmark.py --model all
  python benchmark/evaluate_benchmark.py --model M2

NOTE: FID on N=50 images per period is indicative only (not statistically
      robust). Report as FID50 and note limitation in the paper.
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
GENERATED   = REPO_ROOT / "benchmark" / "generated"
RESULTS_DIR = REPO_ROOT / "benchmark" / "results"

# Test-set ground-truth images for comparison
TEST_DIR_768 = REPO_ROOT / "benchmark_data" / "768" / "D_all" / "test" / "images"
TEST_DIR_512 = REPO_ROOT / "benchmark_data" / "512" / "D_all" / "test" / "images"

PERIODS = ["Ly-Tran", "Le", "Nguyen"]
PERIOD_TRIGGERS = {
    "Ly-Tran": "ly_tran_style",
    "Le":      "le_style",
    "Nguyen":  "nguyen_style",
}

MODEL_RESOLUTION = {
    "M1": 768, "M2": 768, "M3": 512, "M6": 768,
}
MODEL_TRIGGER_USED = {
    "M1": False, "M2": True, "M3": True, "M6": True,
}
# Prompt used for CLIP evaluation per model per period
EVAL_PROMPTS: dict[str, dict[str, str]] = {}   # filled at runtime


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_period_label(period: str) -> str:
    labels = {"Ly-Tran": "Ly-Tran dynasty", "Le": "Le dynasty",
               "Nguyen": "Nguyen dynasty"}
    return labels.get(period, period)


def _build_eval_prompt(model_name: str, period: str) -> str:
    """Return the text prompt used for CLIP score evaluation."""
    if MODEL_TRIGGER_USED.get(model_name, False):
        trigger = PERIOD_TRIGGERS[period]
        label   = _get_period_label(period)
        return (f"{trigger}, Vietnamese {label} ornamental pattern, "
                "traditional Dai Viet art, black and white line art, "
                "high quality, detailed")
    label = _get_period_label(period)
    return (f"Vietnamese {label} traditional ornamental pattern, "
            "black and white line art, traditional Dai Viet art, "
            "high quality, detailed")


def _get_test_images_for_period(period: str, resolution: int) -> list[Path]:
    """Return test images belonging to a given period (detect from filename stem)."""
    test_dir = TEST_DIR_768 if resolution >= 768 else TEST_DIR_512
    period_code = {
        "Ly-Tran": ["LYTRAN", "LT"],
        "Le":      ["LTH"],
        "Nguyen":  ["NGN"],
    }.get(period, [])
    all_imgs = sorted(test_dir.glob("*.png"))
    matched  = [p for p in all_imgs
                if any(code in p.stem.upper() for code in period_code)]
    # Fallback: return all test images if no period-specific match
    return matched if matched else all_imgs


def _load_image_np(path: Path, size: int | None = None) -> np.ndarray:
    """Load image as RGB numpy array, optionally resize."""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


def _load_image_tensor(path: Path, size: int, device: torch.device) -> torch.Tensor:
    """Load image as (1,3,H,W) float tensor in [-1,1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# FID (cleanfid)
# ─────────────────────────────────────────────────────────────────────────────
def compute_fid(gen_dir: Path, ref_images: list[Path], resolution: int) -> float:
    """FID between generated folder and reference images.

    We copy ref images to a temp folder so cleanfid can compare two folders.
    Returns the FID score, or -1.0 if cleanfid is unavailable.
    """
    try:
        from cleanfid import fid as cleanfid_fid
    except ImportError:
        print("  [WARN] cleanfid not installed. FID skipped. "
              "Install: pip install cleanfid")
        return -1.0

    if not gen_dir.exists() or len(list(gen_dir.glob("*.png"))) == 0:
        return -1.0
    if len(ref_images) == 0:
        return -1.0

    import shutil
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for p in ref_images:
            shutil.copy(p, tmp_path / p.name)
        try:
            score = cleanfid_fid.compute_fid(
                str(gen_dir), str(tmp_path),
                mode="clean", num_workers=0,
                batch_size=8, verbose=False,
            )
            return float(score)
        except Exception as e:
            print(f"  [WARN] FID computation failed: {e}")
            return -1.0


# ─────────────────────────────────────────────────────────────────────────────
# CLIP Score
# ─────────────────────────────────────────────────────────────────────────────
_clip_model   = None
_clip_proc    = None
_clip_device  = None


def _init_clip(device: torch.device):
    global _clip_model, _clip_proc, _clip_device
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        print("  Loading CLIP ViT-L/14 ...")
        _clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(device).eval()
        _clip_proc   = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        _clip_device = device


def compute_clip_score(gen_images: list[Path], prompt: str,
                       device: torch.device) -> float:
    """Average CLIP cosine similarity between images and prompt."""
    if not gen_images:
        return 0.0
    _init_clip(device)
    from transformers import CLIPProcessor

    scores = []
    batch_size = 16
    for i in range(0, len(gen_images), batch_size):
        batch_paths = gen_images[i : i + batch_size]
        pil_images  = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = _clip_proc(
            text=[prompt] * len(pil_images),
            images=pil_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(_clip_device)
        with torch.no_grad():
            out        = _clip_model(**inputs)
            img_embeds = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt_embeds = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sim        = (img_embeds * txt_embeds).sum(dim=-1)
        scores.extend(sim.cpu().tolist())

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SSIM / PSNR (scikit-image)
# ─────────────────────────────────────────────────────────────────────────────
def compute_ssim_psnr(gen_images: list[Path],
                      ref_images: list[Path],
                      resolution: int) -> tuple[float, float]:
    """Average SSIM and PSNR of generated images against their nearest reference.

    For each generated image we use the single nearest reference image
    (smallest pixel MSE) — this is a proxy since the datasets are unpaired.
    """
    if not gen_images or not ref_images:
        return 0.0, 0.0

    try:
        from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    except ImportError:
        print("  [WARN] scikit-image not installed. SSIM/PSNR skipped.")
        return 0.0, 0.0

    # Pre-load reference images
    refs_np = [_load_image_np(p, size=resolution) for p in ref_images]

    ssim_scores, psnr_scores = [], []
    for gen_path in gen_images:
        gen_np = _load_image_np(gen_path, size=resolution)
        # Find nearest reference by MSE
        mse_list = [
            np.mean((gen_np.astype(np.float32) - r.astype(np.float32)) ** 2)
            for r in refs_np
        ]
        best_ref = refs_np[int(np.argmin(mse_list))]

        ssim_val = structural_similarity(gen_np, best_ref, channel_axis=2,
                                         data_range=255)
        psnr_val = peak_signal_noise_ratio(best_ref, gen_np, data_range=255)
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)

    return (float(np.mean(ssim_scores)) if ssim_scores else 0.0,
            float(np.mean(psnr_scores)) if psnr_scores else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# LPIPS
# ─────────────────────────────────────────────────────────────────────────────
_lpips_fn = None

def _init_lpips(device: torch.device):
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips
            _lpips_fn = lpips.LPIPS(net="vgg").to(device)
            _lpips_fn.eval()
            print("  LPIPS (VGG) initialised")
        except ImportError:
            print("  [WARN] lpips not installed. Install: pip install lpips")
            _lpips_fn = None


def compute_lpips(gen_images: list[Path],
                  ref_images: list[Path],
                  resolution: int,
                  device: torch.device) -> float:
    """Average LPIPS against nearest reference (same unpaired proxy as SSIM)."""
    if not gen_images or not ref_images:
        return 0.0
    _init_lpips(device)
    if _lpips_fn is None:
        return 0.0

    # Pre-load references as tensors
    refs_t = [_load_image_tensor(p, resolution, device) for p in ref_images]

    scores = []
    for gen_path in gen_images:
        gen_t = _load_image_tensor(gen_path, resolution, device)
        # Find nearest reference (by LPIPS — expensive; use pixel MSE proxy)
        mse_list = [
            ((gen_t - r) ** 2).mean().item()
            for r in refs_t
        ]
        best_ref = refs_t[int(np.argmin(mse_list))]
        with torch.no_grad():
            dist = _lpips_fn(gen_t, best_ref).item()
        scores.append(dist)

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_name: str, device: torch.device) -> list[dict]:
    """Evaluate one model. Returns list of per-period result dicts."""
    resolution = MODEL_RESOLUTION.get(model_name, 768)
    gen_model_dir = GENERATED / model_name

    if not gen_model_dir.exists():
        print(f"  [SKIP] {model_name}: no generated images at {gen_model_dir}")
        return []

    print(f"\n  Evaluating {model_name} (res={resolution}) ...")
    period_results = []

    for period in PERIODS:
        gen_dir    = gen_model_dir / period
        gen_images = sorted(gen_dir.glob("*.png")) if gen_dir.exists() else []

        if not gen_images:
            print(f"    [SKIP] {period}: no images in {gen_dir}")
            continue

        print(f"\n    {period}: {len(gen_images)} generated images")

        # Reference test images for this period
        ref_images = _get_test_images_for_period(period, resolution)
        print(f"    Reference images: {len(ref_images)}")

        # Build evaluation prompt
        prompt = _build_eval_prompt(model_name, period)

        # ── Metrics ────────────────────────────────────────────────────────
        print(f"    Computing FID ...")
        fid_score = compute_fid(gen_dir, ref_images, resolution)

        print(f"    Computing CLIP Score ...")
        clip_score = compute_clip_score(gen_images, prompt, device)

        print(f"    Computing SSIM / PSNR ...")
        ssim_score, psnr_score = compute_ssim_psnr(gen_images, ref_images, resolution)

        print(f"    Computing LPIPS ...")
        lpips_score = compute_lpips(gen_images, ref_images, resolution, device)

        # ── Read average inference time from generation log ──────────────
        inf_time_avg = _read_avg_inference_time(gen_model_dir, period)

        row = {
            "model":       model_name,
            "period":      period,
            "n_generated": len(gen_images),
            "n_reference": len(ref_images),
            "FID":         round(fid_score, 4),
            "CLIP_Score":  round(clip_score, 4),
            "SSIM":        round(ssim_score, 4),
            "LPIPS":       round(lpips_score, 4),
            "PSNR":        round(psnr_score, 4),
            "inf_time_avg_sec": round(inf_time_avg, 2),
        }
        period_results.append(row)

        print(f"    FID={fid_score:.2f}  CLIP={clip_score:.4f}  "
              f"SSIM={ssim_score:.4f}  LPIPS={lpips_score:.4f}  "
              f"PSNR={psnr_score:.2f}")

    return period_results


def _read_avg_inference_time(model_dir: Path, period: str) -> float:
    """Parse generation_log.csv to get average inference time for a period."""
    log_path = model_dir / "generation_log.csv"
    if not log_path.exists():
        return 0.0
    times = []
    try:
        with open(log_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("period") == period:
                    times.append(float(row.get("inference_time_sec", 0)))
    except Exception:
        pass
    return float(np.mean(times)) if times else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate and report
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_overall(per_period: list[dict]) -> dict:
    """Compute macro-average over all periods for a single model."""
    if not per_period:
        return {}
    model = per_period[0]["model"]
    metrics = ["FID", "CLIP_Score", "SSIM", "LPIPS", "PSNR", "inf_time_avg_sec"]
    overall = {"model": model}
    for m in metrics:
        vals = [r[m] for r in per_period if r.get(m, -1) >= 0]
        overall[m] = round(float(np.mean(vals)), 4) if vals else -1.0
    return overall


def write_csv_results(all_period_rows: list[dict], all_overall_rows: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-period CSV
    period_csv = RESULTS_DIR / "metrics_per_period.csv"
    if all_period_rows:
        fieldnames = list(all_period_rows[0].keys())
        with open(period_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_period_rows)
        print(f"\n  Saved: {period_csv}")

    # Overall CSV
    overall_csv = RESULTS_DIR / "metrics_overall.csv"
    if all_overall_rows:
        fieldnames = list(all_overall_rows[0].keys())
        with open(overall_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_overall_rows)
        print(f"  Saved: {overall_csv}")

    return period_csv, overall_csv


def write_report(all_period_rows: list[dict], all_overall_rows: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "evaluation_report.txt"

    lines = [
        "=" * 72,
        "  DAIVIET-PATTERN BENCHMARK — EVALUATION REPORT",
        "  Task 4.1  |  DaiViet-Pattern v2.0",
        "=" * 72,
        "",
        "NOTE: FID computed on N=50 images per period (indicative, denoted FID50).",
        "      SSIM / LPIPS / PSNR use nearest-reference proxy (unpaired setting).",
        "",
        "-" * 72,
        "  OVERALL (macro-average over 3 periods)",
        "-" * 72,
        f"  {'Model':<8}  {'FID50':>7}  {'CLIP':>7}  {'SSIM':>7}  "
        f"{'LPIPS':>7}  {'PSNR':>7}  {'InfTime(s)':>10}",
        "  " + "-" * 60,
    ]
    for row in all_overall_rows:
        fid  = f"{row['FID']:.2f}"   if row['FID'] >= 0 else "  n/a"
        lines.append(
            f"  {row['model']:<8}  {fid:>7}  "
            f"{row['CLIP_Score']:>7.4f}  {row['SSIM']:>7.4f}  "
            f"{row['LPIPS']:>7.4f}  {row['PSNR']:>7.2f}  "
            f"{row['inf_time_avg_sec']:>10.2f}"
        )

    lines += [
        "",
        "-" * 72,
        "  PER-PERIOD BREAKDOWN",
        "-" * 72,
        f"  {'Model':<8}  {'Period':<10}  {'FID50':>7}  {'CLIP':>7}  "
        f"{'SSIM':>7}  {'LPIPS':>7}  {'PSNR':>7}",
        "  " + "-" * 60,
    ]
    for row in all_period_rows:
        fid = f"{row['FID']:.2f}" if row['FID'] >= 0 else "   n/a"
        lines.append(
            f"  {row['model']:<8}  {row['period']:<10}  {fid:>7}  "
            f"{row['CLIP_Score']:>7.4f}  {row['SSIM']:>7.4f}  "
            f"{row['LPIPS']:>7.4f}  {row['PSNR']:>7.2f}"
        )

    lines += [
        "",
        "-" * 72,
        "  METRIC DEFINITIONS",
        "-" * 72,
        "  FID50    : Frechet Inception Distance (lower=better). N=50, indicative.",
        "  CLIP     : CLIP ViT-L/14 image-text cosine sim (higher=better).",
        "  SSIM     : Structural Similarity Index (higher=better).",
        "  LPIPS    : Learned Perceptual Image Patch Similarity (lower=better).",
        "  PSNR     : Peak Signal-to-Noise Ratio in dB (higher=better).",
        "  InfTime  : Average inference time per image in seconds.",
        "=" * 72,
    ]

    text = "\n".join(lines)
    report_path.write_text(text, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print()
    print(text)
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DaiViet benchmark generated images"
    )
    parser.add_argument(
        "--model",
        choices=["M1", "M2", "M3", "M6", "all"],
        required=True,
        help="Model(s) to evaluate",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models = ["M1", "M2", "M3", "M6"]
    else:
        models = [args.model]

    all_period_rows: list[dict] = []
    all_overall_rows: list[dict] = []

    for model_name in models:
        try:
            period_rows = evaluate_model(model_name, device)
            all_period_rows.extend(period_rows)
            if period_rows:
                overall = aggregate_overall(period_rows)
                all_overall_rows.append(overall)
        except Exception as e:
            print(f"\n[ERROR] Evaluation of {model_name} failed: {e}",
                  file=sys.stderr)
            import traceback; traceback.print_exc()

    write_csv_results(all_period_rows, all_overall_rows)
    write_report(all_period_rows, all_overall_rows)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
