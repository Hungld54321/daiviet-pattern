"""
run_eval.py
Full evaluation pipeline for the DaiViet LoRA paper.

Steps:
  1. Flatten generated samples into a single folder for FID
  2. Generate 20 baseline images (vanilla SDXL, no LoRA) with same prompts
  3. FID  - LoRA vs real subset  (clean-fid)
  4. FID  - baseline vs real subset
  5. SSIM - per-period nearest-real comparison
  6. CLIP score - image/text alignment (open_clip ViT-B/32)
  7. Loss curve stats from training log
  8. Print + save comparison table -> evaluation_results.txt

Note on FID sample size: FID is most reliable at >=2048 samples.
With only 20 generated images the score has high variance; we report it
alongside the real-image subset size for transparency (standard practice
in low-resource generative art papers).
"""

import os, re, shutil, random, math, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
BASE        = Path("D:/Hoavandaiviet")
REAL_DIR    = BASE / "train_data" / "images"        # 623 prepared PNGs
GEN_DIR     = BASE / "generated_samples"            # per-period subfolders
LORA_DIR    = BASE / "lora_output" / "daiviet_lora"
LOG_FILE    = BASE / "training_logs" / "train_full.log"
OUT_FILE    = BASE / "evaluation_results.txt"

# Working dirs created by this script
FLAT_LORA   = BASE / "eval_tmp" / "lora_flat"      # 20 lora images flat
FLAT_BASE   = BASE / "eval_tmp" / "baseline_flat"  # 20 baseline images flat
REAL_SUBSET = BASE / "eval_tmp" / "real_subset"    # 20 real images (matched)

BASE_MODEL  = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PERIODS = ["Dong-Son", "Ly-Tran", "Le", "Nguyen"]
N_PER_PERIOD = 5   # images per period already generated

PROMPTS = {
    "Dong-Son": (
        "Vietnamese ancient Bronze Age ornamental pattern, geometric Dong-Son motif, "
        "traditional Dai Viet art style, black and white line art"
    ),
    "Ly-Tran": (
        "Vietnamese Ly-Tran dynasty ornamental pattern, floral lotus motif, "
        "traditional Dai Viet art style, black and white line art"
    ),
    "Le": (
        "Vietnamese Le dynasty ornamental pattern, zoomorphic dragon motif, "
        "traditional Dai Viet art style, black and white line art"
    ),
    "Nguyen": (
        "Vietnamese Nguyen dynasty ornamental pattern, phoenix motif, "
        "traditional Dai Viet art style, black and white line art"
    ),
}

# Period label -> dataset subfolder name
PERIOD_FOLDER = {
    "Dong-Son": "co_dai",
    "Ly-Tran":  "ly_tran",
    "Le":       "le",
    "Nguyen":   "nguyen",
}

lines_out = []   # collected output lines

def log(msg=""):
    print(msg)
    lines_out.append(msg)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Flat copy of LoRA generated images + matching real subset
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 62)
log("DaiViet LoRA -- Evaluation Report")
log("=" * 62)
log()

for d in (FLAT_LORA, FLAT_BASE, REAL_SUBSET):
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)

# Flatten LoRA images
lora_paths = []
for period in PERIODS:
    src = GEN_DIR / period
    for n in range(1, N_PER_PERIOD + 1):
        p = src / f"sample_{n:02d}.png"
        if p.exists():
            dst = FLAT_LORA / f"{period}_{n:02d}.png"
            shutil.copy(p, dst)
            lora_paths.append((period, n, p))

log(f"LoRA generated images: {len(lora_paths)}")

# Build matching real subset: same number per period as generated
random.seed(42)
real_paths_by_period = {}
for period in PERIODS:
    folder = PERIOD_FOLDER[period]
    candidates = sorted(
        p for p in (BASE / "dataset" / folder).glob("*.jpg")
        if not p.name.startswith("._")   # skip macOS metadata files
    )
    chosen = random.sample(candidates, min(N_PER_PERIOD, len(candidates)))
    real_paths_by_period[period] = chosen
    for i, p in enumerate(chosen):
        img = Image.open(p).convert("RGB")
        # Match preprocessing: white-pad to square, resize 512x512
        w, h   = img.size
        maxd   = max(w, h)
        canvas = Image.new("RGB", (maxd, maxd), (255, 255, 255))
        canvas.paste(img, ((maxd - w) // 2, (maxd - h) // 2))
        canvas = canvas.resize((512, 512), Image.LANCZOS)
        canvas.save(REAL_SUBSET / f"{period}_{i+1:02d}.png")

real_subset_total = sum(len(v) for v in real_paths_by_period.values())
log(f"Real matched subset  : {real_subset_total} (5 per period)")
log()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Generate baseline (vanilla SDXL, no LoRA)
# ══════════════════════════════════════════════════════════════════════════════
log("-" * 62)
log("STEP 1 -- Generating baseline (SDXL, no LoRA) ...")
log("-" * 62)

from diffusers import StableDiffusionXLPipeline

pipe_base = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to(DEVICE)
pipe_base.set_progress_bar_config(disable=True)

for period in PERIODS:
    for n in range(1, N_PER_PERIOD + 1):
        gen = torch.Generator(device=DEVICE).manual_seed(42 + n)
        img = pipe_base(
            prompt=PROMPTS[period],
            num_inference_steps=30, guidance_scale=7.5,
            height=512, width=512, generator=gen,
        ).images[0]
        img.save(FLAT_BASE / f"{period}_{n:02d}.png")
        print(f"  baseline {period} {n}/{N_PER_PERIOD}", end="\r")

log(f"  Baseline images saved to {FLAT_BASE}  ({len(list(FLAT_BASE.glob('*.png')))} files)")

# Free baseline pipeline VRAM before LoRA pipeline
del pipe_base
torch.cuda.empty_cache()
log()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FID  (clean-fid)
# ══════════════════════════════════════════════════════════════════════════════
log("-" * 62)
log("STEP 2 -- FID (clean-fid, mode=clean)")
log(f"  Note: FID computed on {real_subset_total} real vs 20 generated.")
log("  Standard FID needs >=2048; scores here are indicative.")
log("-" * 62)

from cleanfid import fid as cleanfid

fid_lora = cleanfid.compute_fid(
    str(REAL_SUBSET), str(FLAT_LORA),
    mode="clean", num_workers=0, verbose=False,
)
log(f"  FID (SDXL + LoRA, ours) : {fid_lora:.2f}")

fid_base = cleanfid.compute_fid(
    str(REAL_SUBSET), str(FLAT_BASE),
    mode="clean", num_workers=0, verbose=False,
)
log(f"  FID (SDXL baseline)     : {fid_base:.2f}")
log()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SSIM  (generated vs nearest real, per period)
# ══════════════════════════════════════════════════════════════════════════════
log("-" * 62)
log("STEP 3 -- SSIM (generated vs period-matched real)")
log("-" * 62)

from skimage.metrics import structural_similarity as ssim_fn

def compute_ssim_for_dir(gen_flat_dir):
    """Average SSIM between each generated image and its period-matched real."""
    scores = []
    for period in PERIODS:
        real_imgs = [
            np.array(Image.open(p).convert("L").resize((512, 512)))
            for p in real_paths_by_period[period]
        ]
        for n in range(1, N_PER_PERIOD + 1):
            gp = gen_flat_dir / f"{period}_{n:02d}.png"
            if not gp.exists():
                continue
            gen_gray = np.array(Image.open(gp).convert("L").resize((512, 512)))
            # Compare against each real from same period, take best match
            best = max(
                ssim_fn(gen_gray, r, data_range=255) for r in real_imgs
            )
            scores.append(best)
    return float(np.mean(scores)), float(np.std(scores))

ssim_lora_mean, ssim_lora_std = compute_ssim_for_dir(FLAT_LORA)
ssim_base_mean, ssim_base_std = compute_ssim_for_dir(FLAT_BASE)

log(f"  SSIM (SDXL + LoRA, ours) : {ssim_lora_mean:.4f}  (std {ssim_lora_std:.4f})")
log(f"  SSIM (SDXL baseline)     : {ssim_base_mean:.4f}  (std {ssim_base_std:.4f})")
log()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CLIP Score  (open_clip ViT-B/32)
# ══════════════════════════════════════════════════════════════════════════════
log("-" * 62)
log("STEP 4 -- CLIP Score (open_clip ViT-B/32)")
log("-" * 62)

import open_clip
from torchvision import transforms

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model = clip_model.to(DEVICE).eval()
tokenizer  = open_clip.get_tokenizer("ViT-B-32")

def clip_score_for_dir(flat_dir):
    scores = []
    for period in PERIODS:
        prompt = PROMPTS[period]
        text_tok = tokenizer([prompt]).to(DEVICE)
        with torch.no_grad():
            text_feat = clip_model.encode_text(text_tok)
            text_feat = F.normalize(text_feat, dim=-1)
        for n in range(1, N_PER_PERIOD + 1):
            gp = flat_dir / f"{period}_{n:02d}.png"
            if not gp.exists():
                continue
            img = clip_preprocess(Image.open(gp).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                img_feat = clip_model.encode_image(img)
                img_feat = F.normalize(img_feat, dim=-1)
            sim = (img_feat @ text_feat.T).item()
            scores.append(sim)
    return float(np.mean(scores)), float(np.std(scores))

clip_lora_mean, clip_lora_std = clip_score_for_dir(FLAT_LORA)
clip_base_mean, clip_base_std = clip_score_for_dir(FLAT_BASE)

log(f"  CLIP Score (SDXL + LoRA, ours) : {clip_lora_mean:.4f}  (std {clip_lora_std:.4f})")
log(f"  CLIP Score (SDXL baseline)     : {clip_base_mean:.4f}  (std {clip_base_std:.4f})")
log()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Per-period CLIP breakdown
# ══════════════════════════════════════════════════════════════════════════════
log("-" * 62)
log("STEP 5 -- Per-period CLIP Score breakdown (LoRA)")
log("-" * 62)

for period in PERIODS:
    prompt   = PROMPTS[period]
    text_tok = tokenizer([prompt]).to(DEVICE)
    with torch.no_grad():
        text_feat = clip_model.encode_text(text_tok)
        text_feat = F.normalize(text_feat, dim=-1)
    sims = []
    for n in range(1, N_PER_PERIOD + 1):
        gp = FLAT_LORA / f"{period}_{n:02d}.png"
        if not gp.exists():
            continue
        img = clip_preprocess(Image.open(gp).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feat = F.normalize(clip_model.encode_image(img), dim=-1)
        sims.append((img_feat @ text_feat.T).item())
    log(f"  {period:<12} : {np.mean(sims):.4f}")

log()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Loss curve statistics from training log
# ══════════════════════════════════════════════════════════════════════════════
log("-" * 62)
log("STEP 6 -- Training loss statistics (from train_full.log)")
log("-" * 62)

log_text   = LOG_FILE.read_text(encoding="utf-8", errors="replace")
step_re    = re.compile(
    r"step\s+(\d+)/\d+\s+ep\s+(\d+)/\d+\s+"
    r"diff=([\d.]+)\s+cult=([\d.eE+\-]+)\s+vgg_style=([\d.eE+\-]+)"
)

records = []
for m in step_re.finditer(log_text):
    records.append({
        "step":      int(m.group(1)),
        "epoch":     int(m.group(2)),
        "diff":      float(m.group(3)),
        "cult":      float(m.group(4)),
        "vgg_style": float(m.group(5)),
    })

# Epoch-level averages from log
epoch_re = re.compile(
    r"Epoch\s+(\d+)/50\s+complete\s+--\s+avg diff=([\d.]+)\s+avg cult=([\d.]+)"
)
epoch_avgs = {}
for m in epoch_re.finditer(log_text):
    epoch_avgs[int(m.group(1))] = {
        "diff": float(m.group(2)),
        "cult": float(m.group(3)),
    }

# Key checkpoints for paper table
checkpoints = {1: None, 10: None, 25: None, 50: None}
for ep, vals in epoch_avgs.items():
    if ep in checkpoints:
        checkpoints[ep] = vals

log(f"  {'Epoch':<8} {'Avg Diff Loss':<18} {'Avg Cult Loss'}")
log(f"  {'-'*8} {'-'*18} {'-'*14}")
for ep in [1, 10, 25, 50]:
    v = checkpoints.get(ep)
    if v:
        log(f"  {ep:<8} {v['diff']:<18.4f} {v['cult']:.6f}")
    else:
        log(f"  {ep:<8} {'N/A':<18} {'N/A'}")

if records:
    diffs = [r["diff"] for r in records]
    log(f"\n  Initial diff loss (step 50) : {records[0]['diff']:.4f}")
    log(f"  Final  diff loss (step 3850): {records[-1]['diff']:.4f}")
    log(f"  Min diff loss               : {min(diffs):.4f}")
    log(f"  Total log steps             : {len(records)}")
log()

# ══════════════════════════════════════════════════════════════════════════════
# FINAL TABLE
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 62)
log("RESULTS TABLE (for paper)")
log("=" * 62)
log()
log(f"{'Method':<28} | {'FID':>7} | {'SSIM':>7} | {'CLIP':>7}")
log(f"{'-'*28}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
log(f"{'SDXL (no LoRA)':<28} | {fid_base:>7.2f} | {ssim_base_mean:>7.4f} | {clip_base_mean:>7.4f}")
log(f"{'SDXL + LoRA (ours)':<28} | {fid_lora:>7.2f} | {ssim_lora_mean:>7.4f} | {clip_lora_mean:>7.4f}")
log()

# Delta
fid_delta  = fid_base  - fid_lora    # positive = LoRA better (lower FID)
ssim_delta = ssim_lora_mean - ssim_base_mean
clip_delta = clip_lora_mean - clip_base_mean

log(f"  FID  improvement  (LoRA vs base): {fid_delta:+.2f}  ({'better' if fid_delta>0 else 'worse'})")
log(f"  SSIM improvement  (LoRA vs base): {ssim_delta:+.4f}  ({'better' if ssim_delta>0 else 'worse'})")
log(f"  CLIP improvement  (LoRA vs base): {clip_delta:+.4f}  ({'better' if clip_delta>0 else 'worse'})")
log()
log("Notes:")
log("  FID computed on 20 generated vs 20 matched real images.")
log("  High variance expected at this scale; treat as indicative.")
log("  SSIM: best-match within period (5 real candidates per period).")
log("  CLIP: ViT-B/32, cosine similarity to generation prompt.")
log()

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
OUT_FILE.write_text("\n".join(lines_out), encoding="utf-8")
print(f"\nAll results saved -> {OUT_FILE}")

# Cleanup temp dirs
shutil.rmtree(BASE / "eval_tmp", ignore_errors=True)
print("Temp dirs cleaned up.")
print("Evaluation complete.")
