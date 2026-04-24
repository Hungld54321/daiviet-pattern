# -*- coding: utf-8 -*-
"""
benchmark/generate_benchmark.py
Generate test images for all models (M1 vanilla, M2, M3, M6).

Per model × per period: 50 images (seeds 0-49)

Usage:
  python benchmark/generate_benchmark.py --model M1
  python benchmark/generate_benchmark.py --model M2
  python benchmark/generate_benchmark.py --model M3
  python benchmark/generate_benchmark.py --model M6
  python benchmark/generate_benchmark.py --model all
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from safetensors.torch import load_file

# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "benchmark" / "checkpoints"
OUTPUT_DIR     = REPO_ROOT / "benchmark" / "generated"
BASELINE_DIR   = REPO_ROOT / "benchmark_data" / "baseline_prompts"

PERIODS = ["Ly-Tran", "Le", "Nguyen"]
N_IMAGES = 50               # images per period per model
SEEDS    = list(range(N_IMAGES))  # 0 … 49

INFERENCE_STEPS    = 30
GUIDANCE_SCALE     = 7.5
NEGATIVE_PROMPT    = (
    "blurry, low quality, modern, western style, photograph, "
    "realistic, 3d render, text, watermark"
)

TRIGGER_WORDS = {
    "Ly-Tran": "ly_tran_style",
    "Le":      "le_style",
    "Nguyen":  "nguyen_style",
}
PERIOD_LABELS = {
    "Ly-Tran": "Ly-Tran dynasty",
    "Le":      "Le dynasty",
    "Nguyen":  "Nguyen dynasty",
}

MODEL_CONFIGS = {
    # M1: SDXL vanilla — no LoRA, no trigger word
    "M1": {
        "base":       "stabilityai/stable-diffusion-xl-base-1.0",
        "model_type": "sdxl",
        "resolution": 768,
        "lora_path":  None,          # no LoRA
        "use_trigger": False,
    },
    # M2: SDXL + LoRA
    "M2": {
        "base":       "stabilityai/stable-diffusion-xl-base-1.0",
        "model_type": "sdxl",
        "resolution": 768,
        "lora_path":  CHECKPOINT_DIR / "M2" / "pytorch_lora_weights.safetensors",
        "use_trigger": True,
    },
    # M3: SD 1.5 + LoRA
    "M3": {
        "base":       "runwayml/stable-diffusion-v1-5",
        "model_type": "sd15",
        "resolution": 512,
        "lora_path":  CHECKPOINT_DIR / "M3" / "pytorch_lora_weights.safetensors",
        "use_trigger": True,
    },
    # M6: SDXL + LoRA + L_cultural
    "M6": {
        "base":       "stabilityai/stable-diffusion-xl-base-1.0",
        "model_type": "sdxl",
        "resolution": 768,
        "lora_path":  CHECKPOINT_DIR / "M6" / "pytorch_lora_weights.safetensors",
        "use_trigger": True,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(period: str, use_trigger: bool,
                 baseline_prompts: dict | None = None) -> str:
    """Build inference prompt.

    M1: use baseline prompt from baseline_prompts.json (no trigger).
    M2/M3/M6: use trigger-word prompt.
    """
    if not use_trigger:
        # M1 — vanilla baseline prompt
        if baseline_prompts and period in baseline_prompts:
            return baseline_prompts[period]
        label = PERIOD_LABELS.get(period, period)
        return (
            f"Vietnamese {label} traditional ornamental pattern, "
            "black and white line art, traditional Dai Viet art, "
            "high quality, detailed"
        )
    trigger = TRIGGER_WORDS[period]
    label   = PERIOD_LABELS.get(period, period)
    return (
        f"{trigger}, Vietnamese {label} ornamental pattern, "
        "traditional Dai Viet art, black and white line art, "
        "high quality, detailed"
    )


def load_baseline_prompts() -> dict | None:
    """Load prompts from baseline_prompts.json if it exists."""
    p = BASELINE_DIR / "baseline_prompts.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline loading
# ─────────────────────────────────────────────────────────────────────────────
def load_pipeline(model_name: str, cfg: dict, device: torch.device):
    """Load diffusion pipeline and optionally inject LoRA weights."""
    base        = cfg["base"]
    model_type  = cfg["model_type"]
    lora_path   = cfg.get("lora_path")
    resolution  = cfg["resolution"]

    print(f"  Loading base model: {base}")

    if model_type == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            base,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)

    # Memory optimisations
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  xformers attention enabled")
        except Exception:
            pass

    # ── Inject LoRA weights ──────────────────────────────────────────────────
    if lora_path is not None:
        if not Path(lora_path).exists():
            raise FileNotFoundError(
                f"LoRA checkpoint not found for {model_name}: {lora_path}\n"
                f"Run: python benchmark/train_benchmark.py --model {model_name}"
            )
        print(f"  Loading LoRA weights from {lora_path}")
        # Load LoRA weights into UNet via diffusers load_lora_weights
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
        print(f"  LoRA weights loaded")

    pipe.unet.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_for_model(model_name: str, cfg: dict, device: torch.device):
    """Generate 50 images × 3 periods for one model. Logs to CSV."""
    print(f"\n{'='*60}")
    print(f"  Generating: {model_name}")
    print(f"{'='*60}")

    out_base = OUTPUT_DIR / model_name
    out_base.mkdir(parents=True, exist_ok=True)

    baseline_prompts = load_baseline_prompts() if not cfg["use_trigger"] else None

    # ── Load pipeline ────────────────────────────────────────────────────────
    pipe = load_pipeline(model_name, cfg, device)
    resolution = cfg["resolution"]

    # ── CSV log ──────────────────────────────────────────────────────────────
    log_path = out_base / "generation_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["filename", "model", "period", "seed",
                         "inference_time_sec", "vram_peak_gb", "prompt"])

    total_generated = 0
    torch.cuda.reset_peak_memory_stats()

    for period in PERIODS:
        period_dir = out_base / period
        period_dir.mkdir(parents=True, exist_ok=True)

        prompt = build_prompt(period, cfg["use_trigger"], baseline_prompts)
        print(f"\n  Period: {period}")
        print(f"  Prompt: {prompt[:90]}...")

        pbar = tqdm(SEEDS, desc=f"  {model_name}/{period}", leave=True)
        for seed in pbar:
            filename = f"{period}_seed_{seed:02d}.png"
            out_path = period_dir / filename

            if out_path.exists():
                pbar.set_postfix(status="skip (exists)")
                total_generated += 1
                continue

            generator = torch.Generator(device=device).manual_seed(seed)
            t0 = time.time()

            with torch.inference_mode(), torch.cuda.amp.autocast():
                if cfg["model_type"] == "sdxl":
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        num_inference_steps=INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        width=resolution,
                        height=resolution,
                        generator=generator,
                    ).images[0]
                else:
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        num_inference_steps=INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        width=resolution,
                        height=resolution,
                        generator=generator,
                    ).images[0]

            elapsed = time.time() - t0
            vram_gb = torch.cuda.max_memory_allocated() / 1e9

            image.save(out_path)
            log_writer.writerow([filename, model_name, period, seed,
                                 f"{elapsed:.2f}", f"{vram_gb:.2f}", prompt])
            log_file.flush()

            total_generated += 1
            pbar.set_postfix(t=f"{elapsed:.1f}s", vram=f"{vram_gb:.1f}GB")

    log_file.close()
    del pipe
    torch.cuda.empty_cache()

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n  {model_name}: generated {total_generated} images")
    print(f"  Peak VRAM: {peak_vram:.2f} GB")
    print(f"  Log: {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark images for DaiViet models"
    )
    parser.add_argument(
        "--model",
        choices=["M1", "M2", "M3", "M6", "all"],
        required=True,
        help="Model to generate images for (or 'all')",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model == "all":
        models = ["M1", "M2", "M3", "M6"]
    else:
        models = [args.model]

    for model_name in models:
        cfg = MODEL_CONFIGS[model_name]
        try:
            generate_for_model(model_name, cfg, device)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {model_name}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"\n[ERROR] {model_name} generation failed: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    print("\nGeneration complete.")


if __name__ == "__main__":
    main()
