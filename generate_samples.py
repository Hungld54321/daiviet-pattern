"""
generate_samples.py
Load the final trained LoRA weights and generate 5 images per period prompt.
Creates individual PNGs and a combined 4×5 grid (samples_grid.png).

Run AFTER full training completes:
  py generate_samples.py
"""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusers import StableDiffusionXLPipeline
from peft import PeftModel

# ==================================================================
# PATHS & CONFIG
# ==================================================================
BASE_DIR    = Path("D:/Hoavandaiviet")
LORA_DIR    = BASE_DIR / "lora_output" / "daiviet_lora"
SAMPLE_DIR  = BASE_DIR / "generated_samples"
BASE_MODEL  = "stabilityai/stable-diffusion-xl-base-1.0"

N_SAMPLES   = 5
STEPS       = 30
CFG_SCALE   = 7.5
SEED_BASE   = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Prompts -------------------------------------------------------
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

# ==================================================================
# LOAD PIPELINE + LORA
# ==================================================================
print("Loading SDXL base pipeline ...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

print(f"Loading LoRA weights from {LORA_DIR} ...")
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
pipe.unet.merge_adapter()       # merge LoRA into weights for faster inference

pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=False)
print()

# ==================================================================
# GENERATE
# ==================================================================
all_images   = []   # list of (period, n, path)
period_order = list(PROMPTS.keys())

for period in period_order:
    prompt  = PROMPTS[period]
    out_dir = SAMPLE_DIR / period
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Period: {period}")

    for n in range(1, N_SAMPLES + 1):
        generator = torch.Generator(device=DEVICE).manual_seed(SEED_BASE + n)
        image = pipe(
            prompt=prompt,
            num_inference_steps=STEPS,
            guidance_scale=CFG_SCALE,
            height=512, width=512,
            generator=generator,
        ).images[0]

        path = out_dir / f"sample_{n:02d}.png"
        image.save(path)
        all_images.append((period, n, path))
        print(f"  [{n}/{N_SAMPLES}] saved -> {path}")
    print()

# ==================================================================
# 4×5 COMPARISON GRID
# ==================================================================
print("Building 4×5 sample grid ...")
fig, axes = plt.subplots(
    N_SAMPLES, len(period_order),
    figsize=(4 * len(period_order), 4 * N_SAMPLES)
)

for col, period in enumerate(period_order):
    axes[0, col].set_title(period, fontsize=12, fontweight="bold", pad=8)

for row in range(N_SAMPLES):
    axes[row, 0].set_ylabel(f"Sample {row+1}", fontsize=9, rotation=90, labelpad=4)
    for col, period in enumerate(period_order):
        img = Image.open(SAMPLE_DIR / period / f"sample_{row+1:02d}.png")
        axes[row, col].imshow(img)
        axes[row, col].axis("off")

fig.suptitle("DaiViet LoRA — Generated Samples (full training)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
grid_path = SAMPLE_DIR / "samples_grid.png"
plt.savefig(grid_path, dpi=150, bbox_inches="tight")
print(f"Grid saved -> {grid_path}")
print("\nGeneration complete.")
