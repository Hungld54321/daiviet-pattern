# -*- coding: utf-8 -*-
"""
benchmark/train_benchmark.py
Train LoRA models for DaiViet-Pattern Benchmark (Task 4.1)

Models:
  M2 — SDXL 1.0 + LoRA  (rank=16, lora_alpha=32, L_diffusion only)
  M3 — SD 1.5  + LoRA   (rank=16, lora_alpha=32, L_diffusion only)
  M6 — SDXL 1.0 + LoRA + L_cultural (Gram-matrix VGG-16 conv3_3, lambda=0.3)

Usage:
  python benchmark/train_benchmark.py --model M2
  python benchmark/train_benchmark.py --model M3
  python benchmark/train_benchmark.py --model M6

Hardware requirement: RTX 4080 16 GB (fp16, gradient checkpointing)
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from tqdm import tqdm

# ── Diffusers / PEFT ──────────────────────────────────────────────────────────
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPTokenizerFast,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "benchmark" / "checkpoints"

SEED          = 42
EPOCHS        = 50
SAVE_EVERY    = 10           # save checkpoint every N epochs
WARMUP_STEPS  = 100          # cosine LR warmup steps

LORA_RANK     = 16
LORA_ALPHA    = 32
# LoRA target modules in UNet (attention projections)
UNET_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
# LoRA target modules in text encoders
TE_LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "out_proj"]

MODEL_CONFIGS = {
    "M2": {
        "base":          "stabilityai/stable-diffusion-xl-base-1.0",
        "model_type":    "sdxl",
        "resolution":    768,
        "batch_size":    4,
        "lr_unet":       1e-4,
        "lr_te":         1e-5,
        "data_dir":      "benchmark_data/768/D_all/train",
        "lambda_cultural": 0.0,
    },
    "M3": {
        "base":          "runwayml/stable-diffusion-v1-5",
        "model_type":    "sd15",
        "resolution":    512,
        "batch_size":    8,
        "lr_unet":       1e-4,
        "lr_te":         1e-5,
        "data_dir":      "benchmark_data/512/D_all/train",
        "lambda_cultural": 0.0,
    },
    "M6": {
        "base":          "stabilityai/stable-diffusion-xl-base-1.0",
        "model_type":    "sdxl",
        "resolution":    768,
        "batch_size":    4,
        "lr_unet":       1e-4,
        "lr_te":         1e-5,
        "data_dir":      "benchmark_data/768/D_all/train",
        "lambda_cultural": 0.3,
    },
}

PERIOD_TRIGGERS = {
    "ly_tran_style": "Ly-Tran",
    "le_style":      "Le",
    "nguyen_style":  "Nguyen",
}

# ─────────────────────────────────────────────────────────────────────────────
# Utility: seed everything
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# VGG-16 feature extractor (conv3_3, layer index 14, after ReLU = index 15)
# ─────────────────────────────────────────────────────────────────────────────
class VGGFeatureExtractor(nn.Module):
    """Extract features at VGG-16 conv3_3 (after ReLU, feature map 256ch)."""

    def __init__(self, device):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # layers 0-15 inclusive: up to and including ReLU after conv3_3
        self.features = nn.Sequential(*list(vgg.children())[:16])
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(device)
        # ImageNet normalisation
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406],
                                                   device=device).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225],
                                                   device=device).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1, 1] → [0, 1] → ImageNet norm
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return self.features(x)


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    B, C, H, W = feat.shape
    f = feat.view(B, C, -1)            # (B, C, H*W)
    gram = torch.bmm(f, f.transpose(1, 2))   # (B, C, C)
    return gram / (C * H * W)


def cultural_loss(pred_px: torch.Tensor,
                  ref_px:  torch.Tensor,
                  vgg_extractor: VGGFeatureExtractor) -> torch.Tensor:
    """MSE between Gram matrices of pred and ref in conv3_3 feature space."""
    pred_feat = vgg_extractor(pred_px.float())
    with torch.no_grad():
        ref_feat = vgg_extractor(ref_px.float())
    return F.mse_loss(gram_matrix(pred_feat), gram_matrix(ref_feat))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
def _detect_period(caption: str) -> str:
    for trigger, period in PERIOD_TRIGGERS.items():
        if trigger in caption:
            return period
    return "Unknown"


class MotifDataset(Dataset):
    """Loads (image, caption, period) tuples from benchmark_data train split."""

    def __init__(self, data_dir: Path, resolution: int, with_ref: bool = False):
        self.img_dir = data_dir / "images"
        self.cap_dir = data_dir / "captions"
        self.images  = sorted(self.img_dir.glob("*.png"))
        self.resolution = resolution
        self.with_ref   = with_ref

        # Build {period -> list[index]} for reference sampling in M6
        self.period_to_indices: dict[str, list[int]] = {}
        self.periods: list[str] = []
        for i, img_path in enumerate(self.images):
            cap = self._load_caption(img_path)
            period = _detect_period(cap)
            self.periods.append(period)
            self.period_to_indices.setdefault(period, []).append(i)

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution),
                              interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _load_caption(self, img_path: Path) -> str:
        cap_path = self.cap_dir / (img_path.stem + ".txt")
        if cap_path.exists():
            return cap_path.read_text(encoding="utf-8").strip()
        return ""

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.images[idx]
        image    = Image.open(img_path).convert("RGB")
        image    = self.transform(image)
        caption  = self._load_caption(img_path)
        period   = self.periods[idx]

        item = {"image": image, "caption": caption, "period": period}

        if self.with_ref:
            # Sample a reference image from the same period (different index)
            candidates = [i for i in self.period_to_indices.get(period, [idx])
                          if i != idx]
            ref_idx  = random.choice(candidates) if candidates else idx
            ref_path = self.images[ref_idx]
            ref_img  = Image.open(ref_path).convert("RGB")
            item["ref_image"] = self.transform(ref_img)

        return item


# ─────────────────────────────────────────────────────────────────────────────
# Text encoding helpers
# ─────────────────────────────────────────────────────────────────────────────
def encode_text_sdxl(captions, tokenizer_1, tokenizer_2,
                     text_encoder_1, text_encoder_2, device):
    """Returns (encoder_hidden_states, pooled_prompt_embeds)."""
    tokens_1 = tokenizer_1(
        captions, padding="max_length", max_length=tokenizer_1.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    tokens_2 = tokenizer_2(
        captions, padding="max_length", max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        out_1 = text_encoder_1(tokens_1, output_hidden_states=True)
        hidden_1 = out_1.hidden_states[-2]          # (B, 77, 768)

        out_2 = text_encoder_2(tokens_2, output_hidden_states=True)
        hidden_2 = out_2.hidden_states[-2]          # (B, 77, 1280)
        pooled   = out_2[0]                         # (B, 1280)

    encoder_hidden_states = torch.cat([hidden_1, hidden_2], dim=-1)  # (B,77,2048)
    return encoder_hidden_states, pooled


def encode_text_sd15(captions, tokenizer, text_encoder, device):
    """Returns encoder_hidden_states for SD 1.5."""
    tokens = tokenizer(
        captions, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        return text_encoder(tokens)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model_sdxl(base: str, device):
    print(f"  Loading SDXL components from {base} ...")
    tokenizer_1 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizerFast.from_pretrained(base, subfolder="tokenizer_2")

    text_encoder_1 = CLIPTextModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=torch.float32,
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        base, subfolder="text_encoder_2", torch_dtype=torch.float32,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        base, subfolder="vae", torch_dtype=torch.float32,
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        base, subfolder="unet", torch_dtype=torch.float32,
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    return dict(
        tokenizer_1=tokenizer_1, tokenizer_2=tokenizer_2,
        text_encoder_1=text_encoder_1, text_encoder_2=text_encoder_2,
        vae=vae, unet=unet, noise_scheduler=noise_scheduler,
    )


def load_model_sd15(base: str, device):
    print(f"  Loading SD 1.5 components from {base} ...")
    # SD 1.5 uses a single CLIP text encoder
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=torch.float32,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        base, subfolder="vae", torch_dtype=torch.float32,
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        base, subfolder="unet", torch_dtype=torch.float32,
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    return dict(
        tokenizer=tokenizer, text_encoder=text_encoder,
        vae=vae, unet=unet, noise_scheduler=noise_scheduler,
    )


def apply_lora(model: nn.Module, target_modules: list[str]) -> nn.Module:
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(model, lora_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler (cosine with linear warmup)
# ─────────────────────────────────────────────────────────────────────────────
def cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────────────────────────────────────
def train(model_name: str, cfg: dict,
          batch_size_override: int | None = None,
          cultural_every: int = 1):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training {model_name} on {device}")
    print(f"{'='*60}")

    out_dir = CHECKPOINT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_type      = cfg["model_type"]
    resolution      = cfg["resolution"]
    # --batch_size overrides config default (useful for M6 VRAM management)
    batch_size      = batch_size_override if batch_size_override else cfg["batch_size"]
    lambda_cultural = cfg["lambda_cultural"]
    use_cultural    = lambda_cultural > 0.0

    if use_cultural:
        print(f"  Cultural loss: lambda={lambda_cultural}, "
              f"every={cultural_every} step(s), "
              f"VAE decode per-sample (saves VRAM)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_dir = REPO_ROOT / cfg["data_dir"]
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    dataset    = MotifDataset(data_dir, resolution, with_ref=use_cultural)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    print(f"  Dataset: {len(dataset)} images, {len(dataloader)} steps/epoch")

    # ── Load model components ─────────────────────────────────────────────────
    if model_type == "sdxl":
        components = load_model_sdxl(cfg["base"], device)
        tokenizer_1     = components["tokenizer_1"]
        tokenizer_2     = components["tokenizer_2"]
        text_encoder_1  = components["text_encoder_1"]
        text_encoder_2  = components["text_encoder_2"]
    else:
        components = load_model_sd15(cfg["base"], device)
        tokenizer    = components["tokenizer"]
        text_encoder = components["text_encoder"]

    vae             = components["vae"]
    unet            = components["unet"]
    noise_scheduler = components["noise_scheduler"]

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    unet = apply_lora(unet, UNET_LORA_TARGETS)
    unet.print_trainable_parameters()

    if model_type == "sdxl":
        text_encoder_1 = apply_lora(text_encoder_1, TE_LORA_TARGETS)
        text_encoder_2 = apply_lora(text_encoder_2, TE_LORA_TARGETS)
    else:
        text_encoder = apply_lora(text_encoder, TE_LORA_TARGETS)

    # ── Gradient checkpointing (saves VRAM) ──────────────────────────────────
    unet.enable_gradient_checkpointing()

    # Freeze VAE (always)
    vae.requires_grad_(False)
    vae.eval()

    # ── Optimiser ─────────────────────────────────────────────────────────────
    if model_type == "sdxl":
        te_params = (list(text_encoder_1.parameters())
                     + list(text_encoder_2.parameters()))
    else:
        te_params = list(text_encoder.parameters())

    optimizer = torch.optim.AdamW([
        {"params": [p for p in unet.parameters() if p.requires_grad],
         "lr": cfg["lr_unet"]},
        {"params": [p for p in te_params if p.requires_grad],
         "lr": cfg["lr_te"]},
    ], betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

    total_steps = EPOCHS * len(dataloader)
    scheduler   = cosine_warmup_scheduler(optimizer, WARMUP_STEPS, total_steps)
    scaler      = torch.cuda.amp.GradScaler()   # fp16 mixed precision

    # ── VGG extractor for M6 ──────────────────────────────────────────────────
    vgg_extractor = VGGFeatureExtractor(device) if use_cultural else None

    # ── CSV logger ────────────────────────────────────────────────────────────
    log_path = out_dir / "training_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "loss_diffusion", "loss_cultural",
                         "loss_total", "lr"])

    # ── Save training config ──────────────────────────────────────────────────
    config_to_save = {**cfg, "epochs": EPOCHS, "lora_rank": LORA_RANK,
                      "lora_alpha": LORA_ALPHA, "seed": SEED,
                      "warmup_steps": WARMUP_STEPS,
                      "batch_size_actual": batch_size,
                      "cultural_every": cultural_every}
    (out_dir / "training_config.json").write_text(
        json.dumps(config_to_save, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    t_start     = time.time()

    for epoch in range(1, EPOCHS + 1):
        unet.train()
        if model_type == "sdxl":
            text_encoder_1.train()
            text_encoder_2.train()
        else:
            text_encoder.train()

        epoch_loss_diff = 0.0
        epoch_loss_cult = 0.0
        n_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for batch in pbar:
            images   = batch["image"].to(device)
            captions = batch["caption"]

            # ── VAE encode → latents ─────────────────────────────────────────
            with torch.no_grad(), torch.cuda.amp.autocast():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # ── Add noise ────────────────────────────────────────────────────
            noise     = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device, dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # ── Text encoding ─────────────────────────────────────────────────
            with torch.cuda.amp.autocast():
                if model_type == "sdxl":
                    enc_hidden, pooled = encode_text_sdxl(
                        list(captions), tokenizer_1, tokenizer_2,
                        text_encoder_1, text_encoder_2, device,
                    )
                    # SDXL added conditioning
                    bs = latents.shape[0]
                    time_ids = torch.tensor(
                        [[resolution, resolution, 0, 0, resolution, resolution]],
                        dtype=torch.float32, device=device,
                    ).repeat(bs, 1)
                    added_cond = {"time_ids": time_ids, "text_embeds": pooled}
                else:
                    enc_hidden = encode_text_sd15(
                        list(captions), tokenizer, text_encoder, device,
                    )
                    added_cond = {}

                # ── UNet forward ─────────────────────────────────────────────
                model_pred = unet(
                    noisy_latents, timesteps, enc_hidden,
                    added_cond_kwargs=added_cond if added_cond else None,
                ).sample

            # ── Diffusion loss ────────────────────────────────────────────────
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction_type: "
                                 f"{noise_scheduler.config.prediction_type}")

            loss_diff = F.mse_loss(model_pred.float(), target.float())

            # ── Cultural loss (M6 only) ───────────────────────────────────────
            loss_cult = torch.tensor(0.0, device=device)
            if use_cultural and (global_step % cultural_every == 0):
                # Predict x0 from epsilon prediction for each sample
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                a_t  = alphas_cumprod[timesteps].sqrt()[:, None, None, None]
                s_t  = (1 - alphas_cumprod[timesteps]).sqrt()[:, None, None, None]
                pred_x0_latent = (noisy_latents - s_t * model_pred) / a_t.clamp(min=1e-8)
                pred_x0_latent = pred_x0_latent / vae.config.scaling_factor

                # Decode per-sample to limit peak VRAM.
                # VAE decode of a full batch at 768px is expensive; looping over
                # individual samples keeps only one frame's activations in memory
                # at a time while still back-propagating through model_pred.
                ref_pixels_batch = batch["ref_image"].to(device)
                cult_losses: list[torch.Tensor] = []
                for i in range(pred_x0_latent.shape[0]):
                    with torch.cuda.amp.autocast(enabled=False):
                        pred_pix_i = vae.decode(
                            pred_x0_latent[i : i + 1].float()
                        ).sample.clamp(-1.0, 1.0)
                    ref_pix_i = ref_pixels_batch[i : i + 1]
                    cult_losses.append(
                        cultural_loss(pred_pix_i, ref_pix_i, vgg_extractor)
                    )
                loss_cult = torch.stack(cult_losses).mean()

            loss_total = loss_diff + lambda_cultural * loss_cult

            # ── Backward ─────────────────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in optimizer.param_groups[0]["params"]] +
                [p for p in optimizer.param_groups[1]["params"]],
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ── Logging ──────────────────────────────────────────────────────
            ld = loss_diff.item()
            lc = loss_cult.item() if use_cultural else 0.0
            lt = loss_total.item()
            lr_now = optimizer.param_groups[0]["lr"]
            log_writer.writerow([global_step, epoch, f"{ld:.6f}",
                                 f"{lc:.6f}", f"{lt:.6f}", f"{lr_now:.2e}"])
            log_file.flush()

            epoch_loss_diff += ld
            epoch_loss_cult += lc
            n_steps += 1
            global_step += 1

            pbar.set_postfix(
                loss=f"{lt:.4f}",
                cultural=f"{lc:.4f}" if use_cultural else "—",
            )

        avg_diff = epoch_loss_diff / max(n_steps, 1)
        avg_cult = epoch_loss_cult / max(n_steps, 1)
        elapsed  = (time.time() - t_start) / 60
        print(f"  Epoch {epoch:>2}/{EPOCHS}  "
              f"loss_diff={avg_diff:.5f}  "
              f"loss_cult={avg_cult:.5f}  "
              f"elapsed={elapsed:.1f}min")

        # ── Checkpoint ───────────────────────────────────────────────────────
        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            ckpt_dir = out_dir / f"epoch_{epoch:03d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Save LoRA weights (UNet)
            unet_lora = {k: v for k, v in unet.state_dict().items()
                         if "lora" in k}
            save_file(unet_lora, ckpt_dir / "unet_lora.safetensors")

            # Save LoRA weights (text encoders)
            if model_type == "sdxl":
                te1_lora = {k: v for k, v in text_encoder_1.state_dict().items()
                            if "lora" in k}
                te2_lora = {k: v for k, v in text_encoder_2.state_dict().items()
                            if "lora" in k}
                save_file(te1_lora, ckpt_dir / "te1_lora.safetensors")
                save_file(te2_lora, ckpt_dir / "te2_lora.safetensors")
            else:
                te_lora = {k: v for k, v in text_encoder.state_dict().items()
                           if "lora" in k}
                save_file(te_lora, ckpt_dir / "te_lora.safetensors")

            print(f"    Checkpoint saved → {ckpt_dir}")

    # ── Final save (best = last epoch) ───────────────────────────────────────
    # UNet LoRA — loaded by generate_benchmark.py (PEFT-compatible keys)
    final_unet_lora = {k: v for k, v in unet.state_dict().items()
                       if "lora" in k}
    save_file(final_unet_lora, out_dir / "pytorch_lora_weights.safetensors")
    print(f"  UNet LoRA saved  → {out_dir / 'pytorch_lora_weights.safetensors'}")

    # Text-encoder LoRA — also saved at root level for generate to pick up
    if model_type == "sdxl":
        te1 = {k: v for k, v in text_encoder_1.state_dict().items() if "lora" in k}
        te2 = {k: v for k, v in text_encoder_2.state_dict().items() if "lora" in k}
        save_file(te1, out_dir / "te1_lora_weights.safetensors")
        save_file(te2, out_dir / "te2_lora_weights.safetensors")
        print(f"  TE LoRA saved    → te1_lora_weights.safetensors, te2_lora_weights.safetensors")
    else:
        te = {k: v for k, v in text_encoder.state_dict().items() if "lora" in k}
        save_file(te, out_dir / "te_lora_weights.safetensors")
        print(f"  TE LoRA saved    → te_lora_weights.safetensors")

    log_file.close()
    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak VRAM: {vram_gb:.2f} GB")
    print(f"  Training complete in {(time.time()-t_start)/60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA model for DaiViet benchmark"
    )
    parser.add_argument("--model", choices=["M2", "M3", "M6"], required=True,
                        help="Model to train")
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override config batch size (e.g. use 2 for M6 to save VRAM)",
    )
    parser.add_argument(
        "--cultural_every", type=int, default=1,
        help=(
            "M6 only: compute cultural loss every N steps (default=1). "
            "Use 4 or 8 to reduce VAE decode overhead while still guiding training."
        ),
    )
    args = parser.parse_args()

    if args.model not in MODEL_CONFIGS:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    cfg = MODEL_CONFIGS[args.model]
    try:
        train(args.model, cfg,
              batch_size_override=args.batch_size,
              cultural_every=args.cultural_every)
    except Exception as e:
        print(f"\n[ERROR] Training {args.model} failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
