"""
test_run.py
Smoke-test LoRA training: 50 images, 5 epochs (~15-20 min).
Also generates 4 sample images at the end so you can confirm the
pipeline is working before committing to the full overnight run.

Output:
  lora_output/test_run/          <- saved LoRA weights
  training_logs/test_loss.png    <- loss curve
  generated_samples/test_run/    <- 4 sample images
"""

import os, sys, math, time, random
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionXLPipeline,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model

# ==================================================================
# PATHS
# ==================================================================
BASE_DIR      = Path("D:/Hoavandaiviet")
TRAIN_IMAGES  = BASE_DIR / "train_data" / "images"
TRAIN_CAPS    = BASE_DIR / "train_data" / "captions"
OUTPUT_DIR    = BASE_DIR / "lora_output" / "test_run"
LOG_DIR       = BASE_DIR / "training_logs"
SAMPLE_DIR    = BASE_DIR / "generated_samples" / "test_run"

for d in (OUTPUT_DIR, LOG_DIR, SAMPLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ==================================================================
# SMOKE-TEST CONFIG  (reduced from full training)
# ==================================================================
BASE_MODEL   = "stabilityai/stable-diffusion-xl-base-1.0"
RESOLUTION   = 512
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
EPOCHS       = 5
LR           = 1e-4
LR_WARMUP    = 20
MIXED_PREC   = "fp16"
LOG_STEPS    = 10
CULTURAL_W   = 0.3
N_TRAIN_IMGS = 50       # <- key smoke-test limit

# LoRA (identical to full run)
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0",
                 "to_add_q", "to_add_k", "to_add_v"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {DEVICE}")
print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"VRAM    : {round(torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB")
print(f"Mode    : SMOKE TEST ({N_TRAIN_IMGS} images, {EPOCHS} epochs)\n")

# ==================================================================
# DATASET
# ==================================================================
class DaiVietDataset(Dataset):
    def __init__(self, img_dir, cap_dir, resolution):
        self.img_dir = img_dir
        self.cap_dir = cap_dir
        self.stems   = sorted([p.stem for p in img_dir.glob("*.png")])
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):  return len(self.stems)

    def __getitem__(self, idx):
        stem    = self.stems[idx]
        img     = Image.open(self.img_dir / f"{stem}.png").convert("RGB")
        cap_p   = self.cap_dir / f"{stem}.txt"
        caption = cap_p.read_text(encoding="utf-8").strip() if cap_p.exists() else ""
        return self.transform(img), caption

# ==================================================================
# GRAM MATRIX & CULTURAL LOSS (VGG16 conv3_3)
# ==================================================================
class VGGStyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.features = nn.Sequential(*list(vgg.children())[:16]).eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def preprocess(self, x):
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    @staticmethod
    def gram(feat):
        B, C, H, W = feat.shape
        f = feat.view(B, C, H * W)
        return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)

    def forward(self, pred, ref):
        pred = self.preprocess(pred)
        ref  = self.preprocess(ref.detach())
        return F.mse_loss(self.gram(self.features(pred)),
                          self.gram(self.features(ref)))

# ==================================================================
# TEXT HELPERS
# ==================================================================
def tokenise(tok1, tok2, captions, device):
    def _tok(tok, caps):
        enc = tok(caps, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt")
        return enc.input_ids.to(device), enc.attention_mask.to(device)
    return _tok(tok1, captions), _tok(tok2, captions)

def encode_text(enc1, enc2, ids1, mask1, ids2, mask2):
    with torch.no_grad():
        out1 = enc1(ids1, attention_mask=mask1, output_hidden_states=True)
        out2 = enc2(ids2, attention_mask=mask2, output_hidden_states=True)
    return (torch.cat([out1.hidden_states[-2], out2.hidden_states[-2]], dim=-1),
            out2[0])

# ==================================================================
# LOAD MODELS
# ==================================================================
print("Loading SDXL ...  (first run will download ~7 GB from HuggingFace)")
tokenizer1  = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
tokenizer2  = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer_2")
text_enc1   = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder").to(DEVICE)
text_enc2   = CLIPTextModelWithProjection.from_pretrained(BASE_MODEL, subfolder="text_encoder_2").to(DEVICE)
vae         = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae").to(DEVICE)
unet        = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet").to(DEVICE)
noise_sched = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

text_enc1.requires_grad_(False)
text_enc2.requires_grad_(False)
vae.requires_grad_(False)

lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
                      lora_dropout=LORA_DROPOUT, bias="none")
unet = get_peft_model(unet, lora_cfg)
unet.print_trainable_parameters()
unet.enable_gradient_checkpointing()
style_loss_fn = VGGStyleLoss().to(DEVICE)
print()

# ==================================================================
# DATALOADER  (first 50 images only)
# ==================================================================
full_ds  = DaiVietDataset(TRAIN_IMAGES, TRAIN_CAPS, RESOLUTION)
if len(full_ds) == 0:
    print("ERROR: no training images found in", TRAIN_IMAGES)
    print("       Run prepare_training_data.py first.")
    sys.exit(1)

n_use   = min(N_TRAIN_IMGS, len(full_ds))
indices = list(range(n_use))
subset  = Subset(full_ds, indices)
loader  = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=0, pin_memory=True, drop_last=True)
print(f"Using {n_use} images for smoke test\n")

# ==================================================================
# OPTIMISER + SCHEDULER
# ==================================================================
optimizer     = torch.optim.AdamW(
    [p for p in unet.parameters() if p.requires_grad], lr=LR)
total_steps   = math.ceil(n_use / (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
lr_lambda     = lambda s: (s / max(1, LR_WARMUP) if s < LR_WARMUP else
                           max(0.0, 0.5*(1+math.cos(math.pi*(s-LR_WARMUP)/
                                         max(1,total_steps-LR_WARMUP)))))
sched  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.cuda.amp.GradScaler(enabled=(MIXED_PREC == "fp16"))

# ==================================================================
# TRAINING LOOP
# ==================================================================
diff_log = []; cult_log = []; step_log = []
global_step = 0
optimizer.zero_grad()
t_start = time.time()

print(f"{'='*58}")
print(f" Smoke-test training: {EPOCHS} epochs × {len(loader)} batches")
print(f"{'='*58}\n")

for epoch in range(1, EPOCHS + 1):
    unet.train()
    for step, (images, captions) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)):
        images = images.to(DEVICE)

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

        noise     = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_sched.config.num_train_timesteps,
                                  (latents.shape[0],), device=DEVICE).long()
        noisy_lat = noise_sched.add_noise(latents, noise, timesteps)

        (ids1,m1),(ids2,m2) = tokenise(tokenizer1, tokenizer2, list(captions), DEVICE)
        prompt_emb, pooled_emb = encode_text(text_enc1, text_enc2, ids1,m1,ids2,m2)

        bs = images.shape[0]
        add_time_ids = torch.tensor(
            [[RESOLUTION,RESOLUTION,0,0,RESOLUTION,RESOLUTION]]*bs,
            dtype=torch.float32, device=DEVICE)
        added_cond = {"text_embeds": pooled_emb, "time_ids": add_time_ids}

        with torch.cuda.amp.autocast(enabled=(MIXED_PREC == "fp16")):
            pred = unet(noisy_lat, timesteps,
                        encoder_hidden_states=prompt_emb,
                        added_cond_kwargs=added_cond).sample
            diff_loss = F.mse_loss(pred.float(), noise.float())

            ac = noise_sched.alphas_cumprod[timesteps].sqrt().view(-1,1,1,1)
            pred_lat = (noisy_lat - pred * ac) / vae.config.scaling_factor
            with torch.no_grad():
                pred_img = vae.decode(pred_lat.to(vae.dtype)).sample.float()
            ref_img   = images[torch.randperm(bs, device=DEVICE)].float()
            cult_loss = style_loss_fn(pred_img, ref_img)

            total = (diff_loss + CULTURAL_W * cult_loss) / GRAD_ACCUM

        scaler.scale(total).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in unet.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer); scaler.update(); sched.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % LOG_STEPS == 0:
                diff_log.append(diff_loss.item())
                cult_log.append(cult_loss.item())
                step_log.append(global_step)
                elapsed = time.time() - t_start
                eta     = (total_steps - global_step) / max(global_step/elapsed, 1e-6)
                print(f"  step {global_step:>3}/{total_steps}  "
                      f"diff={diff_loss.item():.4f}  "
                      f"cult={cult_loss.item():.4f}  "
                      f"ETA {eta/60:.1f} min")

    print(f"  Epoch {epoch}/{EPOCHS} done\n")

# -- Save smoke-test weights ---------------------------------------
unet.save_pretrained(OUTPUT_DIR)
elapsed_min = (time.time() - t_start) / 60
print(f"Smoke-test LoRA saved -> {OUTPUT_DIR}")
print(f"Training time: {elapsed_min:.1f} min\n")

# -- Loss curve ----------------------------------------------------
if step_log:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(step_log, diff_log, label="diffusion", color="steelblue")
    ax.plot(step_log, cult_log, label="cultural",  color="darkorange")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Smoke-Test Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    lp = LOG_DIR / "test_loss.png"
    plt.savefig(lp, dpi=150)
    print(f"Loss curve -> {lp}")

# ==================================================================
# GENERATE 4 SAMPLE IMAGES
# ==================================================================
print("\nGenerating 4 sample images ...")
PROMPTS = [
    ("Dong-Son",
     "Vietnamese ancient Bronze Age ornamental pattern, geometric Dong-Son motif, "
     "traditional Dai Viet art style, black and white line art"),
    ("Ly-Tran",
     "Vietnamese Ly-Tran dynasty ornamental pattern, floral lotus motif, "
     "traditional Dai Viet art style, black and white line art"),
    ("Le",
     "Vietnamese Le dynasty ornamental pattern, zoomorphic dragon motif, "
     "traditional Dai Viet art style, black and white line art"),
    ("Nguyen",
     "Vietnamese Nguyen dynasty ornamental pattern, phoenix motif, "
     "traditional Dai Viet art style, black and white line art"),
]

# Build pipeline from components already in memory
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL,
    unet=unet,
    vae=vae,
    text_encoder=text_enc1,
    text_encoder_2=text_enc2,
    tokenizer=tokenizer1,
    tokenizer_2=tokenizer2,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=False)

sample_paths = []
for label, prompt in PROMPTS:
    out_path = SAMPLE_DIR / f"sample_{label}.png"
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    image = pipe(
        prompt=prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        height=512, width=512,
        generator=generator,
    ).images[0]
    image.save(out_path)
    sample_paths.append((label, out_path))
    print(f"  Saved: {out_path}")

# -- 2×2 comparison grid -------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, (label, path) in zip(axes.flat, sample_paths):
    img = Image.open(path)
    ax.imshow(img)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.axis("off")
fig.suptitle("DaiViet LoRA — Smoke-Test Samples", fontsize=14, fontweight="bold")
plt.tight_layout()
grid_path = SAMPLE_DIR / "samples_grid.png"
plt.savefig(grid_path, dpi=150)
print(f"\nGrid saved -> {grid_path}")
print(f"\nSmoke test complete in {elapsed_min:.1f} min. Ready for review.\n")
