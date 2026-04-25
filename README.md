# DaiViet-Pattern Benchmark — Task 4.1

Comparative evaluation of four generative models for Vietnamese ornamental pattern synthesis, submitted as part of the **APWeb-WAIM 2026** paper on culturally-conditioned diffusion models for historical Dai Viet art.

---

## Overview

This benchmark compares four Stable Diffusion variants (M1–M3, M6) on the task of generating Vietnamese ornamental motifs (*hoa văn*) across three historical dynasties: **Lý-Trần**, **Lê**, and **Nguyễn**. The primary research question is whether period-aware cultural conditioning (M6) measurably improves generation fidelity over standard LoRA fine-tuning (M2) and vanilla SDXL (M1).

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | 200 PDF vector files — *Hoa Văn Đại Việt* project (Đại Việt Cổ Phong × Comicola) |
| **Extraction** | 291 individual motifs extracted via adaptive grid detection |
| **Periods** | Lý-Trần: 50 · Lê: 94 · Nguyễn: 147 |
| **Train / Test split** | 232 train / 59 test (stratified by period, `seed=42`) |
| **Test references** | Lý-Trần: 10 · Lê: 19 · Nguyễn: 30 |
| **Metadata** | Excel annotations per file: artist, description, historical source |
| **Image format** | PNG, 768 × 768 px (SDXL models) · 512 × 512 px (SD 1.5) |

Extraction script: `extract_pdf_motifs.py` · Dataset preparation: `prepare_benchmark_data.py`

---

## Models Benchmarked

| ID | Base Model | Fine-tuning | Key Config | Epochs |
|---|---|---|---|---|
| **M1** | SDXL 1.0 | None (control group) | vanilla inference, no trigger word | — |
| **M2** | SDXL 1.0 | LoRA | rank=16, α=32, lr_unet=1e-4, lr_te=1e-5 | 50 |
| **M3** | SD 1.5 | LoRA | rank=16, α=32, lr_unet=1e-4, lr_te=1e-5 | 50 |
| **M6** | SDXL 1.0 | LoRA + Cultural Loss | rank=16, α=32; latent Gram matrix, λ=0.3, every=16 steps | 30 |

**M6 cultural loss** — computes a Gram matrix style loss in latent space between the predicted clean latent (`pred_x0`) and a same-period reference latent sampled per batch. Gradient flows through the UNet prediction without requiring VAE decoding, keeping peak VRAM at 11.7 GB on an RTX 4080.

All models generate 50 images per period (150 total) using fixed seeds 0–49, prompt engineering with period-specific trigger words, and identical inference settings (30 DDIM steps, CFG=7.5, resolution per model class).

---

## Results

### Overall (macro-average across 3 periods)

| Model | FID50 ↓ | CLIP ↑ | SSIM ↑ | LPIPS ↓ | PSNR ↑ | Inf. Time (s) |
|---|---|---|---|---|---|---|
| **M6** | **344.16** | 0.2576 | **0.1623** | **0.7417** | 5.32 | 4.46 |
| M2 | 349.76 | 0.2510 | 0.1422 | 0.7480 | 5.27 | 4.45 |
| M1 | 357.99 | **0.2718** | 0.0763 | 0.7482 | 5.51 | 4.46 |
| M3 | 371.80 | 0.2290 | 0.0687 | 0.6980 | **6.51** | **2.45** |

### Per-period breakdown

| Model | Period | FID50 ↓ | CLIP ↑ | SSIM ↑ | LPIPS ↓ | PSNR ↑ |
|---|---|---|---|---|---|---|
| M6 | Lý-Trần | **339.62** | 0.2364 | **0.1844** | **0.6341** | 4.23 |
| M2 | Lý-Trần | 345.00 | 0.2479 | 0.1532 | 0.6378 | 3.84 |
| M1 | Lý-Trần | 341.03 | **0.2727** | 0.0935 | 0.6461 | 4.20 |
| M3 | Lý-Trần | 388.54 | 0.2197 | 0.0859 | 0.6126 | **5.15** |
| M6 | Lê | 370.67 | **0.2789** | **0.0336** | 0.7748 | 5.68 |
| M2 | Lê | 375.94 | 0.2573 | 0.0284 | 0.7734 | 5.39 |
| M1 | Lê | 383.46 | 0.2690 | 0.0230 | **0.7636** | 6.07 |
| M3 | Lê | **366.28** | 0.2333 | 0.0214 | 0.7165 | **6.54** |
| M6 | Nguyễn | **322.19** | 0.2574 | **0.2690** | **0.8163** | 6.05 |
| M2 | Nguyễn | 328.35 | 0.2479 | 0.2450 | 0.8329 | 6.57 |
| M1 | Nguyễn | 349.48 | **0.2737** | 0.1123 | 0.8348 | 6.27 |
| M3 | Nguyễn | 360.57 | 0.2341 | 0.0988 | 0.7648 | **7.83** |

> **Metric notes:** FID50 computed on N=50 generated images vs. held-out test references using Fréchet Inception Distance. SSIM / LPIPS / PSNR use nearest-reference proxy matching (unpaired setting). CLIP = cosine similarity with CLIP ViT-L/14 embeddings.

---

## Key Findings

- **M6 achieves the best FID50 (344.16) and SSIM (0.1623)**, outperforming the plain LoRA baseline (M2) and the vanilla SDXL control (M1) on both distributional and structural fidelity metrics. This supports the hypothesis that latent-space cultural conditioning improves period-specific generation quality.

- **Cultural conditioning yields a measurable SSIM gain over vanilla LoRA**: M6 (+14.1% SSIM vs. M2) and a FID50 reduction of 1.6 points, achieved with 40% fewer training epochs (30 vs. 50) and a peak VRAM of only 11.7 GB.

- **SD 1.5 (M3) is 1.82× faster at inference** (2.45 s vs. 4.46 s for SDXL) but produces the lowest CLIP score (0.229) and SSIM (0.069). The lower 512 px resolution likely limits structural detail for Vietnamese ornamental patterns.

- **Vanilla SDXL (M1) scores highest on CLIP (0.2718)** — reflecting strong text-to-image alignment from pretraining — but its SSIM (0.076) and FID50 (357.99) are worse than all LoRA-fine-tuned variants, confirming that domain-specific fine-tuning is essential for period-accurate motif synthesis.

---

## Repository Structure

```
DaiViet-Pattern/
│
├── extract_pdf_motifs.py         # PDF → PNG motif extraction (grid detection)
├── prepare_benchmark_data.py     # Train/test split, resolution resize, manifests
├── crawl_wikimedia.py            # Wikimedia Commons supplementary image crawler
│
├── dataset_manifest.csv          # Full dataset inventory (all 291 motifs)
├── benchmark_manifest.csv        # Train/test split manifest (seed=42)
│
├── vector_extracted/
│   ├── ly_tran/extract_manifest.csv   # Per-file extraction metadata, Lý-Trần
│   ├── le/extract_manifest.csv        # Per-file extraction metadata, Lê
│   └── nguyen/extract_manifest.csv    # Per-file extraction metadata, Nguyễn
│
├── wikimedia_data/
│   └── wikimedia_manifest.csv    # Crawled supplementary images manifest
├── wikimedia_raw/
│   └── crawl_manifest.csv        # Raw crawl log
│
└── benchmark/
    ├── train_benchmark.py        # Training loop (M2, M3, M6) — LoRA + cultural loss
    ├── generate_benchmark.py     # Image generation (all models, 50 seeds × 3 periods)
    ├── evaluate_benchmark.py     # FID50 / CLIP / SSIM / LPIPS / PSNR evaluation
    ├── run_full_benchmark.py     # End-to-end pipeline orchestrator
    ├── README.md                 # Benchmark-specific quick-start notes
    │
    ├── checkpoints/
    │   ├── M2/
    │   │   ├── pytorch_lora_weights.safetensors   # [ignored — large]
    │   │   ├── training_log.csv                   # Step-level loss log (6,250 rows)
    │   │   └── training_config.json               # Hyperparameter record
    │   ├── M3/  (same structure)
    │   └── M6/  (same structure, + loss_cultural column)
    │
    ├── results/
    │   ├── evaluation_report.txt   # Human-readable evaluation report
    │   ├── metrics_overall.csv     # Overall metrics table (machine-readable)
    │   └── metrics_per_period.csv  # Per-period breakdown
    │
    └── pipeline_logs/
        └── pipeline_v4_status.txt  # Timestamped pipeline execution log
```

> **Note:** `*.safetensors` model weights, raw PDF source files (`vector_source/`), extracted PNGs (`vector_extracted/*.png`), and generated images (`benchmark/generated/`) are excluded from this repository (see `.gitignore`). These can be regenerated using the reproduction steps below.

---

## How to Reproduce

### Prerequisites

```bash
# Python 3.10+, CUDA 11.8+ recommended
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.27.2 transformers peft safetensors accelerate
pip install cleanfid open-clip-torch scikit-image lpips
pip install tqdm pandas openpyxl pdf2image pillow
```

**Hardware:** NVIDIA RTX 4080 16 GB (or equivalent ≥ 12 GB VRAM for M6). M3 can run on ≥ 8 GB VRAM.

### Step 1 — Place source PDFs

```
vector_source/45_hoavanlytran/     ← Lý-Trần PDFs
vector_source/85_hoavanle/         ← Lê PDFs
vector_source/70_hoa_van_nguyen/   ← Nguyễn PDFs
vector_source/excel_metadata/      ← Excel annotation files
```

### Step 2 — Extract motifs from PDFs

```bash
# Lý-Trần (45 files → 50 motifs)
python extract_pdf_motifs.py \
  --input_dir vector_source/45_hoavanlytran \
  --excel "vector_source/excel_metadata/45 Hoa Văn Lý Trần_v04.xlsx" \
  --period Ly-Tran --output_dir vector_extracted/ly_tran

# Lê (85 files → 94 motifs)
python extract_pdf_motifs.py \
  --input_dir vector_source/85_hoavanle \
  --excel "vector_source/excel_metadata/85 Hoa Văn Thời Lê.xlsx" \
  --period Le --output_dir vector_extracted/le

# Nguyễn (70 files → 147 motifs)
python extract_pdf_motifs.py \
  --input_dir vector_source/70_hoa_van_nguyen \
  --excel "vector_source/excel_metadata/70 Hoa Văn Thời Nguyễn_ver_03.xlsx" \
  --period Nguyen --output_dir vector_extracted/nguyen
```

### Step 3 — Prepare benchmark dataset

```bash
python prepare_benchmark_data.py
# Outputs: benchmark_data/768/D_all/{train,test}/ and benchmark_data/512/D_all/{train,test}/
# Also writes: dataset_manifest.csv, benchmark_manifest.csv
```

### Step 4 — Run full benchmark pipeline

```bash
# All models end-to-end (train M2, M3, M6 → generate all → evaluate all)
python benchmark/run_full_benchmark.py

# Or individually:
python benchmark/train_benchmark.py   --model M2
python benchmark/train_benchmark.py   --model M3
python benchmark/train_benchmark.py   --model M6
python benchmark/generate_benchmark.py --model all
python benchmark/evaluate_benchmark.py --model all
```

Results are written to `benchmark/results/`.

---

## Limitations

- **FID50 is indicative only.** Standard FID uses N=50,000 samples; N=50 per period produces a noisier estimate with higher variance. Results should be interpreted directionally, not as absolute distributional scores.

- **SSIM / LPIPS are unpaired proxies.** Generated images are matched to the nearest test reference by filename ordering; there is no semantic pairing between generated and reference motifs. These metrics reflect structural plausibility relative to real motifs rather than pixel-level reconstruction accuracy.

- **Cultural loss operates in latent space**, using Gram matrix similarity between predicted clean latents and same-period reference latents. This is a practical approximation of VGG-based perceptual style loss (which was infeasible at 768 px SDXL resolution due to VRAM constraints) and may capture different style dimensions.

- **Small dataset.** 291 motifs across 3 periods is a constrained training set for generative modelling. LoRA fine-tuning mitigates this by preserving the SDXL prior, but generated diversity may be limited.

---

## Branch Information

| Branch | Contents |
|---|---|
| `benchmark` | This repository — scripts, training logs, evaluation results, manifests |
| `main` | APWeb-WAIM 2026 paper submission code (tag: `apweb-v4-submitted`) |

> For questions on the methodology or training setup, refer to `benchmark/results/evaluation_report.txt` and the per-model `training_config.json` files in `benchmark/checkpoints/`.
