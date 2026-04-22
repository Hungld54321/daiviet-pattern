"""
prepare_benchmark_data.py — Task 4.1: Xử lý dataset ĐạiViệt-Pattern v1 cho benchmark
=======================================================================================

Mục tiêu:
  1. Lọc ảnh: line_art + single-motif (loại multi-motif + colored)
  2. Gán caption với TRIGGER WORDS theo thời kỳ (Phương án A — 1 model chung)
  3. Chia train/test = 80/20 (stratified by period, seed cố định)
  4. Resize sang 2 resolution: 512×512 (SD 1.5) và 768×768 (SDXL/FLUX)
  5. Tổ chức thư mục theo dataset variants (D_all + D_per_period)
  6. Xuất metadata JSON + báo cáo thống kê

Output structure:
  benchmark_data/
  ├── metadata.json                 # Full metadata + split info
  ├── stats_report.txt              # Báo cáo thống kê chi tiết
  │
  ├── 512/                          # Resolution 512×512 (cho SD 1.5)
  │   ├── D_all/
  │   │   ├── train/
  │   │   │   ├── images/           # *.png
  │   │   │   └── captions/         # *.txt (trigger word + caption)
  │   │   └── test/
  │   │       ├── images/
  │   │       └── captions/
  │   ├── D_dongson/
  │   │   ├── train/ ...
  │   │   └── test/ ...
  │   ├── D_lytran/ ...
  │   ├── D_le/ ...
  │   └── D_nguyen/ ...
  │
  └── 768/                          # Resolution 768×768 (cho SDXL/FLUX)
      └── (same structure as 512/)

Cách chạy:
  python prepare_benchmark_data.py                          # Dùng đường dẫn mặc định
  python prepare_benchmark_data.py --base_dir /path/to/repo # Tùy chỉnh đường dẫn

Tác giả: Hùng (NCS UIT) — Task 4.1 cho TS. Hải
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

os.environ["PYTHONIOENCODING"] = "utf-8"

import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 42
TEST_SIZE = 0.2  # 80/20 split
RESOLUTIONS = [512, 768]

# Period → subfolder on disk
PERIOD_FOLDER = {
    "Co-Dai":  "co_dai",
    "Le":      "le",
    "Ly-Tran": "ly_tran",
    "Moi":     "moi",
    "Nguyen":  "nguyen",
}

# Period → trigger word (Phương án A: 1 model chung, trigger words phân biệt)
TRIGGER_WORDS = {
    "Co-Dai":  "dong_son_style",
    "Le":      "le_style",
    "Ly-Tran": "ly_tran_style",
    "Moi":     "moi_style",        # Ethnic minority — giữ riêng
    "Nguyen":  "nguyen_style",
}

# Period → human-readable label cho caption
PERIOD_LABELS = {
    "Co-Dai":  "Dong Son Bronze Age",
    "Le":      "Le dynasty",
    "Ly-Tran": "Ly-Tran dynasty",
    "Moi":     "ethnic minority",
    "Nguyen":  "Nguyen dynasty",
}

# Dataset variant → periods to include
DATASET_VARIANTS = {
    "D_all":      ["Co-Dai", "Le", "Ly-Tran", "Nguyen", "Moi"],
    "D_dongson":  ["Co-Dai"],
    "D_lytran":   ["Ly-Tran"],
    "D_le":       ["Le"],
    "D_nguyen":   ["Nguyen"],
}


# ============================================================================
# HELPERS
# ============================================================================
def safe(val, fallback="unknown"):
    """Return cleaned string, or fallback if NaN/empty."""
    if pd.isna(val) or str(val).strip() == "":
        return fallback
    return str(val).strip()


def build_caption(row, use_trigger=True):
    """
    Build caption for an image row from manifest.
    
    Format with trigger word (for training):
      "dong_son_style, Vietnamese Dong Son Bronze Age ornamental pattern, 
       Trống Hoàng Hạ, traditional Dai Viet art, black and white line art"
    
    Format without trigger word (for vanilla baseline prompts):
      "Vietnamese Dong Son Bronze Age ornamental pattern, Trống Hoàng Hạ, 
       traditional Dai Viet art, black and white line art"
    """
    period = safe(row["period"], "Vietnamese")
    subject = safe(row["motif_subject"], "ornamental pattern")
    period_label = PERIOD_LABELS.get(period, period)
    trigger = TRIGGER_WORDS.get(period, "vietnamese_style")
    
    # Rút gọn subject nếu quá dài (>80 chars)
    if len(subject) > 80:
        subject = subject[:77] + "..."
    
    base_caption = (
        f"Vietnamese {period_label} ornamental pattern, "
        f"{subject}, "
        f"traditional Dai Viet art, black and white line art, "
        f"high quality, detailed"
    )
    
    if use_trigger:
        return f"{trigger}, {base_caption}"
    return base_caption


def build_baseline_prompt(period):
    """Build standardized prompt for vanilla model inference (no trigger word)."""
    period_label = PERIOD_LABELS.get(period, "Vietnamese")
    return (
        f"Vietnamese {period_label} traditional ornamental pattern, "
        f"black and white line art, high quality, detailed, "
        f"traditional Dai Viet art style"
    )


def resize_with_padding(img, target_size, bg_color=(255, 255, 255)):
    """Resize image to target_size×target_size with white padding (aspect-preserving)."""
    w, h = img.size
    maxdim = max(w, h)
    canvas = Image.new("RGB", (maxdim, maxdim), bg_color)
    canvas.paste(img, ((maxdim - w) // 2, (maxdim - h) // 2))
    return canvas.resize((target_size, target_size), Image.LANCZOS)


def compute_image_hash(img_path):
    """SHA-256 hash of image file for deduplication check."""
    h = hashlib.sha256()
    with open(img_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def find_image(filename, period, dataset_dir):
    """Find image file on disk, checking period folder first, then fallback."""
    folder = PERIOD_FOLDER.get(period)
    if folder:
        p = dataset_dir / folder / filename
        if p.exists():
            return p
    for folder in PERIOD_FOLDER.values():
        p = dataset_dir / folder / filename
        if p.exists():
            return p
    return None


# ============================================================================
# MAIN PROCESSING
# ============================================================================
def process_dataset(base_dir: Path, output_dir: Path):
    """Main processing pipeline."""
    
    manifest_path = base_dir / "dataset_manifest.csv"
    dataset_dir = base_dir / "dataset"
    
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        sys.exit(1)
    
    # ------------------------------------------------------------------
    # Step 1: Load & filter manifest
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Load & filter manifest")
    print("=" * 60)
    
    df = pd.read_csv(manifest_path)
    total_raw = len(df)
    print(f"  Raw manifest: {total_raw} rows")
    
    # Filter: line_art only
    df = df[df["style"].str.strip().str.lower() == "line_art"]
    after_style = len(df)
    print(f"  After line_art filter: {after_style} rows (-{total_raw - after_style} colored)")
    
    # Filter: single-motif only
    df = df[df["multi_motif"].astype(str).str.strip().str.lower() == "false"]
    after_motif = len(df)
    print(f"  After single-motif filter: {after_motif} rows (-{after_style - after_motif} multi-motif)")
    
    if len(df) == 0:
        print("ERROR: no images remain after filtering.")
        sys.exit(1)
    
    # Verify images exist on disk
    valid_rows = []
    missing = []
    for _, row in df.iterrows():
        img_path = find_image(safe(row["filename"]), safe(row["period"]), dataset_dir)
        if img_path is not None:
            valid_rows.append({**row.to_dict(), "_img_path": str(img_path)})
        else:
            missing.append(safe(row["filename"]))
    
    df_valid = pd.DataFrame(valid_rows)
    print(f"  After disk verification: {len(df_valid)} rows ({len(missing)} missing)")
    if missing:
        print(f"  Missing files: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    print(f"\n  Period distribution:")
    for period, count in df_valid["period"].value_counts().items():
        trigger = TRIGGER_WORDS.get(period, "?")
        print(f"    {period:<12} {count:>4} images  → trigger: {trigger}")
    
    # ------------------------------------------------------------------
    # Step 2: Quality checks
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 2: Quality checks")
    print("=" * 60)
    
    # Check image dimensions and identify potential issues
    dim_stats = defaultdict(int)
    small_images = []
    corrupt_images = []
    
    for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="  Checking quality"):
        try:
            img = Image.open(row["_img_path"])
            w, h = img.size
            dim_stats[f"{w}x{h}"] += 1
            if min(w, h) < 128:
                small_images.append((safe(row["filename"]), w, h))
        except Exception as e:
            corrupt_images.append((safe(row["filename"]), str(e)))
    
    print(f"  Corrupt images: {len(corrupt_images)}")
    for name, err in corrupt_images:
        print(f"    ✗ {name}: {err}")
    
    print(f"  Very small images (<128px): {len(small_images)}")
    for name, w, h in small_images[:5]:
        print(f"    ⚠ {name}: {w}×{h}")
    
    # Top 5 resolutions
    print(f"  Top 5 image dimensions:")
    for dim, count in sorted(dim_stats.items(), key=lambda x: -x[1])[:5]:
        print(f"    {dim}: {count} images")
    
    # Remove corrupt images
    if corrupt_images:
        corrupt_names = {name for name, _ in corrupt_images}
        df_valid = df_valid[~df_valid["filename"].isin(corrupt_names)]
        print(f"  → Removed {len(corrupt_names)} corrupt images. Remaining: {len(df_valid)}")
    
    # ------------------------------------------------------------------
    # Step 3: Stratified train/test split
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"STEP 3: Train/Test split (test_size={TEST_SIZE}, seed={SEED})")
    print("=" * 60)
    
    # Stratified split by period
    train_df, test_df = train_test_split(
        df_valid,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df_valid["period"]
    )
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Test:  {len(test_df)} images")
    print(f"\n  Split distribution:")
    print(f"  {'Period':<12} {'Train':>6} {'Test':>6} {'Total':>6} {'Test%':>6}")
    print(f"  {'-'*42}")
    for period in sorted(df_valid["period"].unique()):
        n_train = len(train_df[train_df["period"] == period])
        n_test = len(test_df[test_df["period"] == period])
        total = n_train + n_test
        pct = n_test / total * 100 if total > 0 else 0
        print(f"  {period:<12} {n_train:>6} {n_test:>6} {total:>6} {pct:>5.1f}%")
    
    # ------------------------------------------------------------------
    # Step 4: Process images & write to disk
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 4: Process images & write to disk")
    print("=" * 60)
    
    # Build lookup: filename → (split, row)
    records = []
    for _, row in train_df.iterrows():
        records.append(("train", row))
    for _, row in test_df.iterrows():
        records.append(("test", row))
    
    # Process for each resolution
    for res in RESOLUTIONS:
        print(f"\n  --- Resolution: {res}×{res} ---")
        
        # Create directory structure for all variants
        for variant_name, variant_periods in DATASET_VARIANTS.items():
            for split in ["train", "test"]:
                (output_dir / str(res) / variant_name / split / "images").mkdir(parents=True, exist_ok=True)
                (output_dir / str(res) / variant_name / split / "captions").mkdir(parents=True, exist_ok=True)
        
        processed = 0
        for split, row in tqdm(records, desc=f"  Resize → {res}px"):
            filename = safe(row["filename"])
            period = safe(row["period"])
            stem = Path(filename).stem
            
            try:
                img = Image.open(row["_img_path"]).convert("RGB")
            except Exception:
                continue
            
            resized = resize_with_padding(img, res)
            
            # Caption with trigger word
            caption_trigger = build_caption(row, use_trigger=True)
            # Caption without trigger word (for reference / vanilla baseline)
            caption_plain = build_caption(row, use_trigger=False)
            
            # Write to each applicable variant
            for variant_name, variant_periods in DATASET_VARIANTS.items():
                if period in variant_periods:
                    img_out = output_dir / str(res) / variant_name / split / "images" / f"{stem}.png"
                    cap_out = output_dir / str(res) / variant_name / split / "captions" / f"{stem}.txt"
                    
                    resized.save(img_out, "PNG")
                    cap_out.write_text(caption_trigger, encoding="utf-8")
            
            processed += 1
        
        print(f"  Processed: {processed} images → {res}×{res}")
    
    # ------------------------------------------------------------------
    # Step 5: Generate baseline prompts (for M1: SDXL vanilla)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 5: Generate baseline prompts")
    print("=" * 60)
    
    prompts_dir = output_dir / "baseline_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_prompts = {}
    for period in sorted(PERIOD_LABELS.keys()):
        prompt = build_baseline_prompt(period)
        baseline_prompts[period] = prompt
        print(f"  {period:<12}: {prompt[:70]}...")
    
    # Save prompts
    with open(prompts_dir / "baseline_prompts.json", "w", encoding="utf-8") as f:
        json.dump(baseline_prompts, f, indent=2, ensure_ascii=False)
    
    # Also save per-period prompt files for easy scripting
    for period, prompt in baseline_prompts.items():
        trigger = TRIGGER_WORDS.get(period, "vietnamese_style")
        with open(prompts_dir / f"prompts_{trigger}.txt", "w", encoding="utf-8") as f:
            f.write(f"# Baseline prompt (no trigger word) for M1:\n")
            f.write(f"{prompt}\n\n")
            f.write(f"# Training prompt (with trigger word) for M2/M3/M4/M6:\n")
            f.write(f"{trigger}, {prompt}\n")
    
    print(f"  Saved to: {prompts_dir}")
    
    # ------------------------------------------------------------------
    # Step 6: Generate metadata JSON
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 6: Generate metadata")
    print("=" * 60)
    
    metadata = {
        "dataset_name": "DaiViet-Pattern Benchmark v1.0",
        "description": "Processed dataset for Task 4.1 benchmark — comparing generative models on Vietnamese ornamental patterns",
        "created": datetime.now().isoformat(),
        "source": "ĐạiViệt-Pattern dataset v1 (hoavandaiviet.vn)",
        "github": "https://github.com/Hungld54321/daiviet-pattern",
        "seed": SEED,
        "test_size": TEST_SIZE,
        "resolutions": RESOLUTIONS,
        "filter_criteria": {
            "style": "line_art",
            "multi_motif": False,
        },
        "trigger_words": TRIGGER_WORDS,
        "total_images": len(df_valid),
        "train_count": len(train_df),
        "test_count": len(test_df),
        "period_distribution": {},
        "variants": {},
        "train_files": [],
        "test_files": [],
    }
    
    # Period distribution
    for period in sorted(df_valid["period"].unique()):
        n_train = len(train_df[train_df["period"] == period])
        n_test = len(test_df[test_df["period"] == period])
        metadata["period_distribution"][period] = {
            "total": n_train + n_test,
            "train": n_train,
            "test": n_test,
            "trigger_word": TRIGGER_WORDS.get(period, "?"),
        }
    
    # Variant stats
    for variant_name, variant_periods in DATASET_VARIANTS.items():
        v_train = len(train_df[train_df["period"].isin(variant_periods)])
        v_test = len(test_df[test_df["period"].isin(variant_periods)])
        metadata["variants"][variant_name] = {
            "periods": variant_periods,
            "train": v_train,
            "test": v_test,
            "total": v_train + v_test,
        }
    
    # File lists (for reproducibility)
    for _, row in train_df.iterrows():
        metadata["train_files"].append({
            "filename": safe(row["filename"]),
            "period": safe(row["period"]),
            "motif_subject": safe(row["motif_subject"], ""),
        })
    for _, row in test_df.iterrows():
        metadata["test_files"].append({
            "filename": safe(row["filename"]),
            "period": safe(row["period"]),
            "motif_subject": safe(row["motif_subject"], ""),
        })
    
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {meta_path}")
    
    # ------------------------------------------------------------------
    # Step 7: Statistics report
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 7: Statistics report")
    print("=" * 60)
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("TASK 4.1 — BENCHMARK DATASET REPORT")
    report_lines.append(f"ĐạiViệt-Pattern v1.0 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    report_lines.append("1. SOURCE")
    report_lines.append(f"   Raw manifest: {total_raw} images")
    report_lines.append(f"   After filter (line_art + single-motif): {after_motif}")
    report_lines.append(f"   After disk verification: {len(df_valid)}")
    report_lines.append(f"   Corrupt/removed: {len(corrupt_images)}")
    report_lines.append(f"   Missing on disk: {len(missing)}")
    report_lines.append("")
    
    report_lines.append("2. TRAIN/TEST SPLIT")
    report_lines.append(f"   Strategy: stratified by period, test_size={TEST_SIZE}, seed={SEED}")
    report_lines.append(f"   Train: {len(train_df)} images ({len(train_df)/len(df_valid)*100:.1f}%)")
    report_lines.append(f"   Test:  {len(test_df)} images ({len(test_df)/len(df_valid)*100:.1f}%)")
    report_lines.append("")
    
    report_lines.append("   Period        Train   Test  Total  Test%")
    report_lines.append("   " + "-" * 48)
    for period in sorted(df_valid["period"].unique()):
        n_train = len(train_df[train_df["period"] == period])
        n_test = len(test_df[test_df["period"] == period])
        total = n_train + n_test
        pct = n_test / total * 100 if total > 0 else 0
        trigger = TRIGGER_WORDS.get(period, "?")
        report_lines.append(f"   {period:<12} {n_train:>5} {n_test:>6} {total:>6} {pct:>5.1f}%  [{trigger}]")
    report_lines.append("")
    
    report_lines.append("3. DATASET VARIANTS")
    for variant_name, variant_periods in DATASET_VARIANTS.items():
        v_train = metadata["variants"][variant_name]["train"]
        v_test = metadata["variants"][variant_name]["test"]
        report_lines.append(f"   {variant_name:<14} train={v_train}, test={v_test}, total={v_train+v_test}")
        report_lines.append(f"                  periods: {', '.join(variant_periods)}")
    report_lines.append("")
    
    report_lines.append("4. RESOLUTIONS")
    for res in RESOLUTIONS:
        report_lines.append(f"   {res}×{res}: all variants prepared")
    report_lines.append("")
    
    report_lines.append("5. CAPTION FORMAT (with trigger word)")
    sample_row = df_valid.iloc[0]
    report_lines.append(f"   Example: {build_caption(sample_row, use_trigger=True)}")
    report_lines.append("")
    
    report_lines.append("6. TRIGGER WORDS")
    for period, trigger in sorted(TRIGGER_WORDS.items()):
        report_lines.append(f"   {period:<12} → {trigger}")
    report_lines.append("")
    
    report_lines.append("7. MODELS TO BENCHMARK (from Task 4.1 spec)")
    report_lines.append("   M1: SDXL 1.0 vanilla     — no fine-tune, baseline_prompts.json")
    report_lines.append("   M2: SDXL + LoRA           — 768/D_all/train, rank=16, lora_alpha=32")
    report_lines.append("   M3: SD 1.5 + LoRA         — 512/D_all/train, rank=16")
    report_lines.append("   M4: FLUX.1-dev + LoRA     — 512/D_all/train (bonus)")
    report_lines.append("   M5: StyleGAN2-ADA         — 512/D_all/train (bonus)")
    report_lines.append("   M6: SDXL + LoRA + L_cult  — 768/D_all/train, λ=0.3")
    report_lines.append("")
    
    report_lines.append("8. EVALUATION PROTOCOL")
    report_lines.append("   Generate: 50 images/model × 4 periods = 200 images/model")
    report_lines.append("   Compare against: test set (per-period)")
    report_lines.append("   Metrics: FID ↓ | CLIP Score ↑ | SSIM ↑ | LPIPS ↓ | PSNR ↑")
    report_lines.append("   Note: FID on N=50 is indicative, N≥2048 for journal version")
    report_lines.append("")
    
    report_lines.append("9. PRIOR RESULTS (from ĐạiViệt-Pattern paper, N=20)")
    report_lines.append("   Method                    FID↓     SSIM    CLIP↑")
    report_lines.append("   SDXL vanilla              368.06   0.1255  0.2888")
    report_lines.append("   SDXL+LoRA+L_cultural      336.07   0.1113  0.3135")
    report_lines.append("   → These serve as reference; will be recomputed on new splits")
    report_lines.append("")
    
    report_lines.append("=" * 60)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    report_path = output_dir / "stats_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    
    print(f"\n  Report saved: {report_path}")
    print(f"  Metadata saved: {meta_path}")
    print(f"  Output directory: {output_dir}")
    print(f"\nDone! Dataset ready for benchmark training.")


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare ĐạiViệt-Pattern v1 dataset for Task 4.1 benchmark"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory of the daiviet-pattern repo. "
             "Default: auto-detect (current dir or D:/Hoavandaiviet)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: <base_dir>/benchmark_data"
    )
    args = parser.parse_args()
    
    # Auto-detect base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    elif Path("dataset_manifest.csv").exists():
        base_dir = Path(".")
    elif Path("D:/Hoavandaiviet/dataset_manifest.csv").exists():
        base_dir = Path("D:/Hoavandaiviet")
    else:
        print("ERROR: Cannot find dataset_manifest.csv.")
        print("Run from the repo root or pass --base_dir")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "benchmark_data"
    
    print(f"Base dir:   {base_dir.resolve()}")
    print(f"Output dir: {output_dir.resolve()}")
    print("")
    
    process_dataset(base_dir, output_dir)
