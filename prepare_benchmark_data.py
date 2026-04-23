#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_benchmark_data.py — Task 4.1: Chuẩn bị benchmark dataset từ PDF vector
=================================================================================

Nguồn dữ liệu MỚI: vector_extracted/ (291 motifs trích từ PDF gốc)
  - vector_extracted/ly_tran/   — 50 motifs + extract_manifest.csv
  - vector_extracted/le/        — 94 motifs + extract_manifest.csv
  - vector_extracted/nguyen/    — 147 motifs + extract_manifest.csv

KHÔNG dùng dataset/ cũ (791 ảnh web-scrape).

Cách chạy:
  python prepare_benchmark_data.py
  python prepare_benchmark_data.py --base_dir D:/Hoavandaiviet --output_dir benchmark_data/

Tác giả: Hùng (NCS UIT) — Task 4.1 Benchmark
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

Image.MAX_IMAGE_PIXELS = None

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED      = 42
TEST_SIZE = 0.2
RESOLUTIONS = [512, 768]

# Subfolder trong vector_extracted/ cho từng period
PERIOD_FOLDER = {
    "Ly-Tran": "ly_tran",
    "Le":      "le",
    "Nguyen":  "nguyen",
}

# Trigger words (Phương án A — 1 model chung)
TRIGGER_WORDS = {
    "Ly-Tran": "ly_tran_style",
    "Le":      "le_style",
    "Nguyen":  "nguyen_style",
}

# Label tiếng Anh cho caption
PERIOD_LABELS = {
    "Ly-Tran": "Ly-Tran dynasty",
    "Le":      "Le dynasty",
    "Nguyen":  "Nguyen dynasty",
}

# Dataset variants
# D_all      = toàn bộ (line_art + colored)
# D_line_art = chỉ line_art
# D_lytran / D_le / D_nguyen = per-period
DATASET_VARIANTS = {
    "D_all":      {"periods": ["Ly-Tran", "Le", "Nguyen"], "style_filter": None},
    "D_line_art": {"periods": ["Ly-Tran", "Le", "Nguyen"], "style_filter": "line_art"},
    "D_lytran":   {"periods": ["Ly-Tran"],                 "style_filter": None},
    "D_le":       {"periods": ["Le"],                      "style_filter": None},
    "D_nguyen":   {"periods": ["Nguyen"],                  "style_filter": None},
}

MIN_DIM = 64    # Loại ảnh nhỏ hơn ngưỡng này


# ============================================================================
# HELPERS
# ============================================================================
def safe(val, fallback=""):
    """Trả về chuỗi sạch, hoặc fallback nếu NaN/rỗng."""
    if pd.isna(val) or str(val).strip() in ("", "nan", "None"):
        return fallback
    return str(val).strip()


def build_caption(row, use_trigger: bool = True) -> str:
    """
    Tạo caption cho một motif.

    Với trigger word (dùng cho training M2, M3, M6):
      "ly_tran_style, Vietnamese Ly-Tran dynasty ornamental pattern,
       Rồng thời Lý, Cửu đỉnh Thế Miếu, traditional Dai Viet art,
       high quality, detailed"

    Không trigger word (dùng cho baseline M1):
      "Vietnamese Ly-Tran dynasty ornamental pattern, Rồng thời Lý,
       traditional Dai Viet art, high quality, detailed"
    """
    period      = safe(row.get("period", ""), "Vietnamese")
    mieu_ta     = safe(row.get("mieu_ta",  ""), "")
    nguon_goc   = safe(row.get("nguon_goc",""), "")
    style       = safe(row.get("style",    ""), "")

    period_label = PERIOD_LABELS.get(period, period)
    trigger      = TRIGGER_WORDS.get(period, "vietnamese_style")

    # Rút gọn nếu quá dài
    if len(mieu_ta) > 80:
        mieu_ta = mieu_ta[:77] + "..."
    if len(nguon_goc) > 100:
        nguon_goc = nguon_goc[:97] + "..."

    # Phần mô tả phong phú (nếu có metadata)
    desc_parts = []
    if mieu_ta:
        desc_parts.append(mieu_ta)
    if nguon_goc:
        desc_parts.append(nguon_goc)
    desc = ", ".join(desc_parts) if desc_parts else "ornamental pattern"

    style_tag = "black and white line art" if style == "line_art" else "colored ornamental art"

    base = (
        f"Vietnamese {period_label} ornamental pattern, "
        f"{desc}, "
        f"traditional Dai Viet art, {style_tag}, "
        f"high quality, detailed"
    )

    return f"{trigger}, {base}" if use_trigger else base


def build_baseline_prompt(period: str) -> str:
    """Prompt chuẩn không trigger word cho M1 (vanilla inference)."""
    period_label = PERIOD_LABELS.get(period, "Vietnamese")
    return (
        f"Vietnamese {period_label} traditional ornamental pattern, "
        f"black and white line art, high quality, detailed, "
        f"traditional Dai Viet art style"
    )


def pad_and_resize(img: Image.Image, size: int) -> Image.Image:
    """White-pad về vuông rồi resize, giữ nguyên aspect ratio."""
    w, h = img.size
    maxdim = max(w, h)
    canvas = Image.new("RGB", (maxdim, maxdim), (255, 255, 255))
    canvas.paste(img.convert("RGB"), ((maxdim - w) // 2, (maxdim - h) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


# ============================================================================
# STEP 1: Load & merge manifests
# ============================================================================
def load_manifests(base_dir: Path) -> pd.DataFrame:
    print("=" * 60)
    print("BƯỚC 1: Đọc và merge 3 file extract_manifest.csv")
    print("=" * 60)

    extracted_dir = base_dir / "vector_extracted"
    frames = []

    for period, folder in PERIOD_FOLDER.items():
        manifest_path = extracted_dir / folder / "extract_manifest.csv"
        if not manifest_path.exists():
            print(f"  [WARN] Không tìm thấy: {manifest_path}")
            continue

        df = pd.read_csv(manifest_path, encoding="utf-8-sig")
        df["period"] = period   # đảm bảo period đúng
        df["_img_path"] = df["filename"].apply(
            lambda f: str(extracted_dir / folder / f)
        )
        frames.append(df)
        print(f"  {period:<10}: {len(df):>4} rows  ← {manifest_path}")

    if not frames:
        print("ERROR: Không tìm thấy bất kỳ extract_manifest.csv nào.")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  Tổng: {len(merged)} rows từ {len(frames)} thời kỳ")
    return merged


# ============================================================================
# STEP 2: Thống kê & quality check
# ============================================================================
def quality_check(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'=' * 60}")
    print("BƯỚC 2: Thống kê chất lượng dataset")
    print("=" * 60)

    # Phân bổ theo period
    print("\n  Period distribution:")
    for period, grp in df.groupby("period"):
        n_line  = (grp["style"] == "line_art").sum()
        n_color = (grp["style"] == "colored").sum()
        print(f"    {period:<10}: {len(grp):>4} total  "
              f"(line_art={n_line}, colored={n_color})")

    # Phân bổ line_art vs colored
    n_line  = (df["style"] == "line_art").sum()
    n_color = (df["style"] == "colored").sum()
    print(f"\n  Style:  line_art={n_line} ({n_line/len(df)*100:.1f}%)  "
          f"colored={n_color} ({n_color/len(df)*100:.1f}%)")

    # Kiểm tra ảnh trên disk
    print(f"\n  Kiểm tra ảnh trên disk...")
    corrupt, small, missing = [], [], []
    widths, heights = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Quality check"):
        p = Path(row["_img_path"])
        if not p.exists():
            missing.append(row["filename"])
            continue
        try:
            img = Image.open(p)
            w, h = img.size
            widths.append(w)
            heights.append(h)
            if min(w, h) < MIN_DIM:
                small.append((row["filename"], w, h))
        except Exception as e:
            corrupt.append((row["filename"], str(e)))

    print(f"  Không tìm thấy:  {len(missing)}")
    print(f"  Ảnh lỗi:         {len(corrupt)}")
    print(f"  Ảnh quá nhỏ:     {len(small)}  (<{MIN_DIM}px)")

    if widths:
        import numpy as np
        print(f"\n  Kích thước (width):  "
              f"min={min(widths)}  max={max(widths)}  "
              f"mean={int(np.mean(widths))}  median={int(np.median(widths))}")
        print(f"  Kích thước (height): "
              f"min={min(heights)}  max={max(heights)}  "
              f"mean={int(np.mean(heights))}  median={int(np.median(heights))}")

    # Loại bỏ ảnh lỗi và missing
    bad = set(missing) | {f for f, _ in corrupt}
    if bad:
        df = df[~df["filename"].isin(bad)].copy()
        print(f"\n  → Đã loại: {len(bad)} ảnh. Còn lại: {len(df)}")

    return df


# ============================================================================
# STEP 3: Train/Test split
# ============================================================================
def split_dataset(df: pd.DataFrame):
    print(f"\n{'=' * 60}")
    print(f"BƯỚC 3: Train/Test split  "
          f"(stratified by period, test={TEST_SIZE}, seed={SEED})")
    print("=" * 60)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["period"],
    )
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["split"] = "train"
    test_df["split"]  = "test"

    print(f"\n  {'Period':<12} {'Train':>6} {'Test':>5} {'Total':>6} {'Test%':>6}")
    print(f"  {'-'*42}")
    for period in sorted(df["period"].unique()):
        n_tr = (train_df["period"] == period).sum()
        n_te = (test_df["period"]  == period).sum()
        tot  = n_tr + n_te
        print(f"  {period:<12} {n_tr:>6} {n_te:>5} {tot:>6} {n_te/tot*100:>5.1f}%")
    print(f"  {'TOTAL':<12} {len(train_df):>6} {len(test_df):>5} "
          f"{len(df):>6} {len(test_df)/len(df)*100:>5.1f}%")

    return train_df, test_df


# ============================================================================
# STEP 4 & 5: Process images → resize & write captions
# ============================================================================
def process_images(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   output_dir: Path):
    print(f"\n{'=' * 60}")
    print("BƯỚC 4 & 5: Resize ảnh và xuất captions")
    print("=" * 60)

    all_df = pd.concat([train_df, test_df], ignore_index=True)

    for res in RESOLUTIONS:
        print(f"\n  --- Resolution: {res}×{res} ---")

        # Tạo thư mục cho tất cả variants và splits
        for vname, vcfg in DATASET_VARIANTS.items():
            for split in ["train", "test"]:
                (output_dir / str(res) / vname / split / "images").mkdir(
                    parents=True, exist_ok=True)
                (output_dir / str(res) / vname / split / "captions").mkdir(
                    parents=True, exist_ok=True)

        n_written = 0
        for _, row in tqdm(all_df.iterrows(), total=len(all_df),
                           desc=f"  {res}px"):
            period = safe(row["period"])
            style  = safe(row["style"])
            split  = row["split"]
            stem   = Path(row["filename"]).stem

            try:
                img = Image.open(row["_img_path"])
                resized = pad_and_resize(img, res)
            except Exception as e:
                tqdm.write(f"  [WARN] Skip {row['filename']}: {e}")
                continue

            caption_trigger = build_caption(row, use_trigger=True)

            for vname, vcfg in DATASET_VARIANTS.items():
                # Lọc theo period
                if period not in vcfg["periods"]:
                    continue
                # Lọc theo style (nếu variant yêu cầu)
                if vcfg["style_filter"] and style != vcfg["style_filter"]:
                    continue

                img_out = (output_dir / str(res) / vname / split
                           / "images" / f"{stem}.png")
                cap_out = (output_dir / str(res) / vname / split
                           / "captions" / f"{stem}.txt")

                resized.save(img_out, "PNG")
                cap_out.write_text(caption_trigger, encoding="utf-8")

            n_written += 1

        print(f"  Đã xử lý: {n_written} ảnh → {res}×{res}")


# ============================================================================
# STEP 6: Baseline prompts
# ============================================================================
def write_baseline_prompts(output_dir: Path):
    print(f"\n{'=' * 60}")
    print("BƯỚC 6: Xuất baseline prompts")
    print("=" * 60)

    prompts_dir = output_dir / "baseline_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    baseline = {}
    for period in sorted(PERIOD_LABELS):
        prompt = build_baseline_prompt(period)
        baseline[period] = prompt
        print(f"  {period}: {prompt[:70]}...")

    # JSON tổng hợp
    with open(prompts_dir / "baseline_prompts.json", "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)

    # File .txt riêng theo period
    for period, prompt in baseline.items():
        trigger = TRIGGER_WORDS.get(period, "vietnamese_style")
        out = prompts_dir / f"prompts_{trigger}.txt"
        out.write_text(
            f"# Baseline prompt (no trigger word) for M1:\n{prompt}\n\n"
            f"# Training prompt (with trigger word) for M2/M3/M6:\n"
            f"{trigger}, {prompt}\n",
            encoding="utf-8",
        )

    print(f"  Saved: {prompts_dir}")


# ============================================================================
# STEP 7: metadata.json
# ============================================================================
def write_metadata(df: pd.DataFrame, train_df: pd.DataFrame,
                   test_df: pd.DataFrame, output_dir: Path) -> dict:
    print(f"\n{'=' * 60}")
    print("BƯỚC 7: Xuất metadata.json")
    print("=" * 60)

    period_dist = {}
    for period in sorted(df["period"].unique()):
        n_tr = (train_df["period"] == period).sum()
        n_te = (test_df["period"]  == period).sum()
        n_line  = (df[df["period"] == period]["style"] == "line_art").sum()
        n_color = (df[df["period"] == period]["style"] == "colored").sum()
        period_dist[period] = {
            "total": int(n_tr + n_te),
            "train": int(n_tr),
            "test":  int(n_te),
            "line_art": int(n_line),
            "colored":  int(n_color),
            "trigger_word": TRIGGER_WORDS.get(period, "?"),
        }

    variant_stats = {}
    for vname, vcfg in DATASET_VARIANTS.items():
        def count(split_df):
            mask = split_df["period"].isin(vcfg["periods"])
            if vcfg["style_filter"]:
                mask &= split_df["style"] == vcfg["style_filter"]
            return int(mask.sum())
        v_tr = count(train_df)
        v_te = count(test_df)
        variant_stats[vname] = {
            "periods":      vcfg["periods"],
            "style_filter": vcfg["style_filter"],
            "train": v_tr,
            "test":  v_te,
            "total": v_tr + v_te,
        }

    # File lists để reproducibility
    train_files = [
        {"filename": safe(r["filename"]), "period": safe(r["period"]),
         "style": safe(r["style"]), "ma_so": safe(r.get("ma_so",""))}
        for _, r in train_df.iterrows()
    ]
    test_files = [
        {"filename": safe(r["filename"]), "period": safe(r["period"]),
         "style": safe(r["style"]), "ma_so": safe(r.get("ma_so",""))}
        for _, r in test_df.iterrows()
    ]

    metadata = {
        "dataset_name":  "DaiViet-Pattern Benchmark v2.0 (vector source)",
        "description":   "Task 4.1 benchmark — 291 motifs từ PDF vector gốc",
        "created":       datetime.now().isoformat(),
        "source":        "Dự án Hoa Văn Đại Việt — Đại Việt Cổ Phong + Comicola",
        "github":        "https://github.com/Hungld54321/daiviet-pattern",
        "seed":          SEED,
        "test_size":     TEST_SIZE,
        "resolutions":   RESOLUTIONS,
        "trigger_words": TRIGGER_WORDS,
        "total_images":  len(df),
        "train_count":   len(train_df),
        "test_count":    len(test_df),
        "period_distribution": period_dist,
        "variants":      variant_stats,
        "train_files":   train_files,
        "test_files":    test_files,
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {meta_path}")
    return metadata


# ============================================================================
# STEP 8: stats_report.txt
# ============================================================================
def write_report(df: pd.DataFrame, train_df: pd.DataFrame,
                 test_df: pd.DataFrame, metadata: dict, output_dir: Path,
                 n_missing: int, n_corrupt: int):
    print(f"\n{'=' * 60}")
    print("BƯỚC 8: Xuất stats_report.txt")
    print("=" * 60)

    lines = []
    sep = "=" * 60

    lines += [sep,
              "TASK 4.1 — BENCHMARK DATASET REPORT",
              f"DaiViet-Pattern v2.0 — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
              sep, ""]

    lines += ["1. NGUỒN DỮ LIỆU",
              "   Nguồn: PDF vector gốc (Hoa Văn Đại Việt — Đại Việt Cổ Phong)",
              f"   Tổng sau merge:    {len(df) + n_missing + n_corrupt} motifs",
              f"   Sau quality check: {len(df)} motifs  "
              f"(loại {n_missing} missing + {n_corrupt} corrupt)", ""]

    lines += ["2. PHÂN BỔ THEO THỜI KỲ",
              f"   {'Period':<12} {'Total':>6} {'line_art':>9} {'colored':>8}"]
    lines.append(f"   {'-'*40}")
    for period, d in metadata["period_distribution"].items():
        lines.append(f"   {period:<12} {d['total']:>6} {d['line_art']:>9} "
                     f"{d['colored']:>8}   → trigger: {d['trigger_word']}")
    lines.append("")

    lines += ["3. TRAIN/TEST SPLIT",
              f"   Strategy: stratified by period, test_size={TEST_SIZE}, seed={SEED}",
              f"   Train: {len(train_df)} images  ({len(train_df)/len(df)*100:.1f}%)",
              f"   Test:  {len(test_df)} images  ({len(test_df)/len(df)*100:.1f}%)", ""]

    lines += ["4. DATASET VARIANTS"]
    for vname, vs in metadata["variants"].items():
        sf = vs.get("style_filter") or "all styles"
        lines.append(f"   {vname:<14} train={vs['train']:>3}, test={vs['test']:>3}, "
                     f"total={vs['total']:>3}  |  {sf}  |  "
                     f"periods: {', '.join(vs['periods'])}")
    lines.append("")

    lines += ["5. RESOLUTIONS"]
    for res in RESOLUTIONS:
        lines.append(f"   {res}×{res}: tất cả variants đã chuẩn bị")
    lines.append("")

    lines += ["6. CAPTION FORMAT (với trigger word — dùng cho training)"]
    sample = df.iloc[0]
    lines.append(f"   Ví dụ: {build_caption(sample.to_dict(), use_trigger=True)}")
    lines.append("")

    lines += ["7. MODELS TO BENCHMARK (Task 4.1)"]
    for m, desc in [
        ("M1", "SDXL 1.0 vanilla           — baseline_prompts.json, no fine-tune"),
        ("M2", "SDXL + LoRA                — 768/D_all/train, rank=16, lora_alpha=32"),
        ("M3", "SD 1.5 + LoRA              — 512/D_all/train, rank=16"),
        ("M6", "SDXL + LoRA + L_cultural   — 768/D_all/train, lambda=0.3"),
    ]:
        lines.append(f"   {m}: {desc}")
    lines.append("")

    lines += ["8. EVALUATION PROTOCOL",
              "   Generate: 50 images/model × 3 periods = 150 images/model",
              "   Compare vs: test set (per-period)",
              "   Metrics: FID↓ | CLIP Score↑ | SSIM↑ | LPIPS↓ | PSNR↑", ""]

    lines += ["9. PRIOR RESULTS (APWeb-WAIM 2026, tag: apweb-v4-submitted)",
              "   Method                     FID↓     SSIM    CLIP↑",
              "   SDXL vanilla               368.06   0.1255  0.2888",
              "   SDXL+LoRA+L_cultural       336.07   0.1113  0.3135",
              "   → Sẽ được recompute trên split mới", ""]

    lines += [sep, "END OF REPORT", sep]

    report = "\n".join(lines)
    report_path = output_dir / "stats_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n  Report saved: {report_path}")


# ============================================================================
# MERGE MANIFEST → benchmark_manifest.csv
# ============================================================================
def write_benchmark_manifest(train_df: pd.DataFrame, test_df: pd.DataFrame,
                              base_dir: Path):
    merged = pd.concat([train_df, test_df], ignore_index=True)
    # Giữ các cột cần thiết, bỏ _img_path (internal)
    cols = [c for c in merged.columns if c != "_img_path"]
    out_path = base_dir / "benchmark_manifest.csv"
    merged[cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  benchmark_manifest.csv saved: {out_path}  ({len(merged)} rows)")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare DaiViet-Pattern benchmark dataset from vector_extracted/"
    )
    parser.add_argument("--base_dir",   default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    # Auto-detect base_dir
    if args.base_dir:
        base_dir = Path(args.base_dir)
    elif (Path(".") / "vector_extracted").exists():
        base_dir = Path(".")
    elif Path("D:/Hoavandaiviet/vector_extracted").exists():
        base_dir = Path("D:/Hoavandaiviet")
    else:
        print("ERROR: Không tìm thấy vector_extracted/. "
              "Chạy từ thư mục repo hoặc dùng --base_dir.")
        sys.exit(1)

    output_dir = (Path(args.output_dir) if args.output_dir
                  else base_dir / "benchmark_data")

    print()
    print("=" * 60)
    print("  PREPARE BENCHMARK DATA — Task 4.1")
    print("=" * 60)
    print(f"  Base dir:   {base_dir.resolve()}")
    print(f"  Output dir: {output_dir.resolve()}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline
    df_raw   = load_manifests(base_dir)
    n_raw    = len(df_raw)
    df_clean = quality_check(df_raw)
    n_missing = n_raw - len(df_clean)  # approximate (includes small)
    n_corrupt = 0

    train_df, test_df = split_dataset(df_clean)
    process_images(train_df, test_df, output_dir)
    write_baseline_prompts(output_dir)
    metadata = write_metadata(df_clean, train_df, test_df, output_dir)
    write_report(df_clean, train_df, test_df, metadata, output_dir,
                 n_missing, n_corrupt)
    write_benchmark_manifest(train_df, test_df, base_dir)

    print(f"\n{'=' * 60}")
    print("  XONG! Dataset sẵn sàng cho benchmark training.")
    print(f"  Output: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
