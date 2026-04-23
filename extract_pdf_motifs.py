#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_pdf_motifs.py — Tách hoa văn từ file PDF vector gốc
=============================================================

Input:  Thư mục chứa PDF files (mỗi file = 1 page, grid các hoa văn)
Output: Từng hoa văn riêng lẻ dưới dạng PNG + extract_manifest.csv

Cách chạy:
  python extract_pdf_motifs.py --input_dir vector_source/45_hoavanlytran \\
      --excel "vector_source/excel_metadata/45 Hoa Văn Lý Trần_v04.xlsx" \\
      --period Ly-Tran --output_dir vector_extracted/ly_tran

  python extract_pdf_motifs.py --input_dir vector_source/85_hoavanle \\
      --excel "vector_source/excel_metadata/85 Hoa Văn Thời Lê.xlsx" \\
      --period Le --output_dir vector_extracted/le

  python extract_pdf_motifs.py --input_dir vector_source/70_hoa_van_nguyen \\
      --excel "vector_source/excel_metadata/70 Hoa Văn Thời Nguyễn_ver_03.xlsx" \\
      --period Nguyen --output_dir vector_extracted/nguyen

Yêu cầu:
  pip install PyMuPDF Pillow numpy tqdm openpyxl

Tác giả: Hùng (NCS UIT) — Task 4.1 Benchmark
"""

import os
import sys
import csv
import re
import argparse
from pathlib import Path
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"

import fitz          # PyMuPDF
import numpy as np
from PIL import Image
from tqdm import tqdm

# Tắt giới hạn decompression bomb — PDF vector 300 DPI có thể rất lớn
Image.MAX_IMAGE_PIXELS = None

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

# ============================================================================
# CONSTANTS
# ============================================================================
MIN_CELL_PX     = 100   # Pixel tối thiểu của mỗi chiều cell
PADDING_PX      = 3     # Trim khỏi mép cell để bỏ đường kẻ grid (chỉ dùng khi >1 cell)
WHITE_THRESH    = 250   # Ngưỡng coi là "trắng" (0-255)
WHITE_GAP_MIN   = 20    # Số pixel liên tiếp tối thiểu để coi là separator (tăng để tránh false positives)
MAX_ASPECT      = 4.0   # Nếu aspect ratio > 4:1 → strip rác, bỏ qua
MIN_CONTENT     = 0.01  # Ít nhất 1% pixel không trắng → có nội dung

BORDER_WHITE_RATIO  = 0.70  # >70% border pixels trắng → nền trắng
SATURATION_THRESH   = 0.15  # Saturation trung bình < 0.15 → line_art


# ============================================================================
# EXCEL METADATA
# ============================================================================
def _norm(text: str) -> str:
    """Chuẩn hóa mã số để dùng làm key: thay space bằng underscore."""
    return re.sub(r"\s+", "_", str(text).strip())


def load_excel_metadata(excel_path: str) -> dict:
    """
    Đọc file Excel metadata.
    Header ở row 4, data từ row 5.
    Cột: A=STT, B=Mã số, C=Họa sĩ, D=Miêu tả, E=Nguồn gốc

    Returns:
        dict: {normalized_ma_so: {ma_so, hoa_si, mieu_ta, nguon_goc}}
    """
    if not HAS_OPENPYXL:
        print("  [!] openpyxl chưa cài — bỏ qua Excel metadata.")
        print("      pip install openpyxl")
        return {}

    path = Path(excel_path)
    if not path.exists():
        print(f"  [!] Không tìm thấy file Excel: {excel_path}")
        return {}

    print(f"  Đọc Excel: {path.name}")
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    ws = wb.active

    lookup = {}
    count = 0
    for row in ws.iter_rows(min_row=5, values_only=True):
        if not any(row[:5]):
            continue
        _stt, ma_so, hoa_si, mieu_ta, *rest = list(row) + [None, None]
        nguon_goc = rest[0] if rest else None

        if ma_so is None:
            continue

        ma_so_str = str(ma_so).strip()
        key = _norm(ma_so_str)
        lookup[key] = {
            "ma_so":    ma_so_str,
            "hoa_si":   str(hoa_si).strip()   if hoa_si   else "",
            "mieu_ta":  str(mieu_ta).strip()  if mieu_ta  else "",
            "nguon_goc": str(nguon_goc).strip() if nguon_goc else "",
        }
        count += 1

    wb.close()
    print(f"  → {count} mục đã tải từ Excel")
    return lookup


def _strip_leading_zeros(s: str) -> str:
    """
    Strip leading zeros khỏi các số trong chuỗi normalized.
    VD: "HVDV_-_LTH_-_CHAM_KHAC_-_086" → "HVDV_-_LTH_-_CHAM_KHAC_-_86"
    Giữ nguyên phần chữ (vd: "023b" → "23b").
    """
    return re.sub(r'(?<=[_-])0+(\d)', r'\1', s)


def match_excel(pdf_stem: str, lookup: dict) -> dict | None:
    """
    Ghép tên file PDF với mục trong Excel.

    PDF stem có thể có spaces hoặc underscores, ví dụ:
      "HVDV - LYTRAN - 001_ver_01"  (spaces)
      "HVDV_-_NGN_-_023b_ver_01"    (underscores)
    Ma số Excel: "HVDV - LYTRAN - 001"
    Normalized:  "HVDV_-_LYTRAN_-_001"

    Chiến lược (theo thứ tự ưu tiên):
    1. norm_stem.startswith(key)           — prefix chính xác
    2. base (bỏ _ver_XX) == key           — sau khi strip version
    3. stripped_base == stripped_key       — bỏ leading zeros ở số
    4. key in norm_stem                    — substring fallback
    """
    if not lookup:
        return None

    # Normalize PDF stem (spaces → underscores)
    norm_stem = _norm(pdf_stem)
    base = re.sub(r"_ver_\d+.*$", "", norm_stem, flags=re.IGNORECASE)
    stripped_base = _strip_leading_zeros(base)

    # 1. prefix match (chính xác)
    for key, meta in lookup.items():
        if norm_stem.startswith(key):
            return meta

    # 2. base == key (sau khi bỏ _ver_XX)
    if base in lookup:
        return lookup[base]

    # 3. so sánh sau khi strip leading zeros ở cả hai phía
    for key, meta in lookup.items():
        if stripped_base == _strip_leading_zeros(key):
            return meta

    # 4. substring fallback
    for key, meta in lookup.items():
        if key and key in norm_stem:
            return meta

    return None


# ============================================================================
# PDF RENDERING
# ============================================================================
def render_pdf_page(pdf_path: Path, dpi: int = 300) -> Image.Image:
    """Render trang đầu tiên của PDF thành PIL Image với độ phân giải dpi."""
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    zoom = dpi / 72          # PDF dùng 72 DPI làm chuẩn
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


# ============================================================================
# GRID DETECTION — PROJECTION PROFILE
# ============================================================================
def _find_cell_ranges(gray2d: np.ndarray,
                      axis: int,
                      white_thresh: int = WHITE_THRESH,
                      min_gap: int = WHITE_GAP_MIN,
                      min_cell: int = MIN_CELL_PX) -> list[tuple[int, int]]:
    """
    Tìm các vùng có nội dung (cell ranges) dọc theo một trục.

    gray2d : mảng 2D grayscale float32 (H, W)
    axis   : 0 → tìm separator theo chiều dọc (hàng), 1 → chiều ngang (cột)

    Điều kiện separator ĐÚNG:
      - Dùng MIN per row/col (không phải mean): toàn bộ pixel trong hàng/cột
        phải > white_thresh. Điều này đảm bảo chỉ những hàng/cột hoàn toàn
        trắng mới được coi là separator — tránh false positive từ nội dung
        thưa (sparse content) có mean cao nhưng vẫn có điểm đen.
      - Phải có >= min_gap px liên tiếp → lọc nhiễu render 1-2px.

    Returns: [(start, end), ...] — tọa độ của từng dải có nội dung.
    """
    # MIN per row (axis=0) hoặc per col (axis=1)
    # profile[i] = pixel tối nhất (darkest) trong hàng i hoặc cột i
    # Nếu profile[i] > white_thresh → không có pixel tối nào → separator
    profile_min = np.min(gray2d, axis=1 - axis)   # shape (H,) hoặc (W,)
    n = len(profile_min)
    is_gap = profile_min > white_thresh

    # Tìm các run liên tiếp là gap
    gaps: list[tuple[int, int]] = []
    in_gap = False
    gap_start = 0
    for i in range(n):
        if is_gap[i] and not in_gap:
            in_gap = True
            gap_start = i
        elif not is_gap[i] and in_gap:
            in_gap = False
            if i - gap_start >= min_gap:
                gaps.append((gap_start, i))
    if in_gap and n - gap_start >= min_gap:
        gaps.append((gap_start, n))

    # Cell ranges = khoảng nội dung giữa các gap
    ranges: list[tuple[int, int]] = []
    prev_end = 0
    for gs, ge in gaps:
        if gs - prev_end >= min_cell:
            ranges.append((prev_end, gs))
        prev_end = ge
    if n - prev_end >= min_cell:
        ranges.append((prev_end, n))

    return ranges


def detect_grid_cells(img: Image.Image,
                      min_cell_px: int = MIN_CELL_PX,
                      max_aspect: float = MAX_ASPECT,
                      min_content: float = MIN_CONTENT) -> list[tuple[int, int, int, int]]:
    """
    Phát hiện các cell hoa văn trong ảnh render từ PDF bằng projection profile.

    Dùng MIN per row/col thay vì MEAN để tránh false separator detection
    trong các motif đơn có nhiều khoảng trắng nội tại.

    Returns: [(x1, y1, x2, y2), ...] — tọa độ pixel từng cell.
    """
    arr = np.array(img)
    gray = np.mean(arr, axis=2).astype(np.float32)   # (H, W)
    H, W = gray.shape

    h_ranges = _find_cell_ranges(gray, axis=0, min_cell=min_cell_px)
    v_ranges = _find_cell_ranges(gray, axis=1, min_cell=min_cell_px)

    cells: list[tuple[int, int, int, int]] = []
    for y1, y2 in h_ranges:
        for x1, x2 in v_ranges:
            cw, ch = x2 - x1, y2 - y1

            # Lọc thin strip (aspect ratio quá lệch)
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            if aspect > max_aspect:
                continue

            # Lọc cell quá trắng (không có nội dung)
            region = gray[y1:y2, x1:x2]
            non_white = np.mean(region < 240)
            if non_white < min_content:
                continue

            cells.append((x1, y1, x2, y2))

    # Fallback: không detect được grid → bounding box của nội dung tối
    if not cells:
        binary = gray < 200
        if np.any(binary):
            rows_with = np.where(np.any(binary, axis=1))[0]
            cols_with = np.where(np.any(binary, axis=0))[0]
            y1, y2 = int(rows_with[0]), int(rows_with[-1]) + 1
            x1, x2 = int(cols_with[0]), int(cols_with[-1]) + 1
            cells = [(x1, y1, x2, y2)]
        else:
            cells = [(0, 0, W, H)]

    return cells


# ============================================================================
# STYLE CLASSIFICATION
# ============================================================================
def classify_style(crop: Image.Image) -> str:
    """
    Phân loại motif:
      line_art — nền trắng VÀ nội dung chủ yếu đen/xám (saturation thấp)
      colored  — có nền màu HOẶC nội dung nhiều màu sắc
    """
    arr = np.array(crop)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return "line_art"

    H, W = arr.shape[:2]
    border = max(8, min(20, min(H, W) // 8))

    top    = arr[:border, :, :]
    bottom = arr[-border:, :, :]
    left   = arr[:, :border, :]
    right  = arr[:, -border:, :]

    border_px = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3),
    ])

    is_white = np.all(border_px > WHITE_THRESH, axis=1)
    white_ratio = float(np.mean(is_white))

    if white_ratio > BORDER_WHITE_RATIO:
        # Nền trắng → kiểm tra saturation của nội dung
        qh, qw = max(H // 4, 1), max(W // 4, 1)
        center = arr[qh:-qh, qw:-qw, :].astype(np.float32)
        if center.size == 0:
            return "line_art"

        r, g, b = center[:, :, 0], center[:, :, 1], center[:, :, 2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)

        with np.errstate(divide="ignore", invalid="ignore"):
            sat = np.where(max_c > 0, (max_c - min_c) / max_c, 0.0)

        non_white_mask = np.any(center < 200, axis=2)
        if np.sum(non_white_mask) > 20:
            avg_sat = float(np.mean(sat[non_white_mask]))
            return "line_art" if avg_sat < SATURATION_THRESH else "colored"
        return "line_art"

    return "colored"


# ============================================================================
# PROCESS SINGLE PDF
# ============================================================================
def process_pdf(pdf_path: Path,
                output_dir: Path,
                period: str,
                dpi: int,
                excel_lookup: dict) -> tuple[list[dict], bool]:
    """
    Xử lý một file PDF: render → detect grid → crop → classify → save.

    Returns:
        (rows, matched_excel)
        rows: list of dict, mỗi dict là một hàng trong manifest
        matched_excel: True nếu tìm được metadata trong Excel
    """
    # Render PDF
    try:
        img = render_pdf_page(pdf_path, dpi=dpi)
    except Exception as exc:
        tqdm.write(f"  [ERROR] Render thất bại: {pdf_path.name} — {exc}")
        return [], False

    # Detect cells
    cells = detect_grid_cells(img)

    # Match Excel
    pdf_stem = pdf_path.stem
    meta = match_excel(pdf_stem, excel_lookup)
    matched = meta is not None

    # File ID cho tên output
    if meta:
        file_id = _norm(meta["ma_so"])
    else:
        # Bỏ _ver_XX để tên ngắn gọn hơn
        file_id = re.sub(r"_ver_\d+.*$", "", pdf_stem, flags=re.IGNORECASE)

    rows: list[dict] = []
    # Chỉ trim padding khi có nhiều cell (grid thật sự).
    # Nếu chỉ 1 cell, không cắt để tránh mất viền motif.
    pad = PADDING_PX if len(cells) > 1 else 0

    for idx, (x1, y1, x2, y2) in enumerate(cells):
        # Trim padding để bỏ đường kẻ grid
        px1 = min(x1 + pad, x2 - 1)
        py1 = min(y1 + pad, y2 - 1)
        px2 = max(x2 - pad, x1 + 1)
        py2 = max(y2 - pad, y1 + 1)

        if (px2 - px1) < 50 or (py2 - py1) < 50:
            continue

        crop = img.crop((px1, py1, px2, py2))
        style = classify_style(crop)

        crop_name = f"{file_id}_crop_{idx + 1:02d}.png"
        crop.save(output_dir / crop_name, "PNG", optimize=False)

        rows.append({
            "filename":   crop_name,
            "source_pdf": pdf_path.name,
            "ma_so":      meta["ma_so"]    if meta else "",
            "period":     period,
            "style":      style,
            "width":      px2 - px1,
            "height":     py2 - py1,
            "cell_index": idx + 1,
            "total_cells": len(cells),
            "hoa_si":     meta["hoa_si"]   if meta else "",
            "mieu_ta":    meta["mieu_ta"]  if meta else "",
            "nguon_goc":  meta["nguon_goc"] if meta else "",
        })

    return rows, matched


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Tách từng hoa văn từ PDF vector thành PNG riêng lẻ"
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Thư mục chứa file PDF")
    parser.add_argument("--excel",      default=None,
                        help="File Excel metadata (tùy chọn)")
    parser.add_argument("--period",     required=True,
                        choices=["Co-Dai", "Ly-Tran", "Le", "Nguyen"],
                        help="Thời kỳ lịch sử")
    parser.add_argument("--output_dir", default=None,
                        help="Thư mục output (mặc định: vector_extracted/<period>)")
    parser.add_argument("--dpi",        type=int, default=300,
                        help="DPI render (mặc định: 300)")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Chỉ phân tích, không lưu file")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[ERROR] Không tìm thấy thư mục: {input_dir}")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        _period_dir = {"Ly-Tran": "ly_tran", "Le": "le",
                       "Nguyen": "nguyen", "Co-Dai": "co_dai"}
        output_dir = Path("vector_extracted") / _period_dir.get(args.period, args.period.lower())

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load Excel metadata
    excel_lookup: dict = {}
    if args.excel:
        excel_lookup = load_excel_metadata(args.excel)

    # Tìm PDFs
    # Lọc bỏ file macOS metadata (._xxx.pdf) — không phải PDF thật
    pdfs = sorted(p for p in input_dir.glob("*.pdf") if not p.name.startswith("._"))
    if not pdfs:
        print(f"[ERROR] Không có file PDF nào trong: {input_dir}")
        sys.exit(1)

    # Header
    print()
    print("=" * 60)
    print(f"  EXTRACT PDF MOTIFS — {args.period}")
    print("=" * 60)
    print(f"  Input:   {input_dir}")
    print(f"  Output:  {output_dir}" + (" [DRY RUN]" if args.dry_run else ""))
    print(f"  DPI:     {args.dpi}")
    print(f"  PDFs:    {len(pdfs)}")
    print(f"  Excel:   {len(excel_lookup)} mục")
    print()

    all_rows: list[dict] = []
    n_matched = 0
    no_motif_pdfs: list[str] = []

    for pdf_path in tqdm(pdfs, desc="Extracting", ncols=70):
        if args.dry_run:
            tqdm.write(f"  [dry] {pdf_path.name}")
            continue

        rows, matched = process_pdf(
            pdf_path, output_dir, args.period, args.dpi, excel_lookup
        )
        all_rows.extend(rows)
        if matched:
            n_matched += 1
        if not rows:
            no_motif_pdfs.append(pdf_path.name)
            tqdm.write(f"  [WARN] Không detect được motif: {pdf_path.name}")
        else:
            tag = "[meta OK]" if matched else "[no meta]"
            tqdm.write(f"  {pdf_path.name}: {len(rows)} motifs {tag}")

    if args.dry_run:
        print("[DRY RUN] Kết thúc — không có file nào được lưu.")
        return

    # Ghi manifest
    manifest_path = output_dir / "extract_manifest.csv"
    fieldnames = [
        "filename", "source_pdf", "ma_so", "period", "style",
        "width", "height", "cell_index", "total_cells",
        "hoa_si", "mieu_ta", "nguon_goc",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    n_line  = sum(1 for r in all_rows if r["style"] == "line_art")
    n_color = sum(1 for r in all_rows if r["style"] == "colored")

    print()
    print("=" * 60)
    print(f"  XONG — {args.period}")
    print("=" * 60)
    print(f"  PDFs xử lý:            {len(pdfs)}")
    print(f"  PDFs có Excel metadata: {n_matched} / {len(pdfs)}")
    if no_motif_pdfs:
        print(f"  PDFs không có motif:   {len(no_motif_pdfs)}")
        for name in no_motif_pdfs:
            print(f"    - {name}")
    print(f"  Tổng motifs:            {len(all_rows)}")
    print(f"    line_art:             {n_line}")
    print(f"    colored:              {n_color}")
    print(f"  Manifest:               {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
