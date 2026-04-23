"""
extract_pdf_motifs.py — Tách hoa văn từ file PDF vector gốc
=============================================================

Input:  Thư mục chứa PDF files (mỗi file = 1 page, grid các hoa văn)
Output: Từng hoa văn riêng lẻ dưới dạng PNG

Cách hoạt động:
  1. Render PDF page thành ảnh lớn (300 DPI)
  2. Phát hiện grid bằng connected component analysis
  3. Crop từng ô trong grid thành file PNG riêng
  4. Phân loại: line_art (nền trắng) vs colored (nền màu)
  5. Lưu metadata vào manifest CSV

Cách chạy:
  python extract_pdf_motifs.py --input_dir "D:/path/to/70_hoa_van_nguyen" --period Nguyen
  python extract_pdf_motifs.py --input_dir "D:/path/to/45_hoavanlytran" --period Ly-Tran
  python extract_pdf_motifs.py --input_dir "D:/path/to/85_hoavanle" --period Le

Tác giả: Hùng (NCS UIT) — Task 4.1
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
RENDER_DPI = 300           # High DPI for vector quality
MIN_CELL_SIZE = 100        # Minimum cell dimension in pixels
PADDING = 5                # Pixels to trim from cell edges (remove grid lines)
WHITE_THRESHOLD = 240      # Pixel value threshold to detect white background
WHITE_RATIO_THRESHOLD = 0.7  # If >70% of border pixels are white → line_art
OUTPUT_FORMAT = "PNG"


# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def render_pdf_page(pdf_path, dpi=300):
    """Render first page of PDF to PIL Image at specified DPI."""
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    
    # Render at high DPI
    zoom = dpi / 72  # PDF default is 72 DPI
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def detect_grid_cells(img, min_cell_size=100):
    """
    Detect individual motif cells in a grid layout.
    
    Strategy: 
    - Find large contiguous regions separated by gaps/borders
    - Use projection profiles to find row/column separators
    """
    arr = np.array(img)
    gray = np.mean(arr, axis=2)
    
    h, w = gray.shape
    
    # If image is small enough to be a single motif, return whole image
    if h < min_cell_size * 2 and w < min_cell_size * 2:
        return [(0, 0, w, h)]
    
    # Use projection profiles to find grid lines
    # A grid line = column/row where most pixels are near-white or near-border-color
    
    # Horizontal profile: for each row, compute mean intensity
    h_profile = np.mean(gray, axis=1)
    # Vertical profile: for each column, compute mean intensity  
    v_profile = np.mean(gray, axis=0)
    
    # Find separators: rows/cols where intensity is very high (white gaps)
    # or where there's a sharp transition
    h_seps = find_separators(h_profile, min_cell_size)
    v_seps = find_separators(v_profile, min_cell_size)
    
    # Build cells from separators
    cells = []
    for i in range(len(h_seps) - 1):
        for j in range(len(v_seps) - 1):
            y1, y2 = h_seps[i], h_seps[i + 1]
            x1, x2 = v_seps[j], v_seps[j + 1]
            
            # Skip tiny cells (likely artifacts)
            cell_w = x2 - x1
            cell_h = y2 - y1
            if cell_w < min_cell_size or cell_h < min_cell_size:
                continue
            
            # Skip thin strips (aspect ratio > 5:1 — likely grid line artifacts)
            aspect = max(cell_w, cell_h) / max(min(cell_w, cell_h), 1)
            if aspect > 5:
                continue
            
            # Check if cell has content (not entirely white)
            cell_region = gray[y1:y2, x1:x2]
            non_white_ratio = np.mean(cell_region < 240)
            if non_white_ratio > 0.01:  # At least 1% non-white pixels
                cells.append((x1, y1, x2, y2))
    
    # If no grid detected, try contour-based approach
    if len(cells) <= 1:
        cells = detect_cells_by_content(arr, gray, min_cell_size)
    
    # If still nothing, return whole image
    if not cells:
        cells = [(0, 0, w, h)]
    
    return cells


def find_separators(profile, min_size):
    """Find separator positions in a projection profile."""
    n = len(profile)
    
    # Threshold: high intensity = gap
    threshold = 250
    is_gap = profile > threshold
    
    # Find runs of gaps
    separators = [0]  # Start
    in_gap = False
    gap_start = 0
    
    for i in range(n):
        if is_gap[i] and not in_gap:
            in_gap = True
            gap_start = i
        elif not is_gap[i] and in_gap:
            in_gap = False
            gap_mid = (gap_start + i) // 2
            # Only add if distance from last separator is large enough
            if gap_mid - separators[-1] > min_size:
                separators.append(gap_mid)
    
    # Add end
    if n - separators[-1] > min_size:
        separators.append(n)
    elif separators[-1] != n:
        separators[-1] = n
    
    return separators


def detect_cells_by_content(arr, gray, min_cell_size):
    """Fallback: detect cells by finding content bounding boxes."""
    from PIL import Image as PILImage
    
    h, w = gray.shape
    
    # Binarize: content = dark pixels
    binary = gray < 200
    
    # Find bounding box of all content
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return []
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return [(cmin, rmin, cmax + 1, rmax + 1)]


def classify_style(img_crop):
    """Classify a cropped motif as line_art or colored."""
    arr = np.array(img_crop)
    
    # Sample border pixels (10px from each edge)
    border_size = min(10, min(arr.shape[0], arr.shape[1]) // 4)
    
    top = arr[:border_size, :, :]
    bottom = arr[-border_size:, :, :]
    left = arr[:, :border_size, :]
    right = arr[:, -border_size:, :]
    
    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3),
    ])
    
    # Check if border is mostly white
    is_white = np.all(border_pixels > WHITE_THRESHOLD, axis=1)
    white_ratio = np.mean(is_white)
    
    if white_ratio > WHITE_RATIO_THRESHOLD:
        # White background — check if content is mostly black/gray (line art)
        # or colorful
        center_h = arr.shape[0] // 4
        center_w = arr.shape[1] // 4
        center = arr[center_h:-center_h, center_w:-center_w, :]
        
        # Check color saturation in center
        r, g, b = center[:,:,0].astype(float), center[:,:,1].astype(float), center[:,:,2].astype(float)
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        saturation = np.where(max_c > 0, (max_c - min_c) / max_c, 0)
        
        # Non-white pixels
        non_white = np.any(center < 200, axis=2)
        if np.sum(non_white) > 0:
            avg_sat = np.mean(saturation[non_white])
            if avg_sat < 0.15:
                return "line_art"
            else:
                return "colored"
        return "line_art"
    else:
        return "colored"


def process_single_pdf(pdf_path, output_dir, period, file_id):
    """Process a single PDF file, extract all motifs."""
    results = []
    
    try:
        img = render_pdf_page(pdf_path, dpi=RENDER_DPI)
    except Exception as e:
        print(f"  Error rendering {pdf_path.name}: {e}")
        return results
    
    cells = detect_grid_cells(img, MIN_CELL_SIZE)
    
    for idx, (x1, y1, x2, y2) in enumerate(cells):
        # Add padding to avoid grid lines
        px1 = min(x1 + PADDING, x2)
        py1 = min(y1 + PADDING, y2)
        px2 = max(x2 - PADDING, x1)
        py2 = max(y2 - PADDING, y1)
        
        if px2 - px1 < 50 or py2 - py1 < 50:
            continue
        
        crop = img.crop((px1, py1, px2, py2))
        style = classify_style(crop)
        
        # Generate filename
        crop_name = f"{file_id}_crop_{idx+1:02d}.png"
        crop_path = output_dir / crop_name
        
        crop.save(crop_path, "PNG")
        
        results.append({
            "filename": crop_name,
            "source_pdf": pdf_path.name,
            "period": period,
            "style": style,
            "width": px2 - px1,
            "height": py2 - py1,
            "cell_index": idx + 1,
            "total_cells": len(cells),
        })
    
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract individual motifs from PDF vector files"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Default: <input_dir>_extracted/")
    parser.add_argument("--period", type=str, required=True,
                        choices=["Co-Dai", "Ly-Tran", "Le", "Nguyen"],
                        help="Historical period for these files")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Render DPI (default: 300)")
    args = parser.parse_args()
    
    global RENDER_DPI
    RENDER_DPI = args.dpi
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / f"{input_dir.name}_extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDFs
    pdfs = sorted(input_dir.glob("*.pdf"))
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Period: {args.period}")
    print(f"PDFs found: {len(pdfs)}")
    print(f"Render DPI: {RENDER_DPI}")
    print()
    
    if not pdfs:
        print("No PDF files found!")
        sys.exit(1)
    
    # Process all PDFs
    all_results = []
    
    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        file_id = pdf_path.stem  # e.g., HVDV_-_NGN_-_023b_ver_01
        results = process_single_pdf(pdf_path, output_dir, args.period, file_id)
        all_results.extend(results)
        
        if results:
            tqdm.write(f"  {pdf_path.name}: {len(results)} motifs extracted")
        else:
            tqdm.write(f"  {pdf_path.name}: WARNING - no motifs detected")
    
    # Save manifest
    manifest_path = output_dir / "extract_manifest.csv"
    fieldnames = ["filename", "source_pdf", "period", "style", 
                  "width", "height", "cell_index", "total_cells"]
    
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    # Summary
    n_line = sum(1 for r in all_results if r["style"] == "line_art")
    n_color = sum(1 for r in all_results if r["style"] == "colored")
    
    print(f"\n{'='*50}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"  PDFs processed: {len(pdfs)}")
    print(f"  Motifs extracted: {len(all_results)}")
    print(f"    line_art: {n_line}")
    print(f"    colored:  {n_color}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Output:   {output_dir}")


if __name__ == "__main__":
    main()
