"""
crawl_wikimedia.py — Task 4.1: Crawl hoa văn Việt Nam từ Wikimedia Commons
===========================================================================

Mục tiêu: Bổ sung ảnh chất lượng cao cho 2 thời kỳ thiếu data:
  - Lý-Trần (hiện có 37 HQ, cần thêm ~43)
  - Nguyễn (hiện có 42 HQ, cần thêm ~38)

Chiến lược crawl:
  1. Category-based: duyệt các category cụ thể trên Wikimedia Commons
  2. Search-based: tìm kiếm theo keyword cho hoa văn/ornament
  3. Chỉ tải ảnh ≥ 256px (cả 2 chiều), Creative Commons
  4. Lưu metadata đầy đủ cho việc gán nhãn thủ công sau

Output:
  wikimedia_raw/
  ├── images/              # Ảnh gốc tải về
  ├── thumbnails/          # Thumbnail 256px để review nhanh
  ├── crawl_manifest.csv   # Metadata: filename, source_url, license, dims, etc.
  └── crawl_report.txt     # Báo cáo kết quả

Sau khi crawl, Hùng cần:
  1. Mở crawl_manifest.csv
  2. Review ảnh, gán cột: period, motif_type, keep (yes/no)
  3. Chạy script merge để gộp vào benchmark dataset

Cách chạy:
  python crawl_wikimedia.py                            # Chạy mặc định
  python crawl_wikimedia.py --output_dir ./wikimedia    # Tùy chỉnh output
  python crawl_wikimedia.py --dry_run                   # Chỉ xem, không tải

Tác giả: Hùng (NCS UIT) — Task 4.1
"""

import os
import sys
import csv
import json
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from urllib.parse import quote, unquote

os.environ["PYTHONIOENCODING"] = "utf-8"

import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "DaiVietPatternBot/1.0 (PhD research, UIT Vietnam; contact: hungld54321@gmail.com)"
MIN_SIZE = 256          # Minimum dimension (width or height) in pixels
MAX_FILE_SIZE_MB = 15   # Skip files larger than this (avoid huge museum photos)
DOWNLOAD_DELAY = 1.0    # Seconds between downloads (be polite)
THUMB_SIZE = 256        # Thumbnail size for quick review

# Categories to crawl — organized by target period
CATEGORIES = {
    # === LÝ-TRẦN (target: +43 ảnh) ===
    "Ly-Tran": [
        "Lý dynasty ceramics in the National Museum of Vietnamese History",
        "Trần dynasty ceramics in the National Museum of Vietnamese History",
        # Broader categories
        "Art of the Lý dynasty",
        "Art of the Trần dynasty",
        "Lý dynasty architecture",
    ],
    # === NGUYỄN (target: +38 ảnh) ===
    "Nguyen": [
        "Nguyễn dynasty ceramics in the National Museum of Vietnamese History",
        "Nguyễn dynasty ceramics in the Vietnam National Museum of Fine Arts",
        "Art of the Nguyễn dynasty",
        "Huế art",
    ],
}

# Search queries — for supplementing category crawl
SEARCH_QUERIES = {
    "Ly-Tran": [
        "Vietnamese Ly dynasty ornament pattern",
        "Vietnamese Tran dynasty ceramic motif",
        "Ly Tran dynasty dragon pattern Vietnam",
        "Vietnamese lotus motif Ly dynasty ceramic",
        "Thang Long citadel ornament decoration",
        "Ly Tran terracotta architectural ornament Vietnam",
    ],
    "Nguyen": [
        "Nguyen dynasty lacquerware pattern Vietnam",
        "Vietnamese Nguyen dynasty ceramic ornament",
        "Hue imperial art ornament pattern",
        "Vietnamese phoenix motif Nguyen dynasty",
        "Nguyen dynasty embroidery pattern",
        "Bat Trang ceramic Nguyen dynasty motif",
    ],
}

# File extensions to accept
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ============================================================================
# WIKIMEDIA API HELPERS
# ============================================================================
def api_request(params, session):
    """Make a request to the Wikimedia Commons API."""
    params["format"] = "json"
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = session.get(API_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  API error: {e}")
        return None


def get_category_members(category_name, session, limit=500):
    """Get all file pages in a Wikimedia Commons category."""
    members = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category_name}",
        "cmtype": "file",
        "cmlimit": min(limit, 500),
    }
    
    while True:
        data = api_request(params, session)
        if not data or "query" not in data:
            break
        
        for member in data["query"].get("categorymembers", []):
            members.append(member["title"])
        
        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
        else:
            break
    
    return members


def search_files(query, session, limit=50):
    """Search for files on Wikimedia Commons by keyword."""
    results = []
    params = {
        "action": "query",
        "list": "search",
        "srnamespace": 6,  # File namespace
        "srsearch": query,
        "srlimit": min(limit, 50),
    }
    
    data = api_request(params, session)
    if data and "query" in data:
        for item in data["query"].get("search", []):
            results.append(item["title"])
    
    return results


def get_image_info(titles, session):
    """Get image metadata (URL, dimensions, license) for a batch of file pages."""
    if not titles:
        return {}
    
    # API accepts max 50 titles per request
    results = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i+50]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": THUMB_SIZE,
        }
        
        data = api_request(params, session)
        if not data or "query" not in data:
            continue
        
        for page_id, page in data["query"].get("pages", {}).items():
            if int(page_id) < 0:  # Missing page
                continue
            title = page.get("title", "")
            info_list = page.get("imageinfo", [])
            if not info_list:
                continue
            
            info = info_list[0]
            ext_meta = info.get("extmetadata", {})
            
            results[title] = {
                "url": info.get("url", ""),
                "thumb_url": info.get("thumburl", ""),
                "width": info.get("width", 0),
                "height": info.get("height", 0),
                "size_bytes": info.get("size", 0),
                "mime": info.get("mime", ""),
                "license": ext_meta.get("LicenseShortName", {}).get("value", "unknown"),
                "description": ext_meta.get("ImageDescription", {}).get("value", "")[:200],
                "author": ext_meta.get("Artist", {}).get("value", "")[:100],
                "categories": ext_meta.get("Categories", {}).get("value", "")[:200],
                "page_url": f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'))}",
            }
    
    return results


def download_image(url, save_path, session):
    """Download an image from URL and save to disk."""
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = session.get(url, headers=headers, timeout=60, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Download error: {e}")
        return False


def make_thumbnail(img_path, thumb_path, size=256):
    """Create a thumbnail for quick visual review."""
    try:
        img = Image.open(img_path)
        img.thumbnail((size, size), Image.LANCZOS)
        img.save(thumb_path, "JPEG", quality=85)
        return True
    except Exception:
        return False


# ============================================================================
# MAIN CRAWL PIPELINE
# ============================================================================
def crawl_wikimedia(output_dir: Path, dry_run: bool = False):
    """Main crawl pipeline."""
    
    images_dir = output_dir / "images"
    thumbs_dir = output_dir / "thumbnails"
    images_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    
    session = requests.Session()
    
    # Collect all candidate file titles
    all_candidates = {}  # title -> suggested_period
    
    # ------------------------------------------------------------------
    # Step 1: Crawl categories
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Crawl Wikimedia Commons categories")
    print("=" * 60)
    
    for period, cats in CATEGORIES.items():
        for cat_name in cats:
            print(f"  [{period}] Category: {cat_name}")
            members = get_category_members(cat_name, session)
            new = 0
            for title in members:
                if title not in all_candidates:
                    all_candidates[title] = period
                    new += 1
            print(f"    → {len(members)} files found, {new} new")
            time.sleep(0.5)
    
    print(f"\n  Total from categories: {len(all_candidates)} unique files")
    
    # ------------------------------------------------------------------
    # Step 2: Search queries
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 2: Search by keyword")
    print("=" * 60)
    
    for period, queries in SEARCH_QUERIES.items():
        for query in queries:
            print(f"  [{period}] Search: '{query}'")
            results = search_files(query, session)
            new = 0
            for title in results:
                if title not in all_candidates:
                    all_candidates[title] = period
                    new += 1
            print(f"    → {len(results)} results, {new} new")
            time.sleep(0.5)
    
    total_candidates = len(all_candidates)
    print(f"\n  Total candidates: {total_candidates} unique files")
    
    # ------------------------------------------------------------------
    # Step 3: Get metadata and filter
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 3: Fetch metadata & filter")
    print("=" * 60)
    
    titles_list = list(all_candidates.keys())
    all_info = {}
    
    for i in range(0, len(titles_list), 50):
        batch = titles_list[i:i+50]
        batch_info = get_image_info(batch, session)
        all_info.update(batch_info)
        time.sleep(0.5)
    
    print(f"  Metadata retrieved: {len(all_info)} files")
    
    # Filter
    filtered = {}
    skip_reasons = {"small": 0, "large": 0, "wrong_type": 0, "no_url": 0}
    
    for title, info in all_info.items():
        url = info.get("url", "")
        w = info.get("width", 0)
        h = info.get("height", 0)
        size_mb = info.get("size_bytes", 0) / (1024 * 1024)
        
        # Check extension
        ext = Path(url).suffix.lower() if url else ""
        if ext not in VALID_EXTENSIONS:
            skip_reasons["wrong_type"] += 1
            continue
        
        if not url:
            skip_reasons["no_url"] += 1
            continue
        
        if min(w, h) < MIN_SIZE:
            skip_reasons["small"] += 1
            continue
        
        if size_mb > MAX_FILE_SIZE_MB:
            skip_reasons["large"] += 1
            continue
        
        filtered[title] = {
            **info,
            "suggested_period": all_candidates.get(title, "Unknown"),
        }
    
    print(f"  After filtering: {len(filtered)} files")
    print(f"  Skipped: {skip_reasons}")
    
    per_period = {}
    for title, info in filtered.items():
        p = info["suggested_period"]
        per_period[p] = per_period.get(p, 0) + 1
    
    print(f"  By period: {per_period}")
    
    if dry_run:
        print("\n  [DRY RUN] Skipping downloads.")
        # Still save manifest for review
        _save_manifest(filtered, output_dir, downloaded=False)
        return
    
    # ------------------------------------------------------------------
    # Step 4: Download images
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 4: Download images")
    print("=" * 60)
    
    downloaded = 0
    failed = 0
    manifest_rows = []
    
    for title, info in tqdm(filtered.items(), desc="  Downloading"):
        url = info["url"]
        ext = Path(url).suffix.lower()
        
        # Create safe filename
        safe_name = title.replace("File:", "").replace(" ", "_")
        # Remove problematic chars
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._-")
        if not safe_name.lower().endswith(ext):
            safe_name += ext
        
        img_path = images_dir / safe_name
        thumb_path = thumbs_dir / (Path(safe_name).stem + "_thumb.jpg")
        
        # Skip if already downloaded
        if img_path.exists():
            downloaded += 1
            info["local_filename"] = safe_name
            manifest_rows.append(info)
            continue
        
        # Download
        success = download_image(url, img_path, session)
        if success:
            # Verify image is valid
            try:
                img = Image.open(img_path)
                img.verify()
                # Make thumbnail
                make_thumbnail(img_path, thumb_path, THUMB_SIZE)
                
                downloaded += 1
                info["local_filename"] = safe_name
                manifest_rows.append(info)
            except Exception as e:
                tqdm.write(f"  Invalid image {safe_name}: {e}")
                img_path.unlink(missing_ok=True)
                failed += 1
        else:
            failed += 1
        
        time.sleep(DOWNLOAD_DELAY)
    
    print(f"\n  Downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    
    # ------------------------------------------------------------------
    # Step 5: Save manifest
    # ------------------------------------------------------------------
    _save_manifest_rows(manifest_rows, output_dir)
    
    # ------------------------------------------------------------------
    # Step 6: Generate report
    # ------------------------------------------------------------------
    _save_report(output_dir, total_candidates, len(filtered), downloaded, 
                 failed, skip_reasons, per_period)


def _save_manifest(filtered, output_dir, downloaded=False):
    """Save manifest CSV from filtered dict (for dry_run or actual)."""
    manifest_path = output_dir / "crawl_manifest.csv"
    fieldnames = [
        "local_filename", "suggested_period", "period_confirmed", "motif_type",
        "keep", "width", "height", "license", "page_url", "url",
        "description", "author", "categories",
    ]
    
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for title, info in filtered.items():
            safe_name = title.replace("File:", "").replace(" ", "_")
            safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._-")
            row = {
                "local_filename": safe_name,
                "suggested_period": info.get("suggested_period", ""),
                "period_confirmed": "",   # Hùng điền thủ công
                "motif_type": "",         # Hùng điền thủ công
                "keep": "",               # Hùng điền: yes/no
                **info,
            }
            writer.writerow(row)
    
    print(f"  Manifest saved: {manifest_path}")
    print(f"  → Hùng mở file này, review ảnh, điền: period_confirmed, motif_type, keep")


def _save_manifest_rows(manifest_rows, output_dir):
    """Save manifest CSV from downloaded rows."""
    manifest_path = output_dir / "crawl_manifest.csv"
    fieldnames = [
        "local_filename", "suggested_period", "period_confirmed", "motif_type",
        "keep", "width", "height", "license", "page_url", "url",
        "description", "author", "categories",
    ]
    
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for info in manifest_rows:
            row = {
                "local_filename": info.get("local_filename", ""),
                "suggested_period": info.get("suggested_period", ""),
                "period_confirmed": "",   # Hùng điền thủ công
                "motif_type": "",         # Hùng điền thủ công
                "keep": "",               # Hùng điền: yes/no
                **info,
            }
            writer.writerow(row)
    
    print(f"\n  Manifest saved: {manifest_path}")
    print(f"  → Hùng mở file này, review ảnh, điền: period_confirmed, motif_type, keep")


def _save_report(output_dir, total_cand, total_filtered, downloaded, 
                 failed, skip_reasons, per_period):
    """Save crawl report."""
    report = []
    report.append("=" * 60)
    report.append("WIKIMEDIA COMMONS CRAWL REPORT")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 60)
    report.append("")
    report.append(f"Total candidates found: {total_cand}")
    report.append(f"After filtering (≥{MIN_SIZE}px, ≤{MAX_FILE_SIZE_MB}MB): {total_filtered}")
    report.append(f"Downloaded successfully: {downloaded}")
    report.append(f"Failed: {failed}")
    report.append("")
    report.append("Skip reasons:")
    for reason, count in skip_reasons.items():
        report.append(f"  {reason}: {count}")
    report.append("")
    report.append("By suggested period:")
    for period, count in sorted(per_period.items()):
        report.append(f"  {period}: {count}")
    report.append("")
    report.append("NEXT STEPS:")
    report.append("  1. Open crawl_manifest.csv")
    report.append("  2. Review images in thumbnails/ folder")
    report.append("  3. For each image, fill in:")
    report.append("     - period_confirmed: Ly-Tran / Nguyen / Le / Co-Dai / skip")
    report.append("     - motif_type: geometric / floral / zoomorphic / cosmic / figural")
    report.append("     - keep: yes / no")
    report.append("  4. Run merge script to integrate into benchmark dataset")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    report_path = output_dir / "crawl_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n{report_text}")


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl Vietnamese ornamental patterns from Wikimedia Commons"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory. Default: ./wikimedia_raw"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only fetch metadata, don't download images"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else Path("wikimedia_raw")
    
    print(f"Output: {output_dir.resolve()}")
    print(f"Dry run: {args.dry_run}")
    print("")
    
    crawl_wikimedia(output_dir, dry_run=args.dry_run)
