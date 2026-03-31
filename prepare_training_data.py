"""
prepare_training_data.py
Reads dataset_manifest.csv, filters to single-motif line art,
resizes each image to 512x512 (white-padded), writes caption files.

Dataset layout:
  D:\Hoavandaiviet\dataset\
      co_dai\  le\  ly_tran\  moi\  nguyen\   (each has *.jpg + *.json)
"""

import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# -- paths ----------------------------------------------------------------------
BASE_DIR    = Path("D:/Hoavandaiviet")
MANIFEST    = BASE_DIR / "dataset_manifest.csv"
DATASET_DIR = BASE_DIR / "dataset"
OUT_IMAGES  = BASE_DIR / "train_data" / "images"
OUT_CAPS    = BASE_DIR / "train_data" / "captions"
RESOLUTION  = 512

# period value in CSV  ->  subfolder name on disk
PERIOD_FOLDER = {
    "Co-Dai":  "co_dai",
    "Le":      "le",
    "Ly-Tran": "ly_tran",
    "Moi":     "moi",
    "Nguyen":  "nguyen",
}

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_CAPS.mkdir(parents=True, exist_ok=True)

# -- load & filter manifest -----------------------------------------------------
if not MANIFEST.exists():
    print(f"ERROR: manifest not found at {MANIFEST}")
    sys.exit(1)

df = pd.read_csv(MANIFEST)
print(f"Loaded manifest: {len(df)} rows")

# style == line_art
df = df[df["style"].str.strip().str.lower() == "line_art"]
print(f"After style filter (line_art): {len(df)} rows")

# multi_motif == False  (stored as Python bool after pandas read_csv)
df = df[df["multi_motif"].astype(str).str.strip().str.lower() == "false"]
print(f"After multi_motif filter (single): {len(df)} rows\n")

if len(df) == 0:
    print("ERROR: no images remain after filtering.")
    sys.exit(1)

# -- helpers --------------------------------------------------------------------
def safe(val, fallback="unknown"):
    if pd.isna(val) or str(val).strip() == "":
        return fallback
    return str(val).strip()

def build_caption(row):
    period  = safe(row["period"], "Vietnamese")
    subject = safe(row["motif_subject"], "ornamental pattern")
    # No motif_type column — derive a readable label from period
    period_labels = {
        "Co-Dai":  "ancient Bronze Age",
        "Le":      "Le dynasty",
        "Ly-Tran": "Ly-Tran dynasty",
        "Moi":     "ethnic minority",
        "Nguyen":  "Nguyen dynasty",
    }
    period_label = period_labels.get(period, period)
    return (
        f"Vietnamese {period_label} ornamental pattern, {subject}, "
        f"traditional Dai Viet art style, black and white line art"
    )

def find_image(filename: str, period: str) -> Path | None:
    folder = PERIOD_FOLDER.get(period)
    if folder:
        p = DATASET_DIR / folder / filename
        if p.exists():
            return p
    # Fallback: search all period folders
    for folder in PERIOD_FOLDER.values():
        p = DATASET_DIR / folder / filename
        if p.exists():
            return p
    return None

# -- process --------------------------------------------------------------------
processed     = 0
skipped       = 0
period_counts = {}

rows = df.to_dict("records")

for row in tqdm(rows, desc="Preparing images"):
    filename = safe(row["filename"])
    period   = safe(row["period"])

    img_path = find_image(filename, period)
    if img_path is None:
        tqdm.write(f"  SKIP (not found): {filename}")
        skipped += 1
        continue

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        tqdm.write(f"  SKIP (bad image) {filename}: {e}")
        skipped += 1
        continue

    # White-pad to square then resize
    w, h   = img.size
    maxdim = max(w, h)
    canvas = Image.new("RGB", (maxdim, maxdim), (255, 255, 255))
    canvas.paste(img, ((maxdim - w) // 2, (maxdim - h) // 2))
    resized = canvas.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

    stem    = Path(filename).stem
    out_img = OUT_IMAGES / f"{stem}.png"
    out_cap = OUT_CAPS   / f"{stem}.txt"

    resized.save(out_img, "PNG")
    out_cap.write_text(build_caption(row), encoding="utf-8")

    period_counts[period] = period_counts.get(period, 0) + 1
    processed += 1

# -- summary --------------------------------------------------------------------
print(f"\n{'-'*52}")
print(f"  Processed : {processed} images")
print(f"  Skipped   : {skipped}")
print(f"\n  Images by period:")
for p, c in sorted(period_counts.items()):
    print(f"    {p:<12} {c}")
print(f"{'-'*52}")
print(f"Output -> {OUT_IMAGES}")
print("Done. Ready for training.\n")
