"""
ingest.py
Reads dataset_manifest.csv + dataset images,
computes CLIP embeddings, stores in local SQLite DB.

Usage:  py ingest.py
"""
import os, sys, json
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import pandas as pd
import torch
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from db import get_conn, init_db

# ── config ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = Path(os.getenv("DATASET_DIR",  "D:/Hoavandaiviet/dataset"))
MANIFEST_CSV = Path(os.getenv("MANIFEST_CSV", "D:/Hoavandaiviet/dataset_manifest.csv"))

PERIOD_FOLDER = {
    "Co-Dai":  "co_dai",
    "Le":      "le",
    "Ly-Tran": "ly_tran",
    "Moi":     "moi",
    "Nguyen":  "nguyen",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP ViT-B/32 on {DEVICE} ...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model = clip_model.to(DEVICE).eval()

def embed_image(img_path: Path):
    try:
        img    = Image.open(img_path).convert("RGB")
        tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = clip_model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze().cpu().tolist()
    except Exception as e:
        tqdm.write(f"  WARN {img_path.name}: {e}")
        return None

# ── init db ────────────────────────────────────────────────────────────────────
init_db()
conn = get_conn()

# Clear real rows, re-ingest
conn.execute("DELETE FROM patterns WHERE is_generated = 0")
conn.commit()
print("Cleared old rows. Starting ingest...")

# ── manifest ───────────────────────────────────────────────────────────────────
df = pd.read_csv(MANIFEST_CSV)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print(f"Manifest: {len(df)} rows")

inserted, skipped = 0, 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting"):
    filename = str(row.get("filename", "")).strip()
    period   = str(row.get("period",   "")).strip()
    folder   = PERIOD_FOLDER.get(period)

    img_path = None
    if folder:
        p = DATASET_DIR / folder / filename
        if p.exists():
            img_path = p
    if img_path is None:
        for f in PERIOD_FOLDER.values():
            p = DATASET_DIR / f / filename
            if p.exists() and not p.name.startswith("._"):
                img_path = p
                break
    if img_path is None or img_path.name.startswith("._"):
        skipped += 1
        continue

    emb = embed_image(img_path)

    def safe(col):
        v = row.get(col, None)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return str(v).strip() or None

    multi = row.get("multi_motif", None)
    if multi is not None and not pd.isna(multi):
        multi = 1 if str(multi).strip().lower() in ("true", "1", "yes") else 0
    else:
        multi = None

    conn.execute("""
        INSERT INTO patterns
          (filename, period, style, multi_motif, motif_subject,
           artifact_source, cropped_from, source_url, is_generated, clip_embedding)
        VALUES (?,?,?,?,?,?,?,?,0,?)
    """, (
        filename, safe("period"), safe("style"), multi,
        safe("motif_subject"), safe("artifact_source"),
        safe("cropped_from"), safe("source_url"),
        json.dumps(emb) if emb else None,
    ))
    inserted += 1

    if inserted % 50 == 0:
        conn.commit()

conn.commit()
conn.close()
print(f"\nDone. Inserted: {inserted}  Skipped: {skipped}")
