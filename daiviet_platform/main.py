"""
main.py — DaiViet Pattern Platform API (SQLite backend)
"""
import os, json, base64, uuid, time
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from pathlib import Path
from io import BytesIO
from typing import Optional

import torch
import numpy as np
import open_clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db import get_conn, init_db, cosine_search

# ── config ─────────────────────────────────────────────────────────────────────
DATASET_DIR   = Path("D:/Hoavandaiviet/dataset")
LORA_PATH     = "D:/Hoavandaiviet/lora_output/daiviet_lora"
GENERATED_DIR = Path("D:/Hoavandaiviet/generated")
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_CSV  = Path("D:/Hoavandaiviet/dataset_manifest.csv")

PERIOD_FOLDER = {
    "Co-Dai":  "co_dai",
    "Le":      "le",
    "Ly-Tran": "ly_tran",
    "Moi":     "moi",
    "Nguyen":  "nguyen",
}
PERIOD_LABEL = {
    "Co-Dai":  "ancient Bronze Age Dong Son",
    "Ly-Tran": "Ly-Tran dynasty",
    "Le":      "Le dynasty",
    "Moi":     "ethnic minority",
    "Nguyen":  "Nguyen dynasty",
}

# ── init ────────────────────────────────────────────────────────────────────────
init_db()

app = FastAPI(title="DaiViet Pattern Platform", version="1.0.0")

# ── CLIP ────────────────────────────────────────────────────────────────────────
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP on {CLIP_DEVICE} ...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model     = clip_model.to(CLIP_DEVICE).eval()
print("CLIP ready.")

# ── LoRA pipeline (lazy) ────────────────────────────────────────────────────────
lora_pipeline = None

# ── static ──────────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

# ── image serving ───────────────────────────────────────────────────────────────
@app.get("/images/{period_folder}/{filename}")
def serve_image(period_folder: str, filename: str):
    p = DATASET_DIR / period_folder / filename
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p))

@app.get("/generated/{filename}")
def serve_generated(filename: str):
    p = GENERATED_DIR / filename
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p))

# ══════════════════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/stats")
def stats():
    conn = get_conn()
    total     = conn.execute("SELECT COUNT(*) FROM patterns WHERE is_generated=0").fetchone()[0]
    generated = conn.execute("SELECT COUNT(*) FROM patterns WHERE is_generated=1").fetchone()[0]
    rows_p    = conn.execute("SELECT period, COUNT(*) c FROM patterns WHERE is_generated=0 GROUP BY period ORDER BY c DESC").fetchall()
    rows_s    = conn.execute("SELECT style, COUNT(*) c FROM patterns WHERE is_generated=0 GROUP BY style ORDER BY c DESC").fetchall()
    conn.close()
    return {
        "total_patterns":    total,
        "generated_patterns": generated,
        "by_period": {r["period"]: r["c"] for r in rows_p},
        "by_style":  {r["style"]:  r["c"] for r in rows_s},
    }

# ══════════════════════════════════════════════════════════════════════════════
# BROWSE & FILTER
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/patterns")
def get_patterns(
    period:      Optional[str]  = Query(None),
    style:       Optional[str]  = Query(None),
    multi_motif: Optional[bool] = Query(None),
    limit:       int            = Query(50, ge=1, le=200),
    offset:      int            = Query(0,  ge=0),
):
    clauses, params = ["is_generated = 0"], []
    if period:      clauses.append("period = ?");      params.append(period)
    if style:       clauses.append("style = ?");       params.append(style)
    if multi_motif is not None:
        clauses.append("multi_motif = ?")
        params.append(1 if multi_motif else 0)

    where = " AND ".join(clauses)
    conn  = get_conn()
    total = conn.execute(f"SELECT COUNT(*) FROM patterns WHERE {where}", params).fetchone()[0]
    rows  = conn.execute(
        f"SELECT id,filename,period,style,multi_motif,motif_subject,source_url "
        f"FROM patterns WHERE {where} ORDER BY id LIMIT ? OFFSET ?",
        params + [limit, offset]
    ).fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        folder = PERIOD_FOLDER.get(d["period"], "")
        d["image_url"] = f"/images/{folder}/{d['filename']}" if folder else None
        results.append(d)

    return {"total": total, "offset": offset, "limit": limit, "patterns": results}

# ══════════════════════════════════════════════════════════════════════════════
# VISUAL SEARCH
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = Query(12)):
    data   = await file.read()
    img    = Image.open(BytesIO(data)).convert("RGB")
    tensor = clip_preprocess(img).unsqueeze(0).to(CLIP_DEVICE)
    with torch.no_grad():
        feat = clip_model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    vec     = feat.squeeze().cpu().tolist()
    results = cosine_search(vec, top_k=top_k)
    for r in results:
        folder = PERIOD_FOLDER.get(r.get("period"), "")
        r["image_url"] = f"/images/{folder}/{r['filename']}" if folder else None
    return {"results": results}

# ══════════════════════════════════════════════════════════════════════════════
# TEXT SEARCH
# ══════════════════════════════════════════════════════════════════════════════
class TextQuery(BaseModel):
    query: str
    top_k: int = 12

@app.post("/api/search/text")
def search_by_text(body: TextQuery):
    tokens = clip_tokenizer([body.query]).to(CLIP_DEVICE)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    vec     = feat.squeeze().cpu().tolist()
    results = cosine_search(vec, top_k=body.top_k)
    for r in results:
        folder = PERIOD_FOLDER.get(r.get("period"), "")
        r["image_url"] = f"/images/{folder}/{r['filename']}" if folder else None
    return {"results": results}

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE — status
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/generate/status")
def generate_status():
    cuda_ok = torch.cuda.is_available()
    vram    = round(torch.cuda.memory_allocated() / 1e9, 2) if cuda_ok else 0
    return {
        "loaded":       lora_pipeline is not None,
        "device":       "cuda" if cuda_ok else "cpu",
        "vram_used_gb": vram,
    }

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE — generate
# ══════════════════════════════════════════════════════════════════════════════
class GenerateRequest(BaseModel):
    prompt:         str
    period:         str
    num_images:     int   = 3
    num_steps:      int   = 30
    guidance_scale: float = 7.5

@app.post("/api/generate")
def generate(body: GenerateRequest):
    global lora_pipeline
    if body.num_images < 1 or body.num_images > 5:
        raise HTTPException(400, "num_images must be 1-5")

    if lora_pipeline is None:
        from diffusers import StableDiffusionXLPipeline
        print("Loading SDXL + LoRA ...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        )
        pipe.load_lora_weights(LORA_PATH)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        pipe.set_progress_bar_config(disable=True)
        lora_pipeline = pipe
        print("Pipeline ready.")

    label = PERIOD_LABEL.get(body.period, body.period)
    full_prompt = (
        f"{body.prompt}, Vietnamese {label} ornamental pattern, "
        f"traditional Dai Viet art style, black and white line art, "
        f"highly detailed, cultural heritage"
    ).strip(", ")

    start  = time.time()
    images = lora_pipeline(
        prompt=full_prompt,
        num_images_per_prompt=body.num_images,
        num_inference_steps=body.num_steps,
        guidance_scale=body.guidance_scale,
        height=512, width=512,
    ).images
    elapsed = round(time.time() - start, 1)

    results = []
    for img in images:
        buf = BytesIO(); img.save(buf, format="PNG")
        results.append(base64.b64encode(buf.getvalue()).decode())

    return {"images": results, "prompt_used": full_prompt, "generation_time_sec": elapsed}

# ══════════════════════════════════════════════════════════════════════════════
# SAVE GENERATED
# ══════════════════════════════════════════════════════════════════════════════
class SaveRequest(BaseModel):
    image_base64: str
    period:       str
    prompt:       str

@app.post("/api/save_generated")
def save_generated(body: SaveRequest):
    img_id    = str(uuid.uuid4())
    img       = Image.open(BytesIO(base64.b64decode(body.image_base64))).convert("RGB")
    png_path  = GENERATED_DIR / f"{img_id}.png"
    img.save(str(png_path))
    (GENERATED_DIR / f"{img_id}.json").write_text(
        json.dumps({"id": img_id, "period": body.period, "prompt": body.prompt},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tensor = clip_preprocess(img).unsqueeze(0).to(CLIP_DEVICE)
    with torch.no_grad():
        feat = clip_model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    emb = feat.squeeze().cpu().tolist()

    conn = get_conn()
    cur  = conn.execute("""
        INSERT INTO patterns (filename, period, style, is_generated, gen_prompt, clip_embedding)
        VALUES (?, ?, 'line_art', 1, ?, ?)
    """, (f"{img_id}.png", body.period, body.prompt, json.dumps(emb)))
    db_id = cur.lastrowid
    conn.commit(); conn.close()

    if MANIFEST_CSV.exists():
        with open(str(MANIFEST_CSV), "a", encoding="utf-8") as f:
            f.write(f"\n{img_id}.png,{body.period},line_art,false,generated,,,")

    return {"success": True, "id": img_id, "db_id": db_id}
