"""
db.py — SQLite database layer (drop-in replacement for pgvector).
Stores embeddings as JSON blobs; cosine similarity done in numpy.
"""
import sqlite3, json, numpy as np
from pathlib import Path

DB_PATH = Path("D:/Hoavandaiviet/daiviet_platform/daiviet.db")

def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS patterns (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        filename        TEXT NOT NULL,
        period          TEXT,
        style           TEXT,
        multi_motif     INTEGER,        -- 0/1/NULL
        motif_subject   TEXT,
        artifact_source TEXT,
        cropped_from    TEXT,
        source_url      TEXT,
        is_generated    INTEGER DEFAULT 0,
        gen_prompt      TEXT,
        clip_embedding  TEXT            -- JSON list[float]
    );
    CREATE INDEX IF NOT EXISTS idx_period   ON patterns (period);
    CREATE INDEX IF NOT EXISTS idx_style    ON patterns (style);
    CREATE INDEX IF NOT EXISTS idx_gen      ON patterns (is_generated);
    """)
    conn.commit()
    conn.close()

def cosine_search(query_vec: list, top_k: int = 12, generated_only=False):
    """Return top_k rows sorted by cosine similarity to query_vec."""
    conn   = get_conn()
    where  = "clip_embedding IS NOT NULL"
    if not generated_only:
        where += " AND is_generated = 0"
    rows   = conn.execute(
        f"SELECT * FROM patterns WHERE {where}"
    ).fetchall()
    conn.close()

    if not rows:
        return []

    qv    = np.array(query_vec, dtype=np.float32)
    qv   /= (np.linalg.norm(qv) + 1e-9)
    sims  = []
    for r in rows:
        ev   = np.array(json.loads(r["clip_embedding"]), dtype=np.float32)
        ev  /= (np.linalg.norm(ev) + 1e-9)
        sim  = float(np.dot(qv, ev))
        sims.append((sim, dict(r)))

    sims.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, row in sims[:top_k]:
        row["similarity"] = round(sim, 4)
        row.pop("clip_embedding", None)   # don't send blob to client
        results.append(row)
    return results
