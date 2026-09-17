"""Ingestion en streaming d'un ConversionResult Docling vers SQLite.

Pipeline : HTTP -> fichier temporaire -> ijson (2 passes) -> SQLite.
Pic memoire borne a ~1 page, quelle que soit la taille du document.

    ingest_url("https://.../convert", {"doc_id": 42}, "doc.db")
    store = WordStore("doc.db")
    words = store.get_page(2)          # numpy + liste de str
    doc = store.get_document()         # DoclingDocument pour le HybridChunker

Dependances : ijson (backend C recommande), httpx. numpy et docling-core optionnels.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import ijson

# Coordonnees stockees en entiers : points * SCALE, arrondis.
SCALE = 10
BATCH = 5000

SCHEMA = """
PRAGMA page_size = 8192;

-- meta porte le DoclingDocument compresse et les dimensions de page.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value BLOB
) WITHOUT ROWID;

-- (page, idx) en cle primaire : les mots d'une page sont physiquement
-- contigus, et aucun index secondaire n'est necessaire.
CREATE TABLE IF NOT EXISTS words (
    page  INTEGER NOT NULL,
    idx   INTEGER NOT NULL,
    l     INTEGER NOT NULL,
    t     INTEGER NOT NULL,
    r     INTEGER NOT NULL,
    b     INTEGER NOT NULL,
    conf  INTEGER,            -- confiance * 100, ou NULL
    text  TEXT NOT NULL,
    PRIMARY KEY (page, idx)
) WITHOUT ROWID;
"""


# --------------------------------------------------------------- geometrie


def rect_to_ltrb(rect: dict[str, Any]) -> tuple[float, float, float, float]:
    """Normalise les formes de rectangle rencontrees dans le JSON Docling.

    BoundingRectangle serialise 4 points (r_x0..r_y3) ; BoundingBox
    serialise l/t/r/b. On accepte les deux et on renvoie une bbox droite.
    """
    if "r_x0" in rect:
        xs = [rect["r_x0"], rect["r_x1"], rect["r_x2"], rect["r_x3"]]
        ys = [rect["r_y0"], rect["r_y1"], rect["r_y2"], rect["r_y3"]]
        return min(xs), min(ys), max(xs), max(ys)
    l, t, r, b = rect["l"], rect["t"], rect["r"], rect["b"]
    return min(l, r), min(t, b), max(l, r), max(t, b)


def to_top_left(
    ltrb: tuple[float, float, float, float], origin: str | None, height: float | None
) -> tuple[float, float, float, float]:
    """Ramene une bbox en origine haut-gauche."""
    if origin != "BOTTOMLEFT" or not height:
        return ltrb
    l, t, r, b = ltrb
    return l, height - b, r, height - t


def _cell_fields(cell: dict[str, Any]) -> tuple[dict | None, str | None]:
    rect = cell.get("rect") or cell.get("bbox")
    if rect is None:
        return None, None
    origin = rect.get("coord_origin")
    if origin is None and "rect" in cell:
        # BoundingRectangle porte l'origine a la racine du rect
        origin = cell["rect"].get("coord_origin")
    return rect, origin


# --------------------------------------------------------------- ingestion


def download(url: str, payload: dict[str, Any], dest: Path, timeout=None) -> Path:
    """Telecharge la reponse en streaming, sans jamais la charger en RAM."""
    import httpx

    with httpx.stream(
        "POST", url, json=payload, timeout=timeout,
        headers={"Accept-Encoding": "gzip"},
    ) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    return dest


def _connect(db_path: str | Path) -> sqlite3.Connection:
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA foreign_keys = OFF")
    return db


def _iter_word_rows(page: dict[str, Any]) -> Iterator[tuple]:
    """Aplati les word_cells d'une page en lignes pretes pour executemany."""
    page_no = page.get("page_no")
    if page_no is None:
        return
    parsed = page.get("parsed_page") or {}
    size = parsed.get("dimension", {}).get("rect") or {}
    height = page.get("size", {}).get("height") or size.get("height")

    for idx, cell in enumerate(parsed.get("word_cells") or ()):
        rect, origin = _cell_fields(cell)
        if rect is None:
            continue
        l, t, r, b = to_top_left(rect_to_ltrb(rect), origin, height)
        conf = cell.get("confidence")
        yield (
            page_no,
            idx,
            round(l * SCALE),
            round(t * SCALE),
            round(r * SCALE),
            round(b * SCALE),
            None if conf is None else round(conf * 100),
            cell.get("text") or "",
        )


def ingest_file(json_path: str | Path, db_path: str | Path) -> None:
    """Deux passes ijson sur le ConversionResult : document, puis pages."""
    json_path, db_path = Path(json_path), Path(db_path)
    if db_path.exists():
        db_path.unlink()

    db = _connect(db_path)
    db.executescript(SCHEMA)
    # Import en masse : pas de journal, on rejoue tout en cas d'echec.
    db.execute("PRAGMA journal_mode = OFF")
    db.execute("PRAGMA synchronous = OFF")

    # --- passe 1 : le DoclingDocument, stocke compresse
    with json_path.open("rb") as f:
        doc = next(ijson.items(f, "document", use_float=True), None)
    if doc is not None:
        blob = zlib.compress(json.dumps(doc, ensure_ascii=False).encode(), 6)
        db.execute("INSERT OR REPLACE INTO meta VALUES ('document', ?)", (blob,))
        del doc, blob

    # --- passe 2 : les pages, une par une
    buf: list[tuple] = []
    sizes: dict[str, tuple[float | None, float | None]] = {}
    with json_path.open("rb") as f:
        for page in ijson.items(f, "pages.item", use_float=True):
            rows = list(_iter_word_rows(page))
            buf.extend(rows)
            size = page.get("size") or {}
            sizes[str(page.get("page_no"))] = (size.get("width"), size.get("height"))
            if len(buf) >= BATCH:
                db.executemany("INSERT OR REPLACE INTO words VALUES (?,?,?,?,?,?,?,?)", buf)
                buf.clear()
            del page, rows
    if buf:
        db.executemany("INSERT OR REPLACE INTO words VALUES (?,?,?,?,?,?,?,?)", buf)
        buf.clear()

    db.execute(
        "INSERT OR REPLACE INTO meta VALUES ('page_sizes', ?)",
        (json.dumps(sizes).encode(),),
    )
    db.commit()
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("VACUUM")
    db.close()


def ingest_url(url: str, payload: dict[str, Any], db_path: str | Path) -> None:
    """Telecharge puis ingere, en supprimant le JSON temporaire."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = download(url, payload, Path(tmp) / "conversion.json")
        ingest_file(raw, db_path)


# --------------------------------------------------------------- lecture


@dataclass
class PageWords:
    page: int
    texts: list[str]
    boxes: Any  # numpy (n, 4) float32 en points, origine haut-gauche

    def __len__(self) -> int:
        return len(self.texts)


class WordStore:
    """Acces paresseux : rien n'est charge tant qu'on ne le demande pas."""

    def __init__(self, db_path: str | Path):
        self.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.db.execute("PRAGMA mmap_size = 268435456")  # 256 Mo

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def page_numbers(self) -> list[int]:
        # `page` est la premiere colonne de la cle primaire : parcours d'index.
        return [
            r[0] for r in self.db.execute("SELECT DISTINCT page FROM words ORDER BY page")
        ]

    @property
    def page_sizes(self) -> dict[int, tuple[float, float]]:
        row = self.db.execute("SELECT value FROM meta WHERE key = 'page_sizes'").fetchone()
        if row is None:
            return {}
        return {int(k): tuple(v) for k, v in json.loads(row[0]).items()}

    def page_size(self, page: int) -> tuple[float, float] | None:
        return self.page_sizes.get(page)

    def get_page(self, page: int) -> PageWords:
        rows = self.db.execute(
            "SELECT l, t, r, b, text FROM words WHERE page = ? ORDER BY idx", (page,)
        ).fetchall()
        texts = [row[4] for row in rows]
        try:
            import numpy as np

            boxes = (
                np.asarray([row[:4] for row in rows], dtype="float32").reshape(-1, 4)
                / SCALE
            )
        except ImportError:
            boxes = [tuple(v / SCALE for v in row[:4]) for row in rows]
        return PageWords(page=page, texts=texts, boxes=boxes)

    def words_in_bbox(
        self, page: int, ltrb: tuple[float, float, float, float], min_overlap=0.5
    ) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Mots dont l'aire est couverte a min_overlap par la bbox donnee.

        Sert au highlight : passer ici la prov[].bbox d'un item Docling,
        ramenee au prealable en origine haut-gauche.
        """
        L, T, R, B = (v * SCALE for v in ltrb)
        out = []
        for l, t, r, b, text in self.db.execute(
            "SELECT l, t, r, b, text FROM words "
            "WHERE page = ? AND r >= ? AND l <= ? AND b >= ? AND t <= ? ORDER BY idx",
            (page, L, R, T, B),
        ):
            inter = max(0, min(r, R) - max(l, L)) * max(0, min(b, B) - max(t, T))
            area = max((r - l) * (b - t), 1)
            if inter / area >= min_overlap:
                out.append(
                    (text, (l / SCALE, t / SCALE, r / SCALE, b / SCALE))
                )
        return out

    def get_document(self):
        """Reconstruit le DoclingDocument (necessaire au HybridChunker)."""
        row = self.db.execute("SELECT value FROM meta WHERE key = 'document'").fetchone()
        if row is None:
            return None
        data = json.loads(zlib.decompress(row[0]))
        from docling_core.types.doc import DoclingDocument

        doc = DoclingDocument.model_validate(data)
        del data
        return doc


if __name__ == "__main__":
    import sys

    ingest_file(sys.argv[1], sys.argv[2])
    with WordStore(sys.argv[2]) as store:
        pages = store.page_numbers()
        print(f"{len(pages)} pages")
        if pages:
            p = store.get_page(pages[0])
            print(p.page, len(p), p.texts[:10])
