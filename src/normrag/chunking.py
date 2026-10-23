"""Segmentación de normativa por artículo, con metadatos para citar.

Cada fragmento conserva documento y artículo de origen: sin eso no hay cita verificable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

# "Artículo 12.-", "ARTICULO 12°", "Art. 12:" al comienzo de línea.
ARTICLE_RE = re.compile(r"^[ \t]*(?:art[ií]culo|art\.)[ \t]+(\d+[\w\.]*?)[ \t]*[\.\-–:°º]*[ \t]*(?=\S|$)", re.I | re.M)


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    label: str  # "Art. 12", "Preámbulo", "§3"
    part: int  # 1 si el artículo no se subdividió
    text: str

    @property
    def base_id(self) -> str:
        """Identificador a nivel de artículo, p. ej. 'reglamento#art-12'."""
        slug = self.label.lower().replace("art. ", "art-").replace(" ", "-").replace("§", "par-")
        return f"{self.doc_id}#{slug}"

    @property
    def id(self) -> str:
        return self.base_id if self.part == 1 else f"{self.base_id}/p{self.part}"

    @property
    def ref(self) -> str:
        """Referencia legible para citar."""
        suffix = "" if self.part == 1 else f" (parte {self.part})"
        return f"{self.doc_id}, {self.label}{suffix}"


def _split_long(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    # Un párrafo único más largo que el máximo se corta por oraciones.
    final: List[str] = []
    for piece in out:
        if len(piece) <= max_chars:
            final.append(piece)
            continue
        sents, cur = re.split(r"(?<=[\.;:])\s+", piece), ""
        for s in sents:
            if cur and len(cur) + len(s) + 1 > max_chars:
                final.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            final.append(cur)
    return final


def chunk_document(doc_id: str, text: str, max_chars: int = 1200) -> List[Chunk]:
    text = text.replace("\r\n", "\n")
    matches = list(ARTICLE_RE.finditer(text))
    chunks: List[Chunk] = []
    if not matches:
        for i, piece in enumerate(_split_long(text.strip(), max_chars), start=1):
            if piece:
                chunks.append(Chunk(doc_id, f"§{i}", 1, piece))
        return chunks

    pre = text[: matches[0].start()].strip()
    if pre:
        for i, piece in enumerate(_split_long(pre, max_chars), start=1):
            chunks.append(Chunk(doc_id, "Preámbulo", i, piece))

    for k, m in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
        body = text[m.start() : end].strip()
        num = m.group(1).rstrip(".")
        for i, piece in enumerate(_split_long(body, max_chars), start=1):
            chunks.append(Chunk(doc_id, f"Art. {num}", i, piece))
    return chunks


def load_corpus(folder: str | Path, max_chars: int = 1200) -> List[Chunk]:
    """Lee todos los .txt y .md de una carpeta. El nombre del archivo es el doc_id."""
    folder = Path(folder)
    files = sorted(list(folder.glob("*.txt")) + list(folder.glob("*.md")))
    if not files:
        raise FileNotFoundError(f"No hay .txt ni .md en {folder}")
    chunks: List[Chunk] = []
    for f in files:
        chunks.extend(chunk_document(f.stem, f.read_text(encoding="utf-8"), max_chars))
    return chunks
