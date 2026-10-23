"""Normalización y tokenización para español (sin dependencias externas)."""

from __future__ import annotations

import re
import unicodedata

# Lista corta y deliberada: palabras funcionales muy frecuentes.
STOPWORDS = frozenset(
    """a al algo ante con contra cual cuales cuando de del desde donde e el ella ellos en entre es esta
    estan este esto ha han hasta hay la las le les lo los mas me mi muy no nos o para pero por que se
    sea ser si sin sobre su sus te todo un una uno unos unas y ya como cual cuanto cuantos cuanta
    cuantas debe deben puede pueden""".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize(s: str) -> str:
    return strip_accents(s).lower()


def _light_stem(tok: str) -> str:
    """Singulariza plurales simples: 'ofertas' -> 'oferta', 'plazos' -> 'plazo'."""
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    toks = _WORD.findall(normalize(text))
    return [_light_stem(t) for t in toks if t not in STOPWORDS and len(t) > 1]
