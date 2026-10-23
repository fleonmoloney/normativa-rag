"""Generación de respuestas: extractiva (sin LLM) y con LLM obligado a citar.

Reglas que se hacen cumplir en código, no solo en el prompt:
  * Toda afirmación debe llevar una cita [n] que apunte a un fragmento realmente recuperado.
  * Si el modelo responde NO_HAY_SUSTENTO, el sistema se abstiene.
  * Citas inexistentes o ausentes invalidan la respuesta (ver pipeline.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

from .providers import Provider
from .retriever import Hit

NO_SUPPORT = "NO_HAY_SUSTENTO"

SYSTEM_PROMPT = f"""Respondés consultas sobre normativa de compras usando EXCLUSIVAMENTE los fragmentos numerados que se te entregan.
Reglas:
1. Cada afirmación termina con la cita del fragmento que la respalda, con el formato [n].
2. No uses conocimiento externo ni completes lo que los fragmentos no dicen.
3. Si los fragmentos no alcanzan para responder, respondé exactamente: {NO_SUPPORT}
4. No des asesoramiento jurídico: describí lo que dice el texto.
5. Respondé en español, de forma breve y literal cuando haya cifras o plazos."""

CITE_RE = re.compile(r"\[(\d+)\]")


def build_user_prompt(question: str, hits: Sequence[Hit]) -> str:
    frags = "\n\n".join(f"[{i}] ({h.chunk.ref})\n{h.chunk.text}" for i, h in enumerate(hits, start=1))
    return f"Fragmentos:\n\n{frags}\n\nPregunta: {question}"


@dataclass
class Generation:
    text: str
    cited: List[int] = field(default_factory=list)  # índices válidos (1..n)
    invalid_citations: List[int] = field(default_factory=list)
    abstained: bool = False
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model: str = ""


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: Sequence[Hit]) -> Generation: ...


class ExtractiveGenerator:
    """Línea base sin LLM: devuelve los pasajes más relevantes, citados."""

    name = "extractive"

    def __init__(self, max_chars: int = 350, n_passages: int = 2):
        self.max_chars, self.n = max_chars, n_passages

    def generate(self, question: str, hits: Sequence[Hit]) -> Generation:
        parts, cited = [], []
        for i, h in enumerate(hits[: self.n], start=1):
            body = re.sub(r"\s+", " ", h.chunk.text).strip()
            snippet = body if len(body) <= self.max_chars else body[: self.max_chars].rsplit(" ", 1)[0] + "…"
            parts.append(f"{snippet} [{i}]")
            cited.append(i)
        return Generation(text="\n".join(parts), cited=cited, model="extractive")


class LLMGenerator:
    name = "llm"

    def __init__(self, provider: Provider, max_tokens: int = 600):
        self.provider, self.max_tokens = provider, max_tokens

    def generate(self, question: str, hits: Sequence[Hit]) -> Generation:
        c = self.provider.complete(SYSTEM_PROMPT, build_user_prompt(question, hits), self.max_tokens)
        text = c.text.strip()
        nums = [int(n) for n in CITE_RE.findall(text)]
        valid = sorted({n for n in nums if 1 <= n <= len(hits)})
        invalid = sorted({n for n in nums if not 1 <= n <= len(hits)})
        return Generation(
            text=text,
            cited=valid,
            invalid_citations=invalid,
            abstained=NO_SUPPORT in text,
            input_tokens=c.input_tokens,
            output_tokens=c.output_tokens,
            model=c.model,
        )
