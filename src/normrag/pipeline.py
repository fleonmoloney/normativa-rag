"""Pipeline de consulta: recuperar -> decidir si hay sustento -> generar -> verificar -> registrar."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .generate import Generator
from .providers import estimate_cost_usd
from .retriever import BM25Retriever

NOTICE = (
    "Borrador de apoyo basado en los textos indexados. No es asesoramiento jurídico y debe ser "
    "verificado contra la norma vigente por una persona responsable antes de usarse."
)

# Temas donde un error cuesta caro: siempre pasan por revisión humana.
RISK_RE = re.compile(
    r"\b(multa|sanci[oó]n|penalidad|plazo|monto|importe|rescisi[oó]n|adjudic\w*|impugna\w*|recurso|garant[ií]a|\d+)\b",
    re.I,
)


@dataclass
class PipelineConfig:
    k: int = 5  # fragmentos recuperados
    context_k: int = 3  # fragmentos que se pasan al generador
    min_coverage: float = 0.34  # cobertura mínima de términos de la consulta en el mejor fragmento
    log_path: Optional[str] = None


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    reason: str  # "ok" | "cobertura_baja" | "modelo_sin_sustento" | "citas_invalidas"
    citations: List[str] = field(default_factory=list)
    hits: List[dict] = field(default_factory=list)
    needs_human_review: bool = True
    latency_ms: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    generator: str = ""
    model: str = ""
    raw_answer: Optional[str] = None  # solo si se descartó una respuesta, para auditoría
    notice: str = NOTICE


class Pipeline:
    def __init__(self, retriever: BM25Retriever, generator: Generator, cfg: Optional[PipelineConfig] = None):
        self.retriever, self.generator, self.cfg = retriever, generator, cfg or PipelineConfig()

    def answer(self, question: str) -> Answer:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k=self.cfg.k)
        hit_info = [
            {"id": h.chunk.id, "ref": h.chunk.ref, "score": round(h.score, 3), "coverage": round(h.coverage, 3)}
            for h in hits
        ]

        if not hits or hits[0].coverage < self.cfg.min_coverage:
            ans = Answer(
                question=question,
                answer="No encuentro sustento suficiente en los documentos indexados para responder esta consulta.",
                abstained=True,
                reason="cobertura_baja",
                hits=hit_info,
                needs_human_review=False,
                generator=self.generator.name,
            )
        else:
            ctx = hits[: self.cfg.context_k]
            gen = self.generator.generate(question, ctx)
            citations = [ctx[i - 1].chunk.ref for i in gen.cited]
            base = dict(
                question=question, hits=hit_info, generator=self.generator.name, model=gen.model,
                input_tokens=gen.input_tokens, output_tokens=gen.output_tokens,
                cost_usd=estimate_cost_usd(gen.input_tokens, gen.output_tokens),
            )
            if gen.abstained:
                ans = Answer(answer="El modelo indicó que los fragmentos no alcanzan para responder.",
                             abstained=True, reason="modelo_sin_sustento", needs_human_review=False, **base)
            elif gen.invalid_citations or not gen.cited:
                # Una respuesta que no se puede verificar no se entrega.
                ans = Answer(answer="La respuesta generada no se pudo verificar contra los fragmentos citados y fue descartada.",
                             abstained=True, reason="citas_invalidas", needs_human_review=True,
                             raw_answer=gen.text, **base)
            else:
                risky = bool(RISK_RE.search(question) or RISK_RE.search(gen.text))
                ans = Answer(answer=gen.text, abstained=False, reason="ok", citations=citations,
                             needs_human_review=risky, **base)

        ans.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._log(ans)
        return ans

    def _log(self, ans: Answer) -> None:
        if not self.cfg.log_path:
            return
        rec = asdict(ans)
        rec["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        p = Path(self.cfg.log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
