"""Evaluación sobre un set anotado (JSONL).

Cada línea:
  {"id": "q01", "question": "...", "answerable": true,
   "relevant": ["reglamento#art-5"],          # artículos que contienen la respuesta
   "must_contain": ["tres días"]}             # opcional: texto esperado en la respuesta

Métricas:
  * Recuperación (solo preguntas respondibles): hit@1/3/5 y MRR@5, con IC95 % de Wilson.
  * Abstención: tasa de abstención correcta (no respondibles) y falsa abstención (respondibles).
  * Calidad de respuesta: citas verificables y presencia de texto esperado.
  * Operación: latencia media/p95, tokens y costo estimado (si hay precios declarados).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pipeline import Pipeline
from .text import normalize


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _prop(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "value": (k / n) if n else float("nan"), "ci95": [lo, hi]}


def load_evalset(path: str | Path) -> List[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                it = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: JSON inválido ({e})") from e
            for key in ("id", "question", "answerable"):
                if key not in it:
                    raise ValueError(f"{path}:{ln}: falta el campo '{key}'")
            if it["answerable"] and not it.get("relevant"):
                raise ValueError(f"{path}:{ln}: pregunta respondible sin 'relevant'")
            items.append(it)
    if not items:
        raise ValueError("Set de evaluación vacío")
    return items


def _rank_of_relevant(hits, relevant: List[str]) -> Optional[int]:
    rel = set(relevant)
    for rank, h in enumerate(hits, start=1):
        if h.chunk.base_id in rel:
            return rank
    return None


def evaluate(pipe: Pipeline, items: List[dict], k: int = 5) -> Dict:
    ans_items = [it for it in items if it["answerable"]]
    unans_items = [it for it in items if not it["answerable"]]

    # --- recuperación (independiente de la abstención)
    ranks = []
    for it in ans_items:
        hits = pipe.retriever.search(it["question"], k=k)
        ranks.append(_rank_of_relevant(hits, it["relevant"]))
    n = len(ans_items)
    retrieval = {
        f"hit@{c}": _prop(sum(1 for r in ranks if r is not None and r <= c), n) for c in (1, 3, 5) if c <= k
    }
    rr = [1.0 / r if r else 0.0 for r in ranks]
    retrieval["mrr@5"] = {"value": float(np.mean(rr)) if rr else float("nan"), "n": n}

    # --- pipeline completo
    answers = {it["id"]: pipe.answer(it["question"]) for it in items}
    false_abst = sum(1 for it in ans_items if answers[it["id"]].abstained)
    true_abst = sum(1 for it in unans_items if answers[it["id"]].abstained)
    delivered = [(it, answers[it["id"]]) for it in ans_items if not answers[it["id"]].abstained]
    with_text = [(it, a) for it, a in delivered if it.get("must_contain")]
    contain_ok = sum(
        1 for it, a in with_text if all(normalize(s) in normalize(a.answer) for s in it["must_contain"])
    )
    cited_ok = sum(1 for _, a in delivered if a.citations)

    lat = np.array([a.latency_ms for a in answers.values()])
    in_tok = [a.input_tokens for a in answers.values() if a.input_tokens is not None]
    out_tok = [a.output_tokens for a in answers.values() if a.output_tokens is not None]
    costs = [a.cost_usd for a in answers.values() if a.cost_usd is not None]

    return {
        "n_total": len(items),
        "n_answerable": n,
        "n_unanswerable": len(unans_items),
        "generator": pipe.generator.name,
        "min_coverage": pipe.cfg.min_coverage,
        "retrieval": retrieval,
        "abstention": {
            "correct_on_unanswerable": _prop(true_abst, len(unans_items)),
            "false_on_answerable": _prop(false_abst, n),
        },
        "answers": {
            "with_verified_citation": _prop(cited_ok, len(delivered)),
            "contains_expected_text": _prop(contain_ok, len(with_text)),
            "flagged_for_human_review": _prop(sum(1 for a in answers.values() if a.needs_human_review), len(answers)),
        },
        "operation": {
            "latency_ms_mean": float(lat.mean()),
            "latency_ms_p95": float(np.percentile(lat, 95)),
            "input_tokens_total": int(sum(in_tok)) if in_tok else None,
            "output_tokens_total": int(sum(out_tok)) if out_tok else None,
            "cost_usd_total": float(sum(costs)) if costs else None,
        },
        "failures": {
            "retrieval_misses": [it["id"] for it, r in zip(ans_items, ranks) if r is None],
            "false_abstentions": [it["id"] for it in ans_items if answers[it["id"]].abstained],
            "missed_abstentions": [it["id"] for it in unans_items if not answers[it["id"]].abstained],
        },
    }


def _fmt(p: dict) -> str:
    if "ci95" not in p:
        return f"{p['value']:.3f} (n={p['n']})"
    lo, hi = p["ci95"]
    return f"{p['value']:.3f} [{lo:.3f}, {hi:.3f}] ({p['k']}/{p['n']})"


def to_markdown(r: Dict) -> str:
    L = [f"### Evaluación ({r['generator']}, cobertura mínima {r['min_coverage']})", ""]
    L += [f"Preguntas: {r['n_total']} ({r['n_answerable']} respondibles, {r['n_unanswerable']} no respondibles)", ""]
    L += ["| Métrica | Valor [IC95 % Wilson] |", "|---|---|"]
    for k, v in r["retrieval"].items():
        L.append(f"| Recuperación {k} | {_fmt(v)} |")
    a = r["abstention"]
    L.append(f"| Abstención correcta (no respondibles) | {_fmt(a['correct_on_unanswerable'])} |")
    L.append(f"| Falsa abstención (respondibles) | {_fmt(a['false_on_answerable'])} |")
    for k, label in (("with_verified_citation", "Respuestas con cita verificable"),
                     ("contains_expected_text", "Respuestas con el texto esperado"),
                     ("flagged_for_human_review", "Marcadas para revisión humana")):
        L.append(f"| {label} | {_fmt(r['answers'][k])} |")
    o = r["operation"]
    L.append(f"| Latencia media / p95 (ms) | {o['latency_ms_mean']:.1f} / {o['latency_ms_p95']:.1f} |")
    if o["input_tokens_total"] is not None:
        L.append(f"| Tokens entrada / salida | {o['input_tokens_total']} / {o['output_tokens_total']} |")
    if o["cost_usd_total"] is not None:
        L.append(f"| Costo estimado (USD) | {o['cost_usd_total']:.4f} |")
    f = r["failures"]
    L += ["", f"Fallos de recuperación: {f['retrieval_misses'] or 'ninguno'}  ",
          f"Falsas abstenciones: {f['false_abstentions'] or 'ninguna'}  ",
          f"Abstenciones omitidas: {f['missed_abstentions'] or 'ninguna'}"]
    return "\n".join(L) + "\n"
