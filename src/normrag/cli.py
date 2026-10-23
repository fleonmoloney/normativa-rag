"""Línea de comandos.

  normrag ask  --docs data/docs "¿Cuál es el plazo para recibir ofertas?"
  normrag eval --docs data/docs --evalset data/eval.jsonl --out results/eval.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .chunking import load_corpus
from .evaluate import evaluate, load_evalset, to_markdown
from .generate import ExtractiveGenerator, LLMGenerator
from .pipeline import Pipeline, PipelineConfig
from .retriever import BM25Retriever


def build_pipeline(args) -> Pipeline:
    chunks = load_corpus(args.docs, max_chars=args.max_chars)
    retriever = BM25Retriever(chunks)
    if args.llm:
        from .providers import AnthropicProvider

        gen = LLMGenerator(AnthropicProvider())
    else:
        gen = ExtractiveGenerator()
    cfg = PipelineConfig(k=args.k, context_k=args.context_k, min_coverage=args.min_coverage, log_path=args.log)
    return Pipeline(retriever, gen, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(prog="normrag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--docs", required=True, help="Carpeta con .txt/.md de normativa")
        p.add_argument("--llm", action="store_true", help="Usar LLM (requiere NORMRAG_MODEL y ANTHROPIC_API_KEY)")
        p.add_argument("--k", type=int, default=5)
        p.add_argument("--context-k", type=int, default=3)
        p.add_argument("--min-coverage", type=float, default=0.34)
        p.add_argument("--max-chars", type=int, default=1200)
        p.add_argument("--log", default=None, help="Archivo JSONL de registro de consultas")

    pa = sub.add_parser("ask", help="Hacer una pregunta")
    common(pa)
    pa.add_argument("question")
    pa.add_argument("--json", action="store_true")

    pe = sub.add_parser("eval", help="Evaluar sobre un set anotado")
    common(pe)
    pe.add_argument("--evalset", required=True)
    pe.add_argument("--out", default=None, help="Escribe el informe markdown y un .json al lado")

    a = ap.parse_args()
    pipe = build_pipeline(a)

    if a.cmd == "ask":
        r = pipe.answer(a.question)
        if a.json:
            print(json.dumps(asdict(r), ensure_ascii=False, indent=2))
            return
        print(r.answer)
        if r.citations:
            print("\nCitas: " + "; ".join(r.citations))
        print(f"\n[abstención: {r.abstained} ({r.reason}) | revisión humana: {r.needs_human_review} | {r.latency_ms} ms]")
        print(r.notice)
    else:
        res = evaluate(pipe, load_evalset(a.evalset), k=a.k)
        md = to_markdown(res)
        print(md)
        if a.out:
            from pathlib import Path

            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(md, encoding="utf-8")
            Path(a.out).with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
