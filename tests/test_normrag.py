"""Pruebas sin red ni claves: corpus ficticio y proveedor simulado."""

import json
from pathlib import Path

import pytest

from normrag.chunking import chunk_document, load_corpus
from normrag.evaluate import evaluate, load_evalset, to_markdown, wilson
from normrag.generate import NO_SUPPORT, ExtractiveGenerator, LLMGenerator
from normrag.pipeline import Pipeline, PipelineConfig
from normrag.providers import MockProvider, estimate_cost_usd
from normrag.retriever import BM25Retriever
from normrag.text import tokenize

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"


@pytest.fixture(scope="module")
def chunks():
    return load_corpus(DATA)


@pytest.fixture(scope="module")
def retriever(chunks):
    return BM25Retriever(chunks)


def make_pipe(retriever, provider_fn=None, **cfg):
    gen = LLMGenerator(MockProvider(provider_fn)) if provider_fn else ExtractiveGenerator()
    return Pipeline(retriever, gen, PipelineConfig(**cfg))


# ------------------------------------------------------------------ texto y fragmentos


def test_tokenize_strips_accents_stopwords_and_plurals():
    assert tokenize("¿Cuáles son los plazos de las ofertas?") == ["son", "plazo", "oferta"]
    assert tokenize("Garantía de mantenimiento de oferta") == ["garantia", "mantenimiento", "oferta"]


def test_chunking_by_article(chunks):
    arts = [c for c in chunks if c.label.startswith("Art.")]
    assert len(arts) == 15
    assert any(c.label == "Preámbulo" for c in chunks)
    c = next(c for c in chunks if c.label == "Art. 2")
    assert c.base_id == "reglamento_ficticio#art-2"
    assert c.ref == "reglamento_ficticio, Art. 2"


def test_long_article_is_split_with_part_suffix():
    long_body = " ".join(["La entidad compra cosas importantes."] * 80)
    chunks = chunk_document("doc", f"Artículo 1.- {long_body}\n\nArtículo 2.- Texto corto.", max_chars=400)
    parts = [c for c in chunks if c.label == "Art. 1"]
    assert len(parts) > 1
    assert parts[0].id == "doc#art-1" and parts[1].id == "doc#art-1/p2"
    assert all(c.base_id == "doc#art-1" for c in parts)


def test_document_without_articles_uses_paragraph_labels():
    chunks = chunk_document("memo", "Primer párrafo.\n\nSegundo párrafo.", max_chars=1000)
    assert chunks and chunks[0].label.startswith("§")


# ------------------------------------------------------------------ recuperación


def test_retrieval_top1_for_clear_question(retriever):
    hits = retriever.search("¿Cuál es la multa por mora en la entrega?", k=3)
    assert hits[0].chunk.base_id == "reglamento_ficticio#art-12"


def test_retrieval_returns_empty_for_out_of_vocabulary(retriever):
    assert retriever.search("capital de Francia", k=3) == []


# ------------------------------------------------------------------ pipeline


def test_extractive_answer_has_verifiable_citation_and_log(retriever, tmp_path):
    log = tmp_path / "q.jsonl"
    pipe = make_pipe(retriever, log_path=str(log))
    a = pipe.answer("¿Hasta qué monto se puede hacer una compra directa?")
    assert not a.abstained and a.reason == "ok"
    assert "reglamento_ficticio, Art. 2" in a.citations
    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert rec["question"].startswith("¿Hasta qué monto") and "ts" in rec and "latency_ms" in rec


def test_abstains_without_support_and_does_not_call_llm(retriever):
    provider_calls = []
    pipe = make_pipe(retriever, lambda s, u: provider_calls.append(1) or "x [1]")
    a = pipe.answer("¿Cuál es la capital de Francia?")
    assert a.abstained and a.reason == "cobertura_baja" and not provider_calls


def test_llm_answer_with_valid_citation_is_delivered(retriever):
    pipe = make_pipe(retriever, lambda s, u: "El plazo es de quince días hábiles [1].")
    a = pipe.answer("¿Cuánto es el plazo mínimo para recibir ofertas en licitación pública?")
    assert not a.abstained and a.citations and a.needs_human_review  # menciona plazo: revisión humana


def test_llm_invalid_citation_is_discarded_but_kept_for_audit(retriever):
    pipe = make_pipe(retriever, lambda s, u: "Es de quince días [9].")
    a = pipe.answer("¿Cuánto es el plazo mínimo para recibir ofertas en licitación pública?")
    assert a.abstained and a.reason == "citas_invalidas"
    assert a.raw_answer == "Es de quince días [9]." and a.needs_human_review


def test_llm_answer_without_citation_is_discarded(retriever):
    pipe = make_pipe(retriever, lambda s, u: "Es de quince días hábiles.")
    a = pipe.answer("¿Cuánto es el plazo mínimo para recibir ofertas en licitación pública?")
    assert a.abstained and a.reason == "citas_invalidas"


def test_llm_declares_no_support(retriever):
    pipe = make_pipe(retriever, lambda s, u: NO_SUPPORT)
    a = pipe.answer("¿Cuál es el plazo mínimo para recibir ofertas?")
    assert a.abstained and a.reason == "modelo_sin_sustento"


def test_prompt_contains_numbered_fragments_and_rules(retriever):
    seen = {}
    pipe = make_pipe(retriever, lambda s, u: seen.update(system=s, user=u) or "Texto [1].")
    pipe.answer("¿Cuál es el plazo mínimo para recibir ofertas?")
    assert NO_SUPPORT in seen["system"]
    assert "[1] (reglamento_ficticio" in seen["user"] and "Pregunta:" in seen["user"]


# ------------------------------------------------------------------ costo


def test_cost_is_none_without_declared_prices(monkeypatch):
    monkeypatch.delenv("NORMRAG_PRICE_IN_PER_MTOK", raising=False)
    monkeypatch.delenv("NORMRAG_PRICE_OUT_PER_MTOK", raising=False)
    assert estimate_cost_usd(1000, 200) is None


def test_cost_uses_declared_prices(monkeypatch):
    monkeypatch.setenv("NORMRAG_PRICE_IN_PER_MTOK", "2")
    monkeypatch.setenv("NORMRAG_PRICE_OUT_PER_MTOK", "10")
    assert estimate_cost_usd(1_000_000, 100_000) == pytest.approx(3.0)


# ------------------------------------------------------------------ evaluación


def test_wilson_interval_sanity():
    lo, hi = wilson(8, 10)
    assert 0.4 < lo < 0.8 < hi < 1.0
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # NaN


def test_evalset_loader_validates(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id":"a","question":"?","answerable":true}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_evalset(bad)


def test_synthetic_evaluation_runs_and_reports_known_failures(retriever):
    pipe = make_pipe(retriever)
    res = evaluate(pipe, load_evalset(DATA / "eval_synthetic.jsonl"))
    assert res["retrieval"]["hit@3"]["value"] >= 0.9
    assert res["answers"]["with_verified_citation"]["value"] == 1.0
    # Límites léxicos conocidos del corpus de prueba (se documentan en el README):
    assert set(res["failures"]["false_abstentions"]) <= {"q12", "q15"}
    assert set(res["failures"]["missed_abstentions"]) <= {"u05"}
    assert "Recuperación hit@3" in to_markdown(res)
