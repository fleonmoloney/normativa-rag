# normrag — asistente de consulta sobre normativa de compras, con citas verificables

Sistema de recuperación y respuesta sobre normativa (pensado para compras públicas) que prioriza lo que
suele faltar en las demos de RAG: **citas verificables, abstención cuando no hay sustento, evaluación
medible, registro de costo y latencia, y una política escrita de supervisión humana**.

> **English summary.** A small, dependency-light RAG assistant for procurement regulations. BM25
> retrieval by article, citations enforced in code (answers with missing or non-existent citations are
> discarded), abstention on low evidence, an evaluation harness (hit@k, MRR, abstention, Wilson CIs),
> JSONL query logs with latency/tokens/cost, and a written human-oversight policy. Works offline with an
> extractive baseline; an optional LLM provider can be plugged in. **Tested on a fictitious corpus; bring
> your own regulations and your own annotated eval set. Not legal advice.**

## Qué es y qué no es

- **Es** una base técnica reproducible y un marco de evaluación, probada con un corpus **ficticio**.
- No sustituye la lectura de la norma vigente. No incluye textos legales reales (ver [`docs/REAL_CORPUS.md`](docs/REAL_CORPUS.md)).

## Cómo funciona

```
consulta ──► BM25 por artículo ──► ¿cobertura ≥ umbral? ──no──► ABSTENCIÓN (sin llamar al modelo)
                                          │ sí
                                          ▼
                       generador (extractivo | LLM obligado a citar [n])
                                          │
                       ¿citas existentes y presentes? ──no──► DESCARTADA (queda en el registro)
                                          │ sí
                                          ▼
              respuesta + citas + needs_human_review (temas de riesgo) + aviso ──► registro JSONL
```

| Decisión | Motivo |
|---|---|
| Fragmentos por **artículo** con documento y número | Sin origen exacto no hay cita verificable. |
| **BM25** como línea base, sin embeddings | En normativa pesan los términos exactos; es reproducible y no descarga modelos. Los embeddings se enchufan implementando `search`. |
| Citas `[n]` **validadas en código** | El prompt no alcanza: se verifica que cada cita apunte a un fragmento recuperado. |
| Abstención por **cobertura de términos** antes de generar | Ahorra costo y evita respuestas inventadas cuando el corpus no tiene el tema. |
| Temas de riesgo (plazos, montos, multas…) → **revisión humana** | Un error ahí es caro. Ver [`docs/HUMAN_OVERSIGHT.md`](docs/HUMAN_OVERSIGHT.md). |
| Costo estimado **solo con precios declarados por vos** | No se embeben precios que cambian; si no los declarás, el costo es `null`. |
| Intervalos de **Wilson** en cada proporción | Con pocas preguntas, un porcentaje sin intervalo engaña. |

## Resultados (corpus ficticio de pruebas)

Medidos con `normrag eval` sobre `data/synthetic/` (15 preguntas respondibles y 5 no respondibles,
**inventadas para probar el código; no son un benchmark**). Generador extractivo, sin LLM.

| Métrica | Valor [IC95 % Wilson] |
|---|---|
| Recuperación hit@1 | 0,933 [0,702; 0,988] (14/15) |
| Recuperación hit@3 | 1,000 [0,796; 1,000] (15/15) |
| Abstención correcta en no respondibles | 0,800 [0,376; 0,964] (4/5) |
| Falsa abstención en respondibles | 0,133 [0,037; 0,379] (2/15) |
| Respuestas con cita verificable | 1,000 (13/13) |

Informe completo en [`results/eval_synthetic.md`](results/eval_synthetic.md).

**Límites que estas cifras ya muestran:**
- Las 2 falsas abstenciones (`q12`, `q15`) son una **brecha léxica**: la consulta dice "entregar tarde" y
  "sin licitación", y la norma dice "mora" y "compras directas". La cobertura de términos no entiende sinónimos.
- La pregunta no respondible `u05` ("monto máximo para comprar bienes informáticos") **no** se detectó:
  comparte vocabulario con el corpus pero el tema no está. Es el caso difícil que una abstención léxica no cubre.
- Siguiente paso natural: recuperación híbrida (BM25 + embeddings) y una verificación semántica de que la cita
  respalda la afirmación. Con corpus y preguntas reales los números serán distintos: **hay que medirlos**.

**Todavía no hay resultados sobre normativa real ni con LLM.** Se obtienen con los pasos de
[`docs/REAL_CORPUS.md`](docs/REAL_CORPUS.md).

## Uso

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                                    # 18 pruebas, sin red ni claves

normrag ask  --docs data/synthetic "¿Cuál es la multa por mora en la entrega?"
normrag eval --docs data/synthetic --evalset data/synthetic/eval_synthetic.jsonl --out results/eval_synthetic.md

# con LLM (opcional): ver docs/REAL_CORPUS.md
```

Cada respuesta incluye: texto, citas (`documento, Art. n`), fragmentos recuperados con puntaje y cobertura,
motivo de abstención si corresponde, `needs_human_review`, latencia, tokens y costo estimado.

## Estructura

```
src/normrag/
  chunking.py    segmentación por artículo con metadatos
  retriever.py   BM25 propio (sin dependencias)
  generate.py    generador extractivo y generador LLM con citas obligatorias
  providers.py   MockProvider (pruebas) y AnthropicProvider (opcional); costo con precios declarados
  pipeline.py    recuperar → abstenerse → generar → verificar → registrar
  evaluate.py    hit@k, MRR, abstención, citas, latencia/tokens/costo, IC de Wilson
  cli.py         normrag ask | eval
data/synthetic/  corpus ficticio y set de evaluación de prueba
docs/            supervisión humana y guía para normativa real
tests/           18 pruebas
```

## Aviso

Apoyo a la consulta, no asesoramiento jurídico. Toda respuesta debe verificarse contra la norma vigente
antes de usarse. El corpus incluido es ficticio y sus cifras no corresponden a ninguna norma.

## Autor

Fernando León Moloney — Ingeniero en Sistemas, especialización en Ingeniería Biomédica (UdelaR)

Licencia MIT (ver `LICENSE`).
