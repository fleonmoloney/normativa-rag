# Cargar normativa real y construir una evaluación propia

El repositorio se entrega probado con un **corpus ficticio**. Para usarlo con normativa real:

## 1. Reunir los textos

1. Descargá vos los textos oficiales (por ejemplo, desde el sitio oficial de normativa del Estado uruguayo,
   IMPO) y guardalos como `.txt` o `.md` en una carpeta, p. ej. `data/real/`. Un archivo por norma.
2. Usá nombres de archivo estables y legibles (`tocaf.txt`, `decreto_xx.txt`): el nombre pasa a ser el
   identificador citado.
3. Revisá que los artículos empiecen en línea nueva con "Artículo N" o "Art. N". Si el texto copiado
   perdió los saltos de línea, la segmentación caerá en párrafos y las citas dirán `§n` en lugar de `Art. n`.
4. Anotá en `data/real/FUENTES.md` la URL, la fecha de descarga y la versión de cada texto.
   Las normas se modifican: sin esa trazabilidad no se puede saber qué versión respondió.

> No se incluyen textos legales reales en este repositorio. Verificá la licencia o las condiciones de
> reutilización de cada fuente antes de redistribuir.

## 2. Construir el set de evaluación (lo más valioso del repositorio)

Un sistema sin evaluación propia no demuestra nada. Con tu experiencia en pliegos y compras, anotá al menos
**30 preguntas**:

- 20 o más **respondibles**, con el/los artículos que contienen la respuesta (`relevant`) y, si es posible,
  el texto literal esperado (`must_contain`).
- 8 o más **no respondibles**, incluyendo casos difíciles: preguntas que comparten vocabulario con la norma
  pero cuya respuesta no está allí.
- Variedad de redacción: preguntas con las palabras de la norma y otras con lenguaje cotidiano.

Formato (una línea JSON por pregunta; ver `data/synthetic/eval_synthetic.jsonl`):

```json
{"id":"q01","question":"...","answerable":true,"relevant":["tocaf#art-33"],"must_contain":["..."]}
```

Se recomienda que **otra persona** revise las anotaciones y que el set se congele antes de ajustar
`min_coverage`, para no sobreajustar el umbral al propio set.

## 3. Evaluar y reportar

```bash
normrag eval --docs data/real --evalset data/real/eval.jsonl --out results/eval_real.md
```

Reportá siempre los intervalos de confianza, la lista de fallos y los límites. Con pocas preguntas los
intervalos son amplios: decirlo es parte de la honestidad del resultado.

## 4. Con LLM (opcional)

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...        # clave propia
export NORMRAG_MODEL=<modelo>       # identificador del modelo a usar
# Opcional, para estimar costo: precios vigentes de tu proveedor, por millón de tokens
export NORMRAG_PRICE_IN_PER_MTOK=...
export NORMRAG_PRICE_OUT_PER_MTOK=...
normrag eval --llm --docs data/real --evalset data/real/eval.jsonl --out results/eval_real_llm.md --log logs/queries.jsonl
```

Compará siempre contra la línea base extractiva: si el LLM no mejora las métricas de forma medible, no
justifica su costo ni su riesgo.
