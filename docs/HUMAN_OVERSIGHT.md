# Política de supervisión humana

Este sistema **asiste** la consulta de normativa; no decide. Esta política describe qué controles
hace el código y cuáles dependen de una persona.

## Principios

1. **Ninguna respuesta es definitiva.** Toda salida incluye el aviso de que es un borrador sujeto a
   verificación contra la norma vigente.
2. **Si no se puede verificar, no se entrega.** Las respuestas sin cita, con citas inexistentes o sin
   sustento en los fragmentos recuperados se descartan (el texto crudo queda en el registro para auditoría).
3. **Ante la duda, abstenerse.** Si el mejor fragmento cubre menos de `min_coverage` de los términos de la
   consulta, el sistema responde que no encuentra sustento y no llama al modelo.
4. **Los temas de riesgo siempre pasan por una persona.** Si la consulta o la respuesta mencionan plazos,
   montos, multas, sanciones, garantías, recursos, adjudicaciones o contienen cifras, el campo
   `needs_human_review` es `true`.
5. **Trazabilidad.** Con `--log`, cada consulta queda en JSONL con fragmentos recuperados, puntajes,
   decisión, latencia, tokens y costo estimado (si se declararon precios).

## Qué hace el código y qué no

| Control | Dónde | Quién |
|---|---|---|
| Cita obligatoria y verificable | `generate.py`, `pipeline.py` | Código |
| Abstención por cobertura baja | `pipeline.py` | Código |
| Marcado de temas de riesgo | `pipeline.py` (`RISK_RE`) | Código |
| Verificar la cita contra el texto oficial vigente | — | **Persona responsable** |
| Decidir qué documentos se indexan y mantenerlos actualizados | — | **Persona responsable** |
| Anotar el set de evaluación con casos reales | — | **Persona experta** |
| Revisar periódicamente los registros y los fallos | — | **Persona responsable** |

## Limitaciones de los controles

- La abstención por cobertura es léxica: puede abstenerse de más cuando la consulta usa sinónimos
  ("entregar tarde" frente a "mora") y puede no abstenerse cuando la consulta comparte palabras con el
  corpus pero pregunta por algo que no está (ver `results/eval_synthetic.md`).
- Verificar que una cita *existe* no prueba que el texto citado *respalde* la afirmación. Esa comprobación
  semántica sigue siendo humana.
- Las normas cambian: el índice puede quedar desactualizado sin que el sistema lo advierta.

## Uso aceptable

Apoyo a la búsqueda y primera lectura por parte de personal que conoce la materia. No debe usarse para
responder a terceros, para resolver recursos ni como fundamento único de un acto administrativo.
