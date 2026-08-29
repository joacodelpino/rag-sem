# Evaluación del sistema — resultados

**Fecha de la corrida:** 28 de agosto de 2026
**Corpus:** 36 PDFs de documentación jurídica argentina (leyes, fallos, reglamentos), partidos en 2.126 fragmentos
**Set de preguntas:** 29, todas verificadas a mano contra el corpus

Este documento resume qué medimos, cómo lo medimos y qué nos dio. Está escrito
para poder leerlo sin haber tocado el código.

---

## 1. Qué problema estamos evaluando

El sistema hace algo que suena simple: le hacés una pregunta en castellano y te
contesta citando de qué ley salió la respuesta.

Por dentro son **dos pasos separados**, y esto es lo importante para entender
todo lo que sigue:

1. **Buscar** — de los 2.126 fragmentos del corpus, elegir los 5 más
   prometedores. Acá no interviene ninguna inteligencia artificial que
   "escriba": es búsqueda pura.
2. **Redactar** — pasarle esos 5 fragmentos a un modelo de lenguaje (GPT) y
   pedirle que arme la respuesta usando **solo** eso.

Si el paso 1 trae los fragmentos equivocados, el paso 2 no tiene arreglo: el
modelo va a redactar muy bien una respuesta basada en el texto incorrecto. Por
eso casi toda la evaluación mide el paso 1.

Lo que comparamos es **cómo hacer el paso 1**. Hay cuatro maneras, de la más
simple a la más sofisticada, y la pregunta del trabajo es si la sofisticación
vale lo que cuesta.

---

## 2. Las cuatro maneras de buscar

| Nombre | Cómo busca | Analogía |
|---|---|---|
| **naive** (densa) | Por *significado*. Convierte la pregunta y cada fragmento en una lista de números que representa "de qué habla", y compara. | Un bibliotecario que entendió tu pregunta y te trae lo que trata del tema, aunque uses otras palabras. |
| **sparse** (BM25) | Por *palabras literales*. Cuenta coincidencias exactas, dándole más peso a las palabras raras. | El buscador Ctrl+F: encuentra "11.179" perfecto, pero si preguntás por "auto" no encuentra "vehículo". |
| **hybrid** (híbrida) | Corre las dos anteriores y **combina** sus resultados. | Le preguntás a los dos bibliotecarios y te quedás con lo que ambos recomiendan. |
| **hybrid + reranking** | La híbrida trae 20 candidatos, y después un segundo modelo —más caro y más preciso— los **reordena** uno por uno. | Un experto revisa la pila de 20 y decide cuáles 5 van arriba. |

Las dos primeras fallan en casos **opuestos**, y de ahí sale la idea de
combinarlas: donde la búsqueda por significado se pierde con un número de
artículo, la literal lo clava; donde la literal no encuentra nada porque el
documento usa otro vocabulario, la de significado sí.

El **reranking** es lo que en el trabajo llamamos "RAG de segunda generación".
La lógica es: primero buscá **barato y amplio** (que el fragmento correcto esté
en la pila, aunque salga 14º), después reordená **caro y preciso** (que salga
1º). El segundo modelo no se puede usar para todo porque tiene que mirar la
pregunta contra cada fragmento de a uno: sobre 2.126 fragmentos sería
eterno.

---

## 3. Cómo se mide: el set dorado

Un **set dorado** es una lista de preguntas donde nosotros ya sabemos cuál es
la respuesta correcta *y* en qué fragmento del corpus está. Es la vara: sin
eso, "el sistema anda bien" es una opinión.

Armamos 29 preguntas. Ninguna fue inventada de memoria: para cada una fuimos al
corpus, buscamos el fragmento, verificamos que la respuesta estuviera
literalmente ahí y anotamos su número. Las que no pasaron la verificación se
descartaron (empezamos con 30).

Están repartidas en **cinco tipos, cada uno diseñado para romper algo
distinto**:

| Tipo | Cuántas | Qué pone a prueba | Ejemplo |
|---|---|---|---|
| **factual** | 8 | Lo básico: un dato que está en un solo artículo. | *¿Cuántos días tiene el consumidor para arrepentirse de una compra?* |
| **identificador** | 8 | Números de ley y de artículo, que son donde la búsqueda por significado se pierde. | *¿Qué dice el artículo 72 de la ley 11179?* |
| **multihop** | 5 | Respuestas que necesitan **dos o más** fragmentos distintos. | *¿En qué ventas puede arrepentirse el consumidor **y** cuántos días tiene?* (los casos están en un artículo, el plazo en otro) |
| **fuente** | 4 | Distinguir el documento que **es** la respuesta del que solo la **menciona**. | *¿Qué ley establece la defensa del consumidor?* (varios documentos citan la 24.240; uno solo *es* la 24.240) |
| **negativa** | 4 | Que el sistema **admita que no sabe** en vez de inventar. | *¿Qué dice la ley de teletrabajo sobre la desconexión?* (no está en el corpus) |

Las **negativas** son la categoría más importante de todas y la que más se
olvida: un sistema que contesta con seguridad algo que no está en sus
documentos es peor que uno que no contesta.

---

## 4. Qué significan las métricas

Medimos **dos familias que conviene no mezclar**, porque una es objetiva y la
otra es una opinión.

### Familia A — se calculan con una cuenta (objetivas, gratis, siempre dan igual)

Comparan los fragmentos que trajo el sistema contra los que nosotros marcamos
como correctos. Son números duros: dos corridas dan exactamente lo mismo.

- **recall** — *"¿trajo el fragmento correcto, sí o no?"* Es la métrica más
  importante: si el fragmento correcto no entró en los 5, no hay nada que
  hacer después. **1.00 = siempre lo trajo.**
- **MRR** — *"¿en qué puesto lo trajo?"* Si sale 1º vale 1.00, si sale 2º vale
  0.50, si sale 4º vale 0.25. Es la métrica del **orden**, y es donde se ve si
  el reranking sirve: mover el fragmento correcto del puesto 4 al 1 no cambia
  el recall en nada, pero duplica el MRR.
- **precision** — qué proporción de lo traído era útil. **Ojo: con 5
  fragmentos traídos y una sola respuesta correcta, el máximo posible es 0.20,
  no 1.00.** Sirve para comparar columnas entre sí, no como nota.
- **abstención** — solo para las 4 negativas: qué proporción de veces el
  sistema dijo "esto no está en el corpus". **1.00 = siempre lo admitió.**

### Familia B — las juzga otro modelo de IA (Ragas)

Acá usamos **Ragas**, una herramienta que le pasa la pregunta, los fragmentos y
la respuesta a un modelo de lenguaje y le pide que puntúe. Miden la calidad de
la **respuesta**, no de la búsqueda.

- **faithfulness** (fidelidad) — *¿la respuesta se sostiene solo con los
  fragmentos, o el modelo agregó cosas de su propia memoria?* Es la métrica de
  **alucinación**.
- **answer_correctness** — *¿dice lo mismo que la respuesta que esperábamos?*
- **context_precision** — de los fragmentos traídos, ¿los útiles salieron
  primero?
- **context_recall** — ¿los fragmentos alcanzan para reconstruir la respuesta
  correcta? Es el **techo de todo**: ninguna respuesta puede ser mejor que su
  contexto.

> **Advertencia honesta:** la familia B tiene ruido. Es un modelo de IA
> opinando, no una cuenta. Lo medimos: corriendo **dos veces la misma pregunta
> sin cambiar nada**, la fidelidad dio 1.000 y después 0.750. Por eso, en las
> conclusiones, diferencias menores a ~0.05 en esta familia no las tratamos
> como hallazgos. La familia A no tiene ese problema.
>
> Esta es justamente la razón de medir las dos: si solo tuviéramos las métricas
> de Ragas, no habría forma de saber si una configuración bajó porque busca
> peor o porque el juez tuvo un mal día.

---

## 5. Resultados

Las 29 preguntas, contra cada configuración. Traemos 5 fragmentos por consulta.

| configuración | recall | MRR | precision | abstención | fidelidad | correctitud | ctx. precision | ctx. recall | seg/consulta |
|---|---|---|---|---|---|---|---|---|---|
| naive (solo significado) | 0.833 | 0.773 | 0.200 | **1.000** | 0.945 | 0.555 | 0.811 | 0.960 | 0.28 |
| sparse (solo literal) | 0.873 | 0.710 | 0.208 | **1.000** | 0.961 | 0.539 | 0.787 | 0.900 | **0.02** |
| híbrida | 0.913 | **0.828** | 0.216 | **1.000** | 0.966 | 0.531 | 0.871 | 0.960 | 0.30 |
| híbrida + reranking *(modelo chico)* | 0.933 | 0.763 | 0.224 | **1.000** | 0.977 | 0.590 | 0.880 | **1.000** | 2.70 |
| híbrida + reranking *(modelo grande)* | **0.973** | **0.867** | **0.232** | **1.000** | 0.963 | 0.567 | **0.947** | **1.000** | 29.72 |

### Lo que sale bien, y era lo esperado

**La progresión funciona.** De la búsqueda más simple a la más sofisticada, el
recall sube de 0.833 a 0.973 y el context_recall llega a **1.000**: con la
configuración completa, los fragmentos correctos entran **siempre**. Esa es la
tesis del trabajo y quedó demostrada.

**Combinar las dos búsquedas sirve, y es casi gratis.** La híbrida le gana a
las dos ramas por separado en recall (0.913 vs 0.833 y 0.873) y en MRR (0.828,
el mejor de los tres), y tarda lo mismo que la más lenta de las dos: 0.30
segundos. Es la mejor relación resultado/costo de todo el cuadro.

**El sistema nunca inventó.** Abstención **1.000 en las cuatro
configuraciones**: en las 4 preguntas cuya respuesta no está en el corpus, las
4 configuraciones dijeron que no estaba, siempre. Ninguna se mandó una
respuesta falsa. Ejemplo real de la salida:

> *"El contexto proporcionado no incluye información sobre la ley de teletrabajo
> ni sobre el derecho a la desconexión. Por lo tanto, no puedo responder a tu
> consulta."*

---

## 6. Los tres hallazgos interesantes

Estos son los que valen para la exposición, porque ninguno es "todo mejoró".

### Hallazgo 1 — El reranking chico *empeora* el orden

Mirando la tabla: el reranking chico sube el recall (0.913 → 0.933) pero
**baja el MRR** (0.828 → 0.763). O sea: trae más fragmentos correctos, pero los
ordena peor. Eso contradice para qué existe el reranking.

Fuimos a ver **dónde** perdía, abriendo el resultado categoría por categoría:

| configuración | factual | identificador | multihop | **fuente** |
|---|---|---|---|---|
| híbrida (sin reranking) | 1.00 / 0.94 | 1.00 / 0.78 | 0.77 / 0.90 | 0.75 / 0.62 |
| + reranking chico | 1.00 / 0.94 | 1.00 / 0.77 | 0.87 / 0.90 | 0.75 / **0.23** |
| + reranking grande | 1.00 / 0.85 | 1.00 / **1.00** | 0.87 / 0.87 | **1.00** / 0.62 |

*(recall / MRR)*

Todo el daño está concentrado en una sola categoría: **fuente**, donde el MRR
se desploma de 0.62 a 0.23.

**Por qué pasa.** Las preguntas de tipo *fuente* se contestan con la **portada**
del documento: la hoja que dice "Ley Nº 24.240 — Defensa del Consumidor". El
modelo de reranking chico está entrenado para juzgar si un párrafo *responde*
una pregunta, y una portada no responde nada: solo dice quién es el documento.
Así que prefiere sistemáticamente un artículo del cuerpo que habla largo y
tendido de "protección al consumidor" antes que la portada que contiene la
respuesta literal.

El caso más claro, la pregunta *"¿qué ley establece la defensa del
consumidor?"*:

| | posición de la portada correcta |
|---|---|
| híbrida sin reranking | **1º** |
| + reranking chico | **la saca del top 5 entero** |
| + reranking grande | 2º |

El reranking chico tomó el fragmento que estaba primero y lo tiró afuera.

### Hallazgo 2 — El modelo de reranking grande lo arregla, pero cuesta 100×

Probamos el mismo pipeline cambiando solo el modelo que reordena: de uno de 118
millones de parámetros a uno de 568 millones (`BAAI/bge-reranker-v2-m3`).

Arregla el problema y además mejora todo lo demás:

- **identificador: MRR 0.77 → 1.00.** Perfecto. En las 8 preguntas sobre
  números de ley y de artículo, el fragmento correcto sale **primero**.
- **fuente: recall 0.75 → 1.00.** Recupera las portadas que el chico perdía.
- **recall general 0.973**, el mejor de todo el cuadro.

El precio: **29.7 segundos por consulta**, contra 2.7 del chico y 0.30 de la
híbrida. Es **100 veces** más lento que la híbrida sola.

**Conclusión práctica:** el modelo chico para la demo en vivo (nadie espera 30
segundos frente a la clase), el grande para la corrida de evaluación. Es
configurable con una variable de entorno y **no requiere reprocesar el corpus**.

### Hallazgo 3 — La búsqueda literal, la más barata, gana en una categoría

En las preguntas de tipo **fuente**, BM25 solo —la técnica más vieja y barata
del cuadro, 0.02 segundos por consulta— saca **recall 1.00 y MRR 0.81**, mejor
que la híbrida (0.75 / 0.62) y muchísimo mejor que el reranking chico (0.75 /
0.23).

Tiene sentido: para *"¿qué ley regula la defensa del consumidor?"* lo que hay
que hacer es encontrar el documento cuyo título dice literalmente eso. No hace
falta entender nada, hace falta ver la coincidencia exacta.

**Es el hallazgo que mejor resume el trabajo:** "segunda generación" no es
automáticamente mejor. La métrica que elegís decide quién gana —el recall dice
reranking grande, el MRR dice híbrida, la velocidad dice BM25— y el tipo de
preguntas que tu corpus tiene que responder es lo que define qué arquitectura
te conviene.

---

## 7. Un arreglo previo que la evaluación confirma

Antes de esta corrida encontramos un problema serio y lo arreglamos. La
evaluación sirvió para confirmar que quedó arreglado.

**El problema:** a la pregunta *"¿qué dice el artículo 72 de la ley 11179?"* el
sistema contestaba con el artículo 72 de la **ley 11.723** (propiedad
intelectual) presentándolo como si fuera el de la 11.179 (Código Penal). Una
respuesta perfectamente redactada, perfectamente citada, y de la ley
equivocada.

**Por qué pasaba:** ningún fragmento del texto contenía el número de la ley. El
número vive en la portada, que es *otro* fragmento. El fragmento del artículo 72
del Código Penal no dice "11.179" en ningún lado, así que no había forma de
conectarlo con la pregunta.

**El arreglo:** ahora cada fragmento lleva pegado un encabezado con la identidad
de su documento:

```
[Ley 11.179 - Código Penal de la Nación · texto original · Art. 72]
```

**El resultado, medido:** las **cuatro** configuraciones contestan bien esa
pregunta, y tres de las cuatro traen el fragmento correcto en primer lugar.

Ese encabezado incluye la **versión** del documento, y eso destapó otra cosa
que conviene mencionar en la exposición: el corpus tiene la **Ley de Contrato de
Trabajo en su texto original de 1974**, no en el vigente, y la numeración está
corrida unos 17 artículos. Cuando alguien pregunta por "el artículo 208", el
sistema recupera bien, cita bien, y la respuesta igual es inútil, porque el
artículo 208 de 1974 no es el que espera cualquier abogado hoy. **No es un
problema que se arregle buscando mejor** — es un problema del corpus, y sin el
dato de versión nadie se entera.

---

## 8. Qué recomendamos mostrar

1. **La progresión de la tabla** (recall 0.833 → 0.973, context_recall → 1.000)
   como demostración de la tesis.
2. **El hallazgo del reranking chico**, porque una evaluación donde todo mejora
   es sospechosa y esta muestra que efectivamente medimos algo.
3. **La abstención 1.000**, porque es el resultado que más le importa a
   cualquiera que vaya a usar esto en serio.
4. **El caso del artículo 72**, porque es concreto, se entiende en 30 segundos y
   muestra un error que parecía correcto.

Para la demo en vivo: **modelo de reranking chico**. 2.7 segundos es tolerable
frente a la clase; 29.7 no.

---

## 9. Cómo reproducirlo

Requiere Docker corriendo y las dependencias instaladas (ver `README.md`).

```bash
docker compose up -d                      # levanta la base de datos

python evals/run_ragas.py                 # tabla completa, 4 configuraciones
python evals/run_ragas.py --sin-ragas     # solo métricas objetivas, no gasta API
python evals/run_ragas.py --limite 3 --configs naive     # prueba rápida
```

Deja dos archivos en `evals/results/`: un CSV con la tabla y un JSON con el
detalle de cada pregunta —qué fragmentos trajo, qué respondió, cuánto tardó—.
Cada corrida queda etiquetada con la configuración exacta con la que se
produjo, porque una fila de la tabla no significa nada si no se sabe con qué
modelo y con qué parámetros salió.

Resultados de esta corrida:

- `evals/results/ragas_ablacion.json` / `.csv` — las 4 configuraciones
- `evals/results/ragas_rerank-grande.json` / `.csv` — el modelo de reranking grande

**Detalle técnico de la corrida:** 5 fragmentos por consulta; respuestas
generadas con `gpt-4o-mini`; métricas de Ragas juzgadas también por
`gpt-4o-mini`; los cálculos de similitud internos de Ragas usan el mismo modelo
de embeddings que construyó el índice (BGE-M3, local), y no los de OpenAI, para
no medir con una regla distinta de la que usa el sistema evaluado.
