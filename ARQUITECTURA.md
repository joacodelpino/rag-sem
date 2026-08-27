# Cómo funciona — guía para el grupo

Recorrido por el proyecto archivo por archivo, pensado para que cualquiera del
grupo pueda levantarlo y explicarlo en la exposición sin haber escrito el
código.

Estado actual: **las tres configuraciones de recuperación implementadas**
(naive, híbrida con RRF, híbrida + reranking), de punta a punta, sobre el
corpus real de 36 PDFs. Falta la evaluación con Ragas.

---

## Cómo probarlo

```bash
docker compose up -d                                   # 1. Qdrant
.venv/Scripts/python.exe src/ingest.py                 # 2. indexar
.venv/Scripts/python.exe -m streamlit run src/app.py   # 3. la app
```

En Linux/Mac el binario del venv es `.venv/bin/python`. Si el venv está
activado (`source .venv/Scripts/activate`), alcanza con `python` y `streamlit`.

El paso 2 solo hace falta la primera vez, o cuando agregues o cambies
documentos en `data/raw/` — el índice queda persistido en el volumen de Docker.

Abrí http://localhost:8501, escribí una consulta y apretá "Consultar".
Consultas de prueba. Las tres primeras son las que exponen un modo de falla
concreto y están explicadas en `evals/snapshot_retrieval.py`:

- *Que dice el articulo 72 de la ley 11179?* — número de ley sin punto
- *¿Qué dice el artículo 208 de la Ley de Contrato de Trabajo?* — versión
- *¿Qué resolvió la Corte en el caso Siri sobre la acción de amparo?* — fallo
- *¿Qué plazo tiene el consumidor para revocar la aceptación en una venta
  domiciliaria?* — consulta sana, sirve de control

Vas a ver la respuesta arriba y, abajo, un desplegable por cada chunk
recuperado con el nombre del documento y el score de similitud. Eso último es
lo importante para la exposición: te deja mostrar **qué** recuperó el sistema,
no solo qué respondió.

---

## El flujo, en una línea

`ingest.py` llena Qdrant una vez → el usuario escribe en `app.py` →
`retrieval.py` trae los chunks relevantes → `generate.py` se los pasa al LLM →
`app.py` muestra respuesta + fuentes.

```
data/raw/*.txt|*.pdf
        │
        │  ingest.py  (una sola vez, offline)
        │  parsing → chunking → embeddings BGE-M3 + pesos BM25
        ▼
   ┌──────────────────────────────────┐
   │ Qdrant — colección "legal_docs"  │
   │   vector "dense" : BGE-M3        │
   │   vector "bm25"  : sparse (IDF)  │
   └──────────────────────────────────┘
        ▲                    ▲
        │ densa              │ BM25          retrieval.py
        └────────┬───────────┘               (en cada consulta)
                 │
                 ▼
              RRF (fusión por posición)      ─┐
                 │  20 candidatos             │ etapa 1: barata y amplia,
                 ▼                            │ importa el RECALL
        cross-encoder (rerank.py)            ─┤
                 │  top_k final               │ etapa 2: cara y precisa,
                 ▼                            │ importa la PRECISIÓN
   consulta ─────┴──────► generate.py ──► respuesta + fuentes
                            (OpenAI)              │
                                                  ▼
                                               app.py
```

---

## Archivo por archivo

### `docker-compose.yml`

Levanta **solo Qdrant** (la base vectorial), en el puerto 6333. Nada más está
dockerizado: la app corre directo en tu Python, así no hay que reconstruir una
imagen cada vez que se toca una línea de código.

El volumen `qdrant_storage` es lo que hace que el índice sobreviva a un
`docker compose down`.

Bonus para la demo: Qdrant trae un dashboard web en
http://localhost:6333/dashboard donde se puede mostrar la colección y los
vectores en vivo.

> La imagen está fijada en `v1.19.0` para que coincida con el `qdrant-client`
> de `requirements.txt`. **No cambies la versión sin reingestar**: Qdrant no
> puede leer storage escrito por una versión anterior, y el contenedor no
> arranca.

### `.env` / `.env.example`

`.env.example` es la plantilla versionada; `.env` es la copia real con la API
key, y está en `.gitignore` (no se sube al repo). Define la key de OpenAI, el
modelo de generación, la URL de Qdrant, el nombre de la colección y el modelo
de embeddings.

### `requirements.txt`

Las dependencias de Python. `torch` **no** está acá a propósito: se instala
aparte desde el índice CPU-only de PyTorch, porque los wheels por defecto traen
CUDA (~2 GB) que sin GPU no sirve. Ver el README para el comando.

### `data/raw/`

Los 36 PDFs del corpus, ~1.48M caracteres: 12 leyes, 15 resoluciones y
disposiciones de 2022, 5 fallos de la CSJN, 2 contratos modelo, y normativa y
material de la UNLAR. Los 36 tienen texto nativo — no hace falta OCR.

Su identidad (título, número de ley, versión) NO sale del nombre de archivo,
que es inservible: vive en `data/manifest.csv`.

### `data/golden_set.csv`

Vacío por ahora, solo el encabezado. Van las 30 preguntas del set dorado con su
respuesta correcta y qué documentos deberían recuperarse. Es el insumo de
Ragas.

### `src/ingest.py` — parsing + chunking + indexado

El pipeline de carga. Corre una vez, offline; no interviene durante la
consulta.

1. **`load_manifest()` + `validar_cobertura()`** — lee `data/manifest.csv` y
   **falla antes de indexar nada** si algún PDF de `data/raw/` no está ahí o no
   tiene título. Que reviente es a propósito: un chunk sin identidad es el bug
   que este manifiesto vino a arreglar, y enterarse después de media hora de
   embeddings —con la colección ya borrada— es el peor momento posible.
2. **`read_document()`** — extrae texto plano. `.txt` lo lee directo, `.pdf` lo
   pasa por PyMuPDF.
3. **`chunk_document()`** — despacha según `CHUNK_STRATEGY` (ver más abajo).
4. **`encabezado()`** — prepende a cada chunk su línea de identidad.
5. **`build_sparse_vectors()`** — calcula el vector BM25 de cada chunk. Hace
   dos pasadas sobre el corpus: la primera mide la longitud promedio de
   documento, la segunda calcula los pesos, porque BM25 normaliza cada
   documento contra ese promedio.
6. **`build_collection()`** — borra y recrea la colección en Qdrant, con los
   **dos vectores nombrados**: `dense` (coseno) y `bm25` (sparse, con
   `Modifier.IDF`). Borra a propósito: cada ingesta parte de un estado limpio
   y reproducible.
7. **`main()`** — imprime el recuento de chunks **antes** de embeber (son ~30
   min de CPU: si el chunking produjo un número disparatado, conviene saberlo
   antes de pagarlos), embebe con BGE-M3 y sube cada punto con sus dos
   vectores y su payload: `source`, `chunk_index`, `text`, `titulo`,
   `numero_ley`, `version`, `seccion`.

Los dos vectores viven en **el mismo punto de la misma colección**. Las tres
configuraciones leen de ahí; no hay que reingestar para cambiar de
configuración, y eso es lo que hace comparable la ablación.

#### El encabezado de identidad

Cada chunk arranca con una línea que dice de qué documento y de qué versión
sale:

```
[Ley 11.179 - Código Penal de la Nación · texto original · Art. 72]
Art. 72. - Son acciones dependientes de instancia privada...
```

Va **dentro del texto que se embebe y que tokeniza BM25**, no solo en el
payload, porque el problema que resuelve es de recuperación y no de
presentación. Sin esto, el chunk del art. 72 del Código Penal no contiene en
ningún lado la cadena "11.179" —vive en la carátula, que es otro chunk— así que
ninguna de las dos ramas puede conectarlo con la consulta *"artículo 72 de la
ley 11179"*. Medido antes del cambio: el pipeline devolvía el art. 72 de la
**ley 11.723** y el LLM lo citaba como si fuera el de la 11.179.

La **versión** va en el encabezado y no solo en el payload por el hallazgo de
la LCT (ver "Lo que el manifiesto destapó"), y `numero_ley` queda en el payload
para poder filtrar por ley desde Qdrant.

#### Chunking por estructura (`CHUNK_STRATEGY`)

| valor | qué hace |
|---|---|
| `articulo` (default) | corta en límites de artículo; en los fallos, que no tienen artículos, corta por voto (`Dictamen del Procurador`, `Considerando`, `Disidencia`). Si el documento no tiene ninguna de las dos estructuras —los contratos modelo, el folleto de la SRT— cae al chunking fijo. |
| `fijo` | ventanas solapadas de 800 caracteres. Es el chunking del walking skeleton, y queda accesible para poder correr la fila "chunking fijo vs. por artículo" de la ablación. |

Un artículo **es** la unidad de respuesta de una consulta jurídica. Cortando
cada 800 caracteres, la mitad de los chunks empiezan a mitad de un inciso y
terminan a mitad del siguiente artículo, así que el chunk que "contiene la
respuesta" contiene también media respuesta a otra pregunta, y el reranker
tiene que elegir con esa mezcla.

Dos topes, porque los artículos varían muchísimo de largo:

- **`CHUNK_MAX_CHARS = 1200`** — lo que se pasa vuelve a partirse en ventanas
  solapadas, pero cada pedazo conserva el encabezado y suma una marca
  `(cont. 2/8)`.
- **`CHUNK_MIN_CHARS = 250`** — lo que no llega se pega al pedazo anterior.
  Sirve para dos cosas distintas: evitar miles de chunks de una línea
  (*"Art. 5 — Derogado"*), y absorber los cortes falsos, porque el regex
  también matchea una **referencia** a un artículo que arranca renglón. Medido:
  el Código Penal da 306 cortes para 285 artículos reales.

Recuento sobre el corpus de 36 PDFs: **2126 chunks** con `articulo` contra
**2264** con `fijo`, −6%. El 94% de los chunks queda con sección identificada.

> **Trampa del regex de artículos.** El cierre del patrón es `(?!\d)` y no
> `\b`, y eso fue un bug real: en `ARTICULO 1º` el indicador ordinal `º`
> (U+00BA) es **letra** para Unicode, así que entre `1` y `º` no hay borde de
> palabra y el match fallaba. Se perdían justo los artículos 1 a 9 de cada ley
> —los que más se consultan— mientras que `ARTICULO 10` matcheaba sin
> problemas, con lo cual el recuento total apenas cambiaba y el error pasaba
> desapercibido.

### `src/bm25.py` — la mitad léxica

BM25 implementado a mano, en unas treinta líneas. Está separado en su propio
archivo porque lo usan tanto la ingesta (pesos de documento) como la
recuperación (pesos de consulta).

El reparto de tareas con Qdrant es lo que conviene poder explicar:

- **El cliente** calcula el peso **TF** de cada término en cada documento: la
  parte de la fórmula que solo depende del documento. Incluye la saturación
  (`k1`: que un término aparezca 20 veces en vez de 10 no debe duplicar el
  score) y la normalización por longitud (`b`: sin esto los documentos largos
  ganan siempre por acumular menciones).
- **Qdrant** calcula el **IDF**, porque depende de toda la colección — en
  cuántos documentos aparece el término. El cliente no tiene esa estadística
  sin recorrer el índice entero. Se activa con `Modifier.IDF` al crear la
  colección; sin esa línea el vector sparse sería solo TF y BM25 quedaría a
  medias.

`tokenize()` saca acentos y mayúsculas: en las consultas de la demo la gente
casi nunca acentúa, y sin normalizar el match léxico fallaría justo donde
debería brillar. Los ids de término se calculan con CRC32 y no con `hash()` de
Python, porque `hash()` de strings está randomizado por proceso — los ids de la
ingesta no coincidirían con los de la consulta.

### `src/retrieval.py` — la recuperación

El archivo central del trabajo. Todas las funciones comparten la misma firma
`retrieve_x(query, top_k) -> list[Chunk]`, y devuelven `Chunk` con `id`,
`text`, `source` y `score`.

- **`retrieve_naive()`** — configuración 1. Vectorial densa pura: embebe la
  consulta con el mismo modelo de la ingesta y trae los `top_k` más cercanos
  por coseno. Fuerte en paráfrasis, débil en términos exactos y raros
  (números de artículo, siglas, nombres propios se diluyen en el embedding).
- **`retrieve_sparse()`** — BM25 puro. Es el complemento exacto: acierta el
  término literal y no entiende nada de sinónimos. No es una de las tres
  configuraciones de la ablación; está para poder mostrar en la demo qué
  aporta cada rama por separado.
- **`reciprocal_rank_fusion()`** — fusiona varias listas rankeadas usando solo
  las **posiciones**: `score(doc) = Σ 1 / (k + posición)`, con `k=60`.
- **`retrieve_hybrid()`** — configuración 2. Corre las dos ramas, cada una
  trayendo 20 candidatos, y las fusiona. Trae más candidatos que el `top_k`
  final a propósito: RRF solo puede rescatar un documento si al menos una rama
  lo trajo.
- **`retrieve_hybrid_rerank()`** — configuración 3. Le pide a la híbrida 20
  candidatos y se los pasa al cross-encoder, que los reordena y recorta a
  `top_k`. Ver `rerank.py`.

**Por qué RRF y no un promedio de los scores**, que es *la* pregunta a
anticipar en la exposición: los dos scores viven en escalas incomparables. El
coseno de BGE-M3 da valores en un rango angosto y siempre positivo
(típicamente 0.4–0.8 incluso para resultados malos); BM25 no tiene techo y
depende del largo de la consulta y del IDF de la colección. Promediarlos —o
normalizarlos min-max por consulta— deja que la rama con más varianza domine
la fusión, y peor, hace que el resultado cambie según qué otros documentos
entraron en el lote. RRF tira los scores y se queda con el orden, que es lo
único comparable entre las dos ramas. Un documento que salió 2º en las dos
listas le gana a uno que salió 1º en una sola: eso es evidencia de dos señales
independientes, que es exactamente lo que se busca.

La fusión está hecha en Python aunque **Qdrant puede hacerla del lado del
servidor** (`prefetch` + `FusionQuery`). Es a propósito: así se puede mostrar
el código en la exposición e inspeccionar los rankings intermedios de cada
rama.

Detalle de implementación: el modelo de embeddings se carga una sola vez por
proceso (singleton perezoso). Streamlit re-ejecuta el script entero en cada
interacción, y sin eso recargaría BGE-M3 en cada consulta.

### `src/rerank.py` — el cross-encoder

La etapa que define el "RAG de segunda generación". La distinción con el modelo
de embeddings es **el concepto a explicar en la exposición**:

|  | Bi-encoder (BGE-M3) | Cross-encoder (`RERANKER_MODEL`) |
|---|---|---|
| Qué recibe | consulta y documento **por separado** | el par **concatenado**, en una pasada |
| Cómo compara | coseno entre dos vectores | atención cruzada entre los dos textos |
| Se puede precomputar | Sí, los documentos se embeben una vez | **No**, depende de la consulta |
| Escala a | millones de documentos | decenas por consulta |
| Precisión | buena | bastante mejor |

El bi-encoder tiene que comprimir cada documento en un solo vector "por las
dudas", sin saber qué le van a preguntar. El cross-encoder ve las dos cosas
juntas y puede razonar sobre la relación específica: si la consulta pide un
plazo y el documento menciona un plazo pero de otro recurso, lo nota.

De ahí sale la **arquitectura en dos etapas**, que es la idea central: se
recupera barato y amplio para bajar de miles de chunks a 20, y se reordena caro
y preciso solo sobre esos 20. Usar el cross-encoder para todo sería correr el
modelo una vez por chunk del corpus en cada consulta.

`RERANK_CANDIDATES` (en `retrieval.py`) es *el* parámetro del trade-off: más
candidatos dan más chances de rescatar algo que la primera etapa dejó en el
puesto 15, pero el costo crece lineal. El reranker **no puede rescatar lo que
la recuperación no trajo** — por eso la primera etapa se optimiza por recall y
la segunda por precisión.

El score que devuelve es una **probabilidad de relevancia 0–1 que usa todo el
rango**: un chunk irrelevante saca 0.00001, no 0.4. Es una ventaja práctica
sobre el coseno que vale la pena mostrar — permite poner un umbral y decir
"ninguno de estos documentos responde la pregunta", algo que con similitud
coseno es mucho más difícil porque hasta la basura puntúa 0.4.

> Trampa con la que nos chocamos: `CrossEncoder.predict()` aplica la
> `activation_fn` del modelo, y **cada modelo trae una distinta**.
> `bge-reranker-v2-m3` trae `Sigmoid()` y devuelve 0–1; `mmarco-mMiniLMv2`
> trae `Identity()` y devuelve el logit crudo (medido: `+7.18` para un chunk
> relevante, `−7.52` para uno que no). Aplicarle una sigmoide encima al
> primero comprime todo entre 0.5 y 0.73; no aplicársela al segundo deja los
> scores en otra escala. En ninguno de los dos casos cambia el orden —la
> sigmoide es monótona, así que las métricas de ranking no se enteran— pero
> los scores dejan de ser legibles y de ser comparables entre corridas.
> Por eso `rerank.py` pasa `activation_fn=torch.nn.Sigmoid()` explícitamente
> en el constructor, en vez de confiar en el default de cada modelo.

#### Los dos modelos soportados

`RERANKER_MODEL` elige cuál corre, y **cambiarlo no requiere reingestar**: el
reranker actúa sobre candidatos ya recuperados, no sobre el índice.

| Modelo | Params | Para qué |
|---|---|---|
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | ~118M | demo en vivo |
| `BAAI/bge-reranker-v2-m3` | ~568M | corrida de evaluación |

Los dos son XLM-RoBERTa multilingüe con español entre los idiomas de
entrenamiento. Existen los dos a propósito: el trade-off calidad/latencia es
en sí mismo un resultado para la tabla de ablación, no solo un problema de
ingeniería a resolver.

Latencia medida con `RERANK_CANDIDATES=20`, en régimen (o sea descartando la
primera consulta, ver más abajo), vía `python evals/bench_rerank.py` →
`evals/results/rerank_latency.csv`:

| Modelo | s / par | 20 candidatos |
|---|---|---|
| mmarco-mMiniLMv2 (118M) | ~0.10 | **~2.0 s** |
| bge-reranker-v2-m3 (568M) | ~0.94 | **~18.7 s** |

**~9× de diferencia**, con 4.8× de parámetros. Los 18.7s del grande son
inviables en vivo; los 2.0s del chico son perfectamente presentables. Lo que
todavía no sabemos es cuánta calidad cuesta esa diferencia — eso lo tiene que
contestar Ragas sobre el set dorado, no la intuición.

#### Precarga de modelos

`retrieval.warmup()` precalienta los dos modelos y la conexión a Qdrant, y
`app.py` lo llama al arrancar dentro de un `@st.cache_resource`. Sin eso, ese
costo lo paga la **primera consulta**, que en una exposición en vivo es la
peor persona posible a quien cobrárselo.

El detalle que casi se nos pasa: **cargar los pesos no alcanza**. El
benchmark mostró que la primera llamada a `predict()` cuesta bastante más que
las siguientes aunque el modelo ya esté en memoria — 13.6s vs 2.0s en el
chico, 27.8s vs 18.7s en el grande. Es la inicialización de los kernels y el
grafo de cómputo de torch. Por eso `warmup()` no se limita a instanciar los
modelos: les hace correr una inferencia de descarte.

Verificado después del cambio: startup 17.5s, y la primera consulta real
**1.29s**, igual que las siguientes.

### `src/chunk.py` — el tipo compartido

Solo la dataclass `Chunk` que viaja por todo el pipeline. Está en su propio
módulo para que `rerank.py` pueda construir Chunks sin importar `retrieval.py`,
que a su vez lo importa a él.

Ojo con `score`: **significa algo distinto en cada configuración** (coseno,
BM25, RRF, probabilidad del cross-encoder). Se pueden comparar dentro de una
columna de la demo, nunca entre columnas.

### `src/generate.py` — el LLM

El único archivo que le habla a OpenAI. Si algún día hay que cambiar de
proveedor, se cambia solo acá.

Arma el contexto concatenando los chunks recuperados con su fuente entre
corchetes, y el system prompt le pide al modelo que responda **solo** con ese
contexto, que cite la fuente, y que diga explícitamente cuando el contexto no
alcanza. `temperature=0` para que la demo sea reproducible.

Si no hay chunks, devuelve un mensaje fijo sin llamar a la API — no tiene
sentido gastar una llamada pidiéndole al modelo que responda sin contexto.

### `src/app.py` — Streamlit

La interfaz. Muestra la misma consulta resuelta por cada configuración en
columnas paralelas: densa, BM25, híbrida e híbrida+rerank. Cada columna muestra la
respuesta del LLM y, debajo, los chunks recuperados **numerados por posición**
— la posición es lo que cambia entre configuraciones y lo que miden MRR y NDCG.

El diccionario `CONFIGS` de arriba del archivo es lo único que hay que tocar
para sumar una configuración a la demo, justamente porque todas comparten la
misma firma.

El checkbox "Generar respuesta con el LLM" se puede desactivar para comparar
solo la recuperación sin gastar llamadas a la API — útil mientras se afina el
chunking.

### `.streamlit/config.toml`

Una sola cosa relevante: `fileWatcherType = "poll"`. El watcher por defecto de
Streamlit recorre los módulos importados para detectar cambios, y al hacerlo
dispara imports perezosos de `transformers`/`torch` que revientan
(`ModuleNotFoundError: torchvision`). Con `poll` mira el disco en vez de los
módulos; el hot-reload sigue funcionando.

### `evals/` y `notebooks/`

`evals/bench_rerank.py` mide la latencia de cada reranker con
`RERANK_CANDIDATES=20` sobre las mismas consultas y escribe
`evals/results/rerank_latency.csv`.

Ojo con una limitación del benchmark mientras el corpus sea el de ejemplo:
con 8 chunks la recuperación no puede devolver 20 candidatos, así que el
script **completa la lista repitiendo chunks** para poder medir el costo real
de 20 pares. Eso no afecta la latencia (no hay caché por par) pero invalida
cualquier lectura de calidad sobre esas corridas: el script mide tiempo y
nada más. Con el corpus real el relleno deja de aplicarse solo.

`evals/run_ragas.py` es el próximo módulo grande, y ahí entra
`retrieval.config_snapshot()`: devuelve el reranker, los candidatos y el
chunking con los que efectivamente se corrió, para que cada fila de la tabla
quede etiquetada sin anotarlo a mano (que es como termina desincronizado).

---

## Por qué está armado así

Lo único que cambia entre configuraciones es **`retrieval.py`**. El resto del
pipeline —parsing, chunking, prompt, LLM, interfaz— queda idéntico, y las tres
leen de la misma colección.

Eso es justamente lo que hace válida la ablación: cambia una sola pieza por
vez, así que la diferencia en las métricas se le puede atribuir a esa pieza y
no a otra cosa.

---

## Lo que el manifiesto destapó

`data/manifest.csv` (36 filas: archivo, título, número de ley, versión) lo
genera `evals/build_manifest.py`. La columna `version` **no se adivina**: sale
de la URL de origen que infoleg y argentina.gob.ar imprimen en la primera
página (`norma.htm` = texto como se publicó, `texact.htm` = texto actualizado;
`/texto` vs `/actualizacion` en argentina.gob.ar). Tres documentos se clasifican
por su propia parte dispositiva ("Apruébase el texto ordenado..."). Los fallos,
los contratos modelo y el folleto de la SRT quedan en `no_aplica`, porque ahí
la distinción no existe.

Hacer el manifiesto sacó a la luz dos problemas de corpus que ninguna mejora
de recuperación podía arreglar:

**1. La LCT está indexada en su texto original de 1974, no en el texto ordenado
vigente.** La numeración está corrida. El art. 208 del corpus habla de menores
que trabajan en horas de la mañana y de la tarde; el art. 208 que espera
cualquier abogado —enfermedades inculpables y plazos de licencia paga— está en
el **art. 225** de ese PDF. Y el contenido del art. 57 vigente (presunción en
contra del empleador por su silencio) directamente **no existe** en ese texto.

Esto no es una alucinación ni una falla de recuperación: el chunk es correcto,
la cita es literal y exacta, y la respuesta es inútil. Es el mejor argumento
del trabajo a favor de que el RAG de segunda generación necesita **metadatos**,
y no solo mejores recuperadores. La mitigación acá es doble: la versión viaja
en el encabezado de cada chunk, y el system prompt le pide al LLM que advierta
cuando el contexto es un texto original y la consulta parece referirse al
régimen vigente.

Esa segunda mitad es un interruptor, no una constante: `ADVERTIR_VERSION`
(env), `generate.system_prompt(advertir_version)`, el checkbox de la app y
`--sin-advertencia-version` en `snapshot_retrieval.py`. Existe porque la
demostración necesita **las dos** ramas: con la advertencia prendida desde el
arranque no hay antes que mostrar. Y apagarla deja intactos los otros dos
cambios —el encabezado de identidad y la cita por etiqueta—, con lo cual lo que
se ve es el efecto del prompt **aislado** del efecto de los metadatos. Los dos
prompts son idénticos salvo esa frase; no hay ninguna otra diferencia entre las
ramas.

**2. El nombre de archivo no identifica nada.** Cinco documentos se llaman
`0N_argentinagobar.pdf` y `08_ley-15.pdf` no es la ley 15 sino el Decreto-Ley
15.348/46. Por eso los títulos del manifiesto están curados a mano leyendo la
primera página de cada PDF, y por eso la app y las citas del LLM muestran
`Chunk.etiqueta()` en vez de `source`.

## Pendientes conocidos

- **Ragas**: no implementado. Es el módulo que falta para la tabla de
  ablación, y el que va a poner número a si el chunking por artículo y el
  encabezado de identidad efectivamente mejoran el recall, en vez de saberlo
  solo por las consultas de prueba.
- **Cobertura del corpus**: hay 12 leyes y 15 resoluciones de 2022 muy
  parecidas entre sí (AABE, dumping). Para el set dorado conviene chequear que
  las 30 preguntas no caigan todas sobre los mismos tres documentos.
- **La versión de las normas**: el manifiesto la registra pero no la corrige.
  Reemplazar el PDF de la LCT por su texto ordenado sería la solución de fondo;
  como el caso es didáctico, conviene decidir si conviene más **mostrarlo** en
  la exposición que arreglarlo.
