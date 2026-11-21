# Diseño experimental

Documento de trabajo para los experimentos del artículo. Después se condensa en la
sección de Metodología y en el párrafo 4 de la Introducción.

Estado: **borrador v1 (21 nov. 2026)**. Todo lo marcado con ⚠ está por verificar o decidir.

---

## 1. Pregunta de investigación

Objetivo general: medir cómo la cuantización de pesos afecta la reparación automática de
vulnerabilidades con un CLM, en tres dimensiones:

| ID | Pregunta | Métrica principal |
|---|---|---|
| RQ1 | ¿La cuantización reduce la capacidad del modelo para **eliminar** la vulnerabilidad? | Tasa de eliminación (según analizador estático) |
| RQ2 | ¿La cuantización hace que el modelo **introduzca** vulnerabilidades nuevas? | Tasa de parches con hallazgos nuevos |
| RQ3 | ¿Cuánto **ahorra** la cuantización (memoria, tiempo) y a qué costo en calidad? | VRAM pico, latencia por parche, tokens/s |
| RQ4 (secundaria) | ¿Los modelos cuantizados reparan **los mismos** casos que el original? | Coincidencia (Jaccard) de los casos resueltos |

RQ4 replica, en seguridad, la observación de Vallecillos-Ruiz et al. de que los modelos
cuantizados resuelven una cantidad similar de errores, pero no los mismos.

## 2. Qué se sabe del artículo base (PatchLM)

Revisado del repositorio oficial https://github.com/SmartSecLab/PatchLM (`config.yaml`,
`source/*.py`) el 21 nov. 2026:

- Modelo base: `codellama/CodeLlama-7b-Instruct-hf` (alternativa: `Salesforce/codet5-base`).
- Ajuste fino con LoRA: `r=4`, `lora_alpha=64`, `lora_dropout=0.1`, módulos `q_proj` y `v_proj`.
- Métricas: ROUGE y CodeBLEU (pesos 0,25 cada uno). **No** hay exact match ni ningún
  análisis de seguridad de los parches generados.
- Generación: `max_new_tokens=512`, muestreo con `temperature=0.7`, `top_p=0.9`.
- Datasets: FixMe (CVE, Zenodo 10.5281/zenodo.10955342) o `ASSERT-KTH/repairllama-datasets`.
- **Los pesos ajustados de PatchLM no están publicados** (solo rutas locales `models/...`).

Observaciones para el análisis crítico (⚠ confirmar con el PDF del artículo):
- En el `config.yaml` publicado, el dataset activo es `repairllama`, cuyos ejemplos parecen
  errores funcionales en Java, no vulnerabilidades con CVE. Hay que confirmar con el PDF
  qué datos se usaron realmente en los resultados reportados.
- En el `config.yaml` publicado aparece `use_4bit_quantization: False`, pero existe una ruta a
  `models/CodeLLama-7b-quantized-4bit`. La afirmación de la Introducción ("cuantizó CodeLlama a
  4 bits por limitaciones de recursos") debe respaldarse con una cita del PDF, no del repo.
- Ninguna de sus métricas comprueba si el parche elimina la vulnerabilidad: CodeBLEU mide
  parecido textual con la corrección de referencia. Esto refuerza nuestra brecha.

**Consecuencia para el diseño:** como no hay pesos de PatchLM, se cuantiza su **modelo base**,
CodeLlama-7B-Instruct, y se evalúa con el mismo prompt de PatchLM. Así se mide el efecto de
la cuantización sobre el mismo modelo y el mismo formato de tarea, sin el ajuste fino.
(Extensión opcional si queda tiempo: reproducir un ajuste fino LoRA pequeño en Colab.)

## 3. Modelos y niveles de cuantización

Método: `bitsandbytes` a través de `transformers` en todas las configuraciones, para que la
única variable sea el número de bits.

| Config | Modelo | Precisión | Dónde |
|---|---|---|---|
| L-16 | CodeLlama-7B-Instruct | FP16 (línea base) | Colab T4 |
| L-8 | CodeLlama-7B-Instruct | INT8 (`load_in_8bit`) | Colab T4 |
| L-4 | CodeLlama-7B-Instruct | NF4 (`load_in_4bit`, doble cuantización) | Colab T4 |
| L-4-local | CodeLlama-7B-Instruct | NF4 | Laptop (⚠ probar si cabe en 4 GB) |
| Q-16 | Qwen2.5-Coder-1.5B-Instruct | FP16 (línea base) | Laptop |
| Q-8 | Qwen2.5-Coder-1.5B-Instruct | INT8 | Laptop |
| Q-4 | Qwen2.5-Coder-1.5B-Instruct | NF4 | Laptop |

- Los 3 y 2 bits (que Giagnorio et al. reportan con caídas fuertes) quedan como **extensión
  opcional** con HQQ (`transformers.HqqConfig(nbits=3|2)`), porque `bitsandbytes` solo ofrece 8 y 4.
- Hardware: Colab T4 (16 GB VRAM). Laptop: RTX 3050 Laptop 4 GB VRAM, 8 GB RAM, i5-12500H.

## 4. Datos

Fuente: **FixMe** (Bhandari y Gavric, Zenodo `FixMe-v1.db`, 814,7 MB, CC-BY-4.0). Es el
dataset de vulnerabilidades del mismo grupo de PatchLM, extraído de commits asociados a CVE.

Filtrado previsto originalmente (**superado por 4.1 y 4.2**, se deja como registro):
1. Cambios a nivel de **función/método** en **Python**, con su versión vulnerable y corregida.
2. Longitud de la función vulnerable ≤ 1500 caracteres (para que quepa con el prompt y
   generar sea rápido).
3. **Filtro de oráculo:** se conservan solo los casos donde Bandit reporta al menos un hallazgo
   en la versión vulnerable que desaparece en la versión corregida de referencia. Así se
   puede medir objetivamente si el modelo eliminó "la" vulnerabilidad.
4. Muestra aleatoria de hasta **N = 150** casos con semilla 42 (⚠ ajustar según cuántos
   sobrevivan al filtro).

¿Por qué Python y no Java como PatchLM? Para Python hay un analizador de seguridad maduro,
gratuito y multiplataforma (Bandit), que corre sobre funciones sueltas. Para Java, la
validación seria requiere Vul4J (tests de prueba de vulnerabilidad en Docker, Java 8/11 y
Maven), que es inviable con 8 GB de RAM en Windows. Esta decisión va en "amenazas a la validez".

### 4.1 Hallazgos al explorar los datos (21 nov. 2026)

`FixMe-v1.db` descargado y verificado (MD5 `fd5de8fd5f9f85f380787a641b72182e`). Tablas:
`cve` (18 063), `repository` (2 133), `patch_collection` (19 109), `hunk_collection` (44 059).
**No hay tabla de funciones:** los pares antes/después están por *hunk* (fragmento de diff,
columnas `code_before` / `code_after`), no por función completa.

Hunks por lenguaje (top): PHP 15 214, desconocido 7 527, JavaScript 4 651, JSON 3 936,
C 2 972, **Python 1 627**, Java 1 439, Go 1 072.

Embudo en Python (Bandit 1.9.4):

| Paso | Casos |
|---|---|
| Hunks de Python | 1 627 |
| Antes y después sintácticamente válidos y distintos | 588 |
| Bandit reporta algo en la versión vulnerable | 35 |
| El hallazgo desaparece en la corrección de referencia | **11** |

**Conclusión:** con FixMe y Bandit solo quedan 11 casos con oráculo de seguridad. Son
demasiado pocos para RQ1. La mayoría de los CVE son errores lógicos (validaciones,
permisos) que un analizador de patrones no ve.

Alternativa revisada: **SecurityEval** (Siddiq y Santos, MSR4P&S 2022;
`s2e-lab/SecurityEval`): 121 funciones Python inseguras por CWE (campo `Insecure_code`).
Bandit reporta hallazgos en **49 de 121**, sin errores de análisis.

### 4.2 Decisión: dos experimentos (aprobada 21 nov. 2026)

Resultado de `01_preparar_dataset.py` (`datos/embudo.json`): E1 = 150 casos (de 583 hunks
válidos con ≤ 1500 caracteres; en 5 de ellos Bandit reporta algo en la versión vulnerable);
E2 = 49 casos (de 121). SecurityEval no declara licencia en GitHub, así que `E2.jsonl` no se
sube al repo: el script lo descarga y lo genera.

Dos experimentos complementarios con los mismos modelos y configuraciones:

| | E1: CVE reales | E2: oráculo de seguridad |
|---|---|---|
| Datos | FixMe, hunks Python válidos (muestra de 150 de 588) | SecurityEval, los 49 casos con hallazgo de Bandit |
| Tarea | Reparar el fragmento vulnerable | Reparar la función insegura |
| RQ1 eliminación | Parecido a la corrección real (CodeBLEU, ROUGE-L), como PatchLM | Hallazgo de Bandit eliminado |
| RQ2 introducción | Hallazgos de Bandit nuevos | Hallazgos de Bandit nuevos |
| RQ3 costo | VRAM, latencia | VRAM, latencia |

E1 conecta con PatchLM (mismos datos de CVE y métricas). E2 aporta lo que PatchLM no mide:
si la vulnerabilidad realmente desaparece. Ampliación posible de E2: añadir Semgrep en
Colab (Linux) para cubrir más de los 121 casos.

## 5. Procedimiento

Para cada configuración de la tabla 3 y cada caso del dataset:
1. Construir el prompt de evaluación de PatchLM (`generate_eval_prompt_codellama`).
2. Generar con decodificación **voraz** (`do_sample=False`), `max_new_tokens=512`.
   Se usa voraz y no el muestreo de PatchLM para que la diferencia entre configuraciones
   no se deba al azar (cada configuración genera una sola vez y el resultado es reproducible).
3. Registrar: texto generado, tiempo de generación, tokens generados, VRAM pico
   (`torch.cuda.max_memory_allocated`).
4. Extraer el código de la respuesta (bloque de código o texto tras `### Response:`).

## 6. Métricas y evaluación

Por parche generado:

| Métrica | Cómo | Pregunta |
|---|---|---|
| Sintácticamente válido | `ast.parse` tras `textwrap.dedent` | todas |
| Vulnerabilidad eliminada | Los hallazgos de Bandit (por `test_id`) que tenía la versión vulnerable ya no aparecen | RQ1 |
| Vulnerabilidad introducida | Aparece algún `test_id` de Bandit que no estaba en la versión vulnerable | RQ2 |
| Reparación "segura" | Válido + eliminada + sin introducidas + no idéntico al vulnerable | RQ1 |
| CodeBLEU, ROUGE-L | Contra la corrección de referencia (comparables con PatchLM) | RQ1 |
| VRAM pico, s/parche, tokens/s | Registrados durante la generación | RQ3 |

En E1 casi ningún caso tiene oráculo de eliminación (5 de 150), así que ahí RQ1 se mide con
CodeBLEU/ROUGE-L y RQ2 con "introducida". En E2 la métrica principal es "reparación segura".

Comparación estadística: cada configuración cuantizada contra su línea base FP16 con la
**prueba de McNemar** (datos pareados y binarios: mismos casos, reparado sí/no), α = 0,05.
RQ4: índice de Jaccard entre los conjuntos de casos reparados.

## 7. Amenazas a la validez (borrador)

- **Oráculo aproximado:** Bandit detecta patrones; un parche puede "engañarlo" (p. ej., borrar
  el código) o no detectar vulnerabilidades lógicas. Mitigación: exigir validez sintáctica,
  descartar parches idénticos o vacíos y revisar manualmente una muestra (⚠ ~20 casos).
- **Contaminación de datos:** los commits de FixMe son públicos y pueden estar en el
  entrenamiento de CodeLlama y Qwen. Afecta por igual a todas las configuraciones, así que la
  comparación entre bits sigue siendo válida, aunque las cifras absolutas pueden inflarse.
- **Sin ajuste fino:** se evalúa el modelo base de PatchLM, no PatchLM.
- **Un solo lenguaje (Python)** y un solo método de cuantización (bitsandbytes).
- **Hardware:** latencia medida en T4 y RTX 3050; no generaliza a otras GPU.

## 8. Plan de ejecución

| Paso | Dónde | Estado |
|---|---|---|
| Instalar entorno (`requirements.txt`) | Laptop | en curso |
| Descargar FixMe y preparar dataset | Laptop | **hecho** |
| Probar evaluación y análisis con parches simulados | Laptop | **hecho** |
| Piloto: Q-16/Q-8/Q-4 con pocos casos | Laptop | **hecho** (ver 9) |
| Corrida completa Qwen (Q-16/8/4) | Laptop | en curso (`scripts/correr_laptop.ps1`) |
| Prueba L-4-local (¿cabe en 4 GB?) | Laptop | pendiente |
| Corrida CodeLlama L-16/8/4 | Colab | pendiente |
| Evaluación (Bandit, CodeBLEU) y análisis | Laptop | pendiente |
| Revisión manual de ~20 parches | Laptop | pendiente |

## 9. Piloto en la laptop (21 nov. 2026)

Qwen2.5-Coder-1.5B-Instruct, 5 casos de E2, RTX 3050 Laptop 4 GB, PyTorch 2.6.0+cu124,
transformers 5.20.2, bitsandbytes 0.50.2.

| Config | Memoria del modelo | VRAM pico | s/parche (media) | tokens/s |
|---|---|---|---|---|
| Q-16 | 2 944 MiB | 2 970 MiB | 10,6 | 9,8 |
| Q-8 | 1 702 MiB | 1 733 MiB | 43,3 | 2,5 |
| Q-4 | 1 100 MiB | 1 150 MiB | 9,4 | 6,8 |

Primeras observaciones (5 casos, **no concluyentes**):
- 4 bits reduce la memoria del modelo un 63 % frente a FP16; 8 bits, un 42 %.
- INT8 de bitsandbytes es unas 4 veces más lento que FP16 en esta GPU. Esto coincide con lo
  que reportan Vallecillos-Ruiz et al.: menos memoria no implica menos tiempo de inferencia.
- Q-4 produjo 2 de 5 respuestas que no conservaban la función original (p. ej., solo
  `obj['key']`); el control `conserva_funciones` las descarta como reparación.

Ajustes hechos tras el piloto:
- Extracción de código robusta: prueba cada tramo entre cercas ``` y quita prosa inicial.
- La generación se corta al cerrar un bloque de código válido (antes llegaba a 512 tokens
  con explicaciones). Con decodificación voraz esto no cambia el código generado.
- Control `conserva_funciones`: el parche debe definir las mismas funciones/clases.

Estimación para la corrida completa en la laptop (199 casos × 3 configs): ~3–4 h,
la mayor parte en Q-8.

## 10. Resultados en la laptop: Qwen2.5-Coder-1.5B (21 nov. 2026)

Corrida completa: E1 = 150 casos, E2 = 49 casos, configuraciones Q-16, Q-8 y Q-4.
Archivos: `resultados/resumen.csv`, `resultados/comparaciones.csv`, `resultados/costo_*.json`.

### Calidad (frente a la línea base FP16 del mismo modelo)

| Métrica | Q-16 | Q-8 | Q-4 | Prueba (Q-16 vs Q-4) |
|---|---|---|---|---|
| E1 CodeBLEU (media) | 0,568 | 0,568 | **0,509** | Wilcoxon p = 0,020 |
| E1 ROUGE-L (media) | 0,590 | 0,582 | **0,481** | Wilcoxon p = 0,002 |
| E1 cambio válido | 74,0 % | 70,0 % | 73,3 % | McNemar p = 1,00 |
| E1 introduce hallazgos | 0,7 % | 0,7 % | 0,7 % | McNemar p = 1,00 |
| E2 reparación segura | 12,2 % | 12,2 % | 10,2 % | McNemar p = 1,00 |
| E2 reparación segura (sin avisos de import) | 13,0 % | 13,0 % | 10,9 % | McNemar p = 1,00 |
| E2 cambio válido | 83,7 % | 83,7 % | 71,4 % | McNemar p = 0,070 |
| E2 introduce hallazgos | 6,1 % | 6,1 % | 4,1 % | McNemar p = 1,00 |

Q-16 vs Q-8: ninguna diferencia significativa (CodeBLEU p = 0,38; ROUGE-L p = 0,23; en E2
las respuestas son las mismas, Jaccard = 1,0).

### Costo (medición controlada: 10 prompts × 128 tokens × 3 repeticiones, RTX 3050 4 GB)

| Config | Memoria del modelo | VRAM pico | ms/token (media ± desv.) | tokens/s |
|---|---|---|---|---|
| Q-16 | 2 944 MiB | 2 968 MiB | 43,5 ± 0,7 | 23,0 |
| Q-8 | 1 702 MiB (−42 %) | 1 733 MiB | 188,7 ± 1,3 (×4,3) | 5,3 |
| Q-4 | 1 100 MiB (−63 %) | 1 150 MiB | 61,1 ± 0,6 (×1,4) | 16,4 |

### Lectura preliminar (para la discusión; falta contrastar con CodeLlama-7B)

1. **8 bits es prácticamente sin pérdida** en calidad, pero en esta GPU es 4,3 veces más lento.
   Menos memoria no implica menos tiempo (coincide con Vallecillos-Ruiz et al.).
2. **4 bits baja de forma significativa el parecido con la corrección real** (CodeBLEU −10 %,
   ROUGE-L −19 %) y tiende a producir más respuestas inválidas en E2 (p = 0,07, no significativo).
3. **No se observa que la cuantización introduzca más vulnerabilidades** (tasas ≤ 6 %, iguales o
   menores que en FP16). Con tan pocos eventos, la potencia estadística es baja.
4. **Efecto suelo en E2:** el modelo de 1,5B repara de forma segura solo ~12 % de los casos aun en
   FP16. En 34 de 49 casos (FP16) mantiene el mismo hallazgo de Bandit; excluir los avisos por
   solo importar un módulo casi no cambia la tasa (13,0 %). Con tasas tan bajas, McNemar no puede
   detectar diferencias: hace falta el modelo de 7B (Colab) para que RQ1 en E2 sea informativa.
5. Los tiempos de la corrida principal (`segundos` en `resumen.csv`) dependen de la longitud de
   cada respuesta y de la carga de la laptop; para RQ3 se usan solo los de la medición controlada.
