# TinyStressNet: un modelo listo para cuantización para la detección de estrés académico independiente del sujeto. (TinyStressNet: A Quantization-Ready Model for Subject-Independent Academic Stress Sensing)

**[📗 Read this in English / Leer en inglés](README.md)**

---

## 📚 Cita

Si usa o referencia este repositorio, por favor cite el artículo publicado:

> **Alcaraz-Valencia, P. A., Santana-Mancilla, P. C., Gaytán-Lugo, L. S., & Anido-Rifón, L. E.**
> *TinyStressNet: A Quantization-Ready Model for Subject-Independent Academic Stress Sensing.*
> **Applied Sciences**, 2026. <https://doi.org/10.3390/app16189028>

---

Este repositorio reproduce todas las tablas, figuras y estadísticos del artículo a partir del conjunto de datos publicado. TinyStressNet es un clasificador de **3,139 parámetros y 2,800 MACs** que estima tres niveles de estrés académico (bajo, medio, alto) a partir de tres señales fisiológicas —respuesta galvánica de la piel, frecuencia cardiaca y temperatura cutánea— con una ruta de exportación a **int8** validada.

> Desarrollado con **fines educativos y de investigación**. Es una prueba de concepto no clínica: tres estadísticos resumen por intervalo no permiten caracterizar un estado fisiológico en términos clínicos.

---

## 🧩 Descripción general

### Datos (`stress_data.csv`)

1,000 observaciones a nivel de intervalo de cinco estudiantes de ingeniería, recolectadas entre febrero y mayo de 2024. Una fila por medición: `user_id`, `timestamp`, `gsr`, `heart_rate`, `skin_temperature`, `stress_label`. Las etiquetas provienen de autorreportes STAI-6 y son independientes de las lecturas de los sensores. `user_id` y `timestamp` se usan solo para agrupar y para la partición cronológica, nunca como variables predictoras.

### Evaluación (`evaluate.py`)

- **LOSO anidado** (*leave-one-subject-out*), el protocolo principal. Los folds internos eligen el presupuesto de épocas, el participante retenido nunca interviene en la selección del modelo y la estandarización se ajusta solo dentro de los folds de entrenamiento.
- **Partición cronológica por usuario**, un objetivo de generalización distinto (60/20/20 en el tiempo).

### Augmentación como invariancia aproximada (`augmentations.py`)

Cuatro condiciones: sin augmentación, ruido gaussiano, una familia de reescalado radial y una familia afín acotada (BA) cuyos parámetros corresponden a deriva de ganancia, desplazamientos de calibración por sensor y un cambio correlacionado de línea base a lo largo de un eje orientado a la activación.

### Análisis

- `robustness.py` — sonda acotada de escala y desplazamiento (§3.10 del artículo)
- `stats_tests.py` — permutación exacta por cambio de signo, prueba de signos y Wilcoxon (§3.11)
- `quantize.py` — cuantización estática post-entrenamiento a int8 y exportación a TorchScript (§4.6)

---

## ⚙️ Requisitos

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9+, PyTorch 2.1+, scikit-learn, pandas, numpy, scipy, matplotlib. `statsmodels` es opcional y solo aporta la línea base de logit ordenado. No se requiere GPU: el modelo es tan pequeño que la CPU resulta más rápida en la práctica.

---

## ▶️ Uso

**Reproducir las tablas y figuras del artículo** (semilla fija, ~10 minutos en un núcleo):

```bash
python run.py --csv stress_data.csv --outdir results
```

Genera las Tablas 1–8, las matrices de confusión y `results.json`.

**Reproducir el estudio de diez semillas** que sustenta las conclusiones sobre augmentación (~70 minutos):

```bash
python multiseed.py --seeds 0 1 2 3 4 5 6 7 8 9 --probe-draws 30 --outdir results_multiseed
```

Los resultados se guardan en caché después de cada semilla, de modo que una ejecución interrumpida continúa donde se detuvo. La salida queda en `multiseed_report.txt` (tablas legibles), `multiseed_results.json` y `multiseed_runs.csv`.

---

## 🔁 Notas de reproducibilidad

**Semillas.** `--seed-scheme legacy` reinicia cada ajuste con la misma semilla y reproduce los valores de semilla fija de las Tablas 4, 5 y 7. `--seed-scheme nested` (predeterminado en `multiseed.py`) deriva una semilla distinta por ajuste y es el esquema del análisis de sensibilidad. En ambos esquemas **las particiones no se mueven**: los folds LOSO los determina el participante y la partición cronológica la fecha, de modo que la dispersión reportada refleja únicamente variabilidad de entrenamiento.

**Qué encontró el estudio de semillas.** A lo largo de diez semillas, el weighted F1 independiente del sujeto es 0.900 ± 0.006 (BA), 0.899 ± 0.004 (radial), 0.898 ± 0.005 (sin augmentación) y 0.894 ± 0.006 (ruido gaussiano). Las diferencias entre las tres primeras son menores que la variación entre semillas. La familia BA es la condición más consistente —la mejor en cinco de diez semillas y nunca la peor— pero esto es una observación de estabilidad, no una ganancia de desempeño medida.

**El piso del p-valor.** Con cinco participantes existen 2⁵ = 32 asignaciones de signo, así que el menor p-valor bilateral alcanzable es 0.0625, igual para la prueba de permutación, la de signos y Wilcoxon. Ninguna comparación de este estudio puede alcanzar significancia convencional, con cualquier tamaño de efecto y cualquier número de semillas. Todos los ordenamientos de modelos aquí son descriptivos.

**Backends de cuantización.** `quantize.py` resuelve automáticamente un backend disponible (`x86`, `fbgemm`, `qnnpack` u `onednn`). El tamaño de los artefactos varía ligeramente entre compilaciones de PyTorch; la cifra a nivel de parámetros es estable en 12.3 KB en float32 frente a 3.1 KB en int8.

---

## ⚖️ Licencia

Publicado bajo la **Licencia MIT**, que permite uso, modificación y redistribución libres con fines educativos y de investigación. Consulte [LICENSE](LICENSE).
