# FAMACHA-Net — metodología

Red neuronal para estimar el grado FAMACHA© (1–5) y la decisión **tratar / no tratar** a partir de una foto de la conjuntiva de cabras y borregos. Opcionalmente también estima el **hematocrito (PCV)** y usa datos clínicos del *Five Point Check*.

```
imagen 0-255 ──► Balance de blancos ──┬─► EfficientNetV2-B0 (ImageNet) ─► GAP ────────┐
                 (en el grafo)        └─► Estadísticas de color de la conjuntiva ─────┤
datos clínicos (CC, edema, diarrea, edad…) ───────────────────────────────────────────┤
                                                                                       ▼
                                                      Dense ─► CORAL ordinal (FAMACHA 1-5)
                                                           └► PCV % (si hay datos)
```

---

## 1. Sobre la meta del 98 %

**Un 98 % en los 5 grados no es una meta alcanzable de forma honesta**, y si tu modelo lo reporta, lo más probable es que haya fuga de datos. Las razones:

1. **La etiqueta es subjetiva.** El grado FAMACHA lo asigna una persona comparando colores con una tarjeta. Dos evaluadores capacitados suelen diferir en ±1 grado, sobre todo entre 2–3 y 3–4. Un modelo no puede superar de forma confiable el acuerdo entre los humanos que lo etiquetaron. Por eso el CSV trae la columna `famacha_2`: si dos personas califican la misma foto, el script calcula ese techo automáticamente.
2. **FAMACHA no es lo mismo que anemia.** El estándar de oro es el **PCV** (microhematocrito). Contra el PCV, el propio método FAMACHA tiene sensibilidad y especificidad lejos del 100 %, y con mucha variación entre especies y razas.
3. **Hay fuga de datos si el split no es por animal.** Si el mismo animal (el otro ojo o la foto de otra semana) aparece en train y en test, la accuracy se infla. Este pipeline divide **por animal** y lo verifica con `assert`.

**Metas realistas y defendibles para reportar:**

| Métrica | Qué significa | Meta razonable |
|---|---|---|
| Accuracy ±1 grado | Error de máximo un grado | ≥ 95 % |
| Kappa ponderada cuadrática | Acuerdo ordinal (la métrica estándar para escalas como FAMACHA) | ≥ 0.80 |
| Accuracy “tratar” (≥ 4 vs ≤ 3) | La decisión que de verdad importa en campo | 90–95 % |
| Sensibilidad “tratar” | Anémicos detectados | ≥ 95 % (el umbral se ajusta para esto) |
| Accuracy de 5 grados | Coincidencia exacta | Cerca del acuerdo inter-evaluador |

Si después de entrenar ves **98–100 % en 5 grados**, sospecha primero de: duplicados, fotos del mismo animal en ambos splits, o que el modelo aprendió algo ajeno a la conjuntiva (el pelaje, la granja, el teléfono). Para descartarlo usa `scripts/gradcam.py`.

---

## 2. Protocolo de datos (lo que más influye en el resultado)

La arquitectura importa mucho menos que los datos. Lo mínimo recomendado:

- **≥ 150–200 animales** distintos, **≥ 100 fotos por grado** (los grados 1 y 5 suelen escasear: búscalos).
- **Ambos ojos**, varias fechas, **varias granjas, varios teléfonos y varias condiciones de luz**. Si todo sale de un solo rancho y un solo celular, el modelo no generalizará.
- **Fotos originales del teléfono, no capturas de pantalla ni fotos reenviadas por WhatsApp.** WhatsApp recomprime y altera el color, que es precisamente la señal.
- Luz natural indirecta, sin flash, con el párpado inferior bien evertido y enfocado.
- **Sin fotos con logos ni marcas de agua** (por ejemplo, fotos promocionales de redes sociales). Además de los derechos de uso, el modelo puede aprender el logo.
- **PCV por microhematocrito** al menos en una parte de los animales, tomado el mismo día que la foto. Esto convierte el proyecto de “imitar al evaluador” en “detectar anemia real”, y es lo que un revisor o un jurado van a preguntar primero.
- **Dos evaluadores** independientes (`famacha` y `famacha_2`).
- Recomendado: **bounding box** de la conjuntiva (`x1,y1,x2,y2`) anotado en CVAT, Label Studio o Roboflow. Sin bbox se usa un recorte central.

### Formato del CSV (`data/labels.csv`)

Ver `data/labels_template.csv`. Solo `image_path`, `animal_id` y `famacha` son obligatorias.

| Columna | Descripción |
|---|---|
| `image_path` | Ruta relativa a la carpeta del CSV |
| `animal_id` | Identificador único del animal (**crítico** para el split) |
| `famacha`, `famacha_2` | Grado 1–5 del evaluador 1 y del evaluador 2 (opcional) |
| `pcv` | Hematocrito % (opcional; activa la cabeza de PCV y la evaluación contra el estándar de oro) |
| `x1,y1,x2,y2` | Caja de la conjuntiva en píxeles (opcional) |
| `bcs` | Condición corporal 1–5 |
| `dag_score` | Suciedad perianal / diarrea 0–5 |
| `bottle_jaw`, `nasal_discharge`, `diarrhea` | 0/1 |
| `age_months`, `days_since_deworm`, `is_goat`, `rainy_season` | Contexto |

### Parámetros adicionales a FAMACHA que incluye el modelo

- **Five Point Check** (Bath & van Wyk): ojo (FAMACHA), condición corporal, edema submandibular (*bottle jaw*), cola o diarrea, y descarga nasal. Juntos mejoran la decisión de tratar, sobre todo en el grado 3, el más ambiguo.
- **Estadísticas de color** calculadas dentro de la red: cromaticidad r/g/b, índice de enrojecimiento (R−G)/(R+G), saturación y luminancia, excluyendo la pupila y los brillos.
- **Balance de blancos** dentro del grafo, usando los brillos especulares de la mucosa húmeda como referencia del iluminante.
- **PCV estimado**, como salida auxiliar.

> Solo actívalos (`metadata.enabled: true`) si tu app los va a pedir. El modelo debe recibir en producción exactamente las mismas entradas que en el entrenamiento.

---

## 3. Uso

```bash
pip install -r requirements.txt

# 0) (opcional) prueba que todo corra con datos sintéticos
python scripts/make_synthetic.py --out data/synth
#    -> en config.yaml apunta labels_csv a data/synth/labels.csv

# 1) validar, recortar y hacer control de calidad
python scripts/prepare.py --config config.yaml

# 2) entrenar: CV de 5 folds por animal + ensamble + evaluación en test
python scripts/train.py --config config.yaml      # --drop-flagged para excluir fotos malas

# 3) matriz de confusión y Grad-CAM
python scripts/evaluate.py --run runs/exp1
python scripts/evaluate.py --run runs/exp1 --csv otra_granja/processed/processed.csv   # validación externa
python scripts/gradcam.py  --run runs/exp1

# 4) predecir una foto
python scripts/predict.py --run runs/exp1 --image foto.jpg --meta bcs=2 bottle_jaw=1

# 5) exportar para la app
python scripts/export_tflite.py --run runs/exp1            # ensamble
python scripts/export_tflite.py --run runs/exp1 --single   # un solo modelo (~12 MB)

# 6) exportar para la página web "Análisis Rápido" (pip install tensorflowjs)
python scripts/export_tfjs.py --run runs/exp1 --single
#    -> runs/exp1/tfjs/: en la página, pestaña Modelo, selecciona TODOS esos archivos
```

La página web también exporta el dataset que vayas registrando (`famacha_dataset.zip`) con `labels.csv` ya en el formato de este proyecto: descomprímelo en `data/` y entrena.

En CPU, cada fold tarda varios minutos por cada cien fotos. Con un dataset real, usa una GPU (Google Colab):

```python
!git clone <tu-repo> && cd famacha_net && pip install -r requirements.txt
!python scripts/prepare.py && python scripts/train.py
```

### Qué genera `runs/exp1/`

| Archivo | Contenido |
|---|---|
| `fold_*.keras` | Modelos del ensamble |
| `model_meta.json` | Umbral “tratar”, codificador de metadatos, normalización de PCV |
| `splits.csv` | Qué animales quedaron en dev y en test (reproducibilidad) |
| `report_validacion_cruzada.json` | Métricas out-of-fold |
| `report_test.json` | Métricas en animales nunca vistos, con IC 95 % por bootstrap **a nivel animal** |
| `confusion_test.png`, `gradcam_test.png` | Figuras para tesis o presentación |
| `famacha_net.tflite` + `famacha_net_app_config.json` | Modelo y configuración para la app |

---

## 4. Decisiones técnicas (y por qué)

| Decisión | Justificación |
|---|---|
| **Regresión ordinal CORAL** en lugar de softmax | FAMACHA es ordinal: confundir 1 con 5 es mucho peor que confundir 3 con 4. CORAL garantiza predicciones monótonas y penaliza según la distancia entre grados. |
| **Split por animal** (`StratifiedGroupKFold`) | Evita la fuga de datos, el error más común en esta área. |
| **Sin jitter de tono ni de saturación** | El color es el diagnóstico. Un jitter de tono convertiría un 2 en un 4 sin cambiar la etiqueta. Solo se aplican transformaciones geométricas, brillo leve y ±4 % de ganancia por canal. |
| **Preprocesamiento dentro del grafo** | La app solo recorta y redimensiona. No hay que reimplementar nada en Kotlin o Dart, y se evitan discrepancias entre entrenamiento y producción. |
| **Pesos por clase** | Los grados extremos son raros pero críticos. |
| **Fine-tuning en dos fases con BatchNorm congelada** | Estable con datasets pequeños. |
| **Umbral “tratar” ajustado a una sensibilidad del 95 %** | No detectar a un animal anémico puede costarle la vida. El umbral se elige con predicciones out-of-fold, nunca con el test. |
| **Ensamble de K folds + desacuerdo entre modelos** | Mejora la precisión y sirve como medida de incertidumbre. Si los modelos no coinciden, la app pide repetir la foto. |
| **Bootstrap por animal** | Las fotos del mismo animal no son independientes. Tratarlas como si lo fueran da intervalos de confianza falsamente estrechos. |
| **Grad-CAM** | Comprueba que el modelo mira la conjuntiva y no los dedos ni el pelaje. |

---

## 5. Integración con la app y errores detectados en las capturas actuales

**Contrato del `.tflite`:**
- Entrada `image`: `float32 [1,224,224,3]`, RGB en 0–255, recorte cuadrado del párpado, **sin normalizar**.
- Entrada `meta` (si aplica): codificar igual que `meta_encoder` en `famacha_net_app_config.json` (valor estandarizado + indicador de faltante por campo).
- Salida: 4 logits CORAL → `P(y>k) = sigmoid(logit_k)`; `grado = 1 + #{P(y>k) > 0.5}`; `P(tratar) = sigmoid(logit[2])`, que se compara con `threshold_treat`.
- Recomendación: mostrar en la cámara una **guía rectangular** para centrar el párpado y recortar esa región. Así no hace falta un detector.

**Errores en las capturas de la versión actual:**
1. **El color de la tarjeta no corresponde al grado.** “Grado 5” y uno de los “Grado 4” salen en **verde**, mientras que otro “Grado 4” sale en rojo. El color debe derivarse del grado (1–2 verde, 3 amarillo, 4–5 rojo). Ver `RECOMMENDATION` en `famacha/utils.py`.
2. **Grado 4 y grado 5 muestran el mismo mensaje.** El 5 debería indicar urgencia y consulta veterinaria.
3. **Posible sesgo hacia el grado 4.** En al menos una foto (la cabra negra) la mucosa se ve claramente roja-rosada y la app dice 4. Es una observación visual, no un diagnóstico, pero vale la pena revisar la distribución de clases del modelo actual.
4. Falta un **control de calidad**: la app no avisa cuando la foto está borrosa, oscura o es poco concluyente. `predict.py` sí lo hace.
5. “Diagnóstico instantáneo” promete demasiado. Conviene un aviso del tipo *“herramienta de apoyo; confirme con su veterinario”*.

---

## Referencias clave
- Cao, Mirjalili & Raschka (2020). *Rank consistent ordinal regression for neural networks* (CORAL). Pattern Recognition Letters.
- Kaplan et al. (2004). *Validation of the FAMACHA© eye color chart for detecting clinical anemia in sheep and goats on farms in the southern United States*. Veterinary Parasitology 123.
- Bath & van Wyk (2009). *The Five Point Check© for targeted selective treatment of internal parasites in small ruminants*. Small Ruminant Research.
- *Smart farming with AI: enhancing anemia detection in small ruminants* (2025). Veterinary Parasitology 338, 110525.
- *Image analysis to automatically classify anemia based on Famacha score in sheep using ocular conjunctiva images* (2023). PubMed 38023419.
