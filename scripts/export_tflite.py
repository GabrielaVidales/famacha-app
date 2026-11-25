"""Paso 4 — Exporta a TensorFlow Lite para la app móvil.

Uso:  python scripts/export_tflite.py --run runs/exp1 [--single]

Por defecto exporta el ENSAMBLE (promedio de logits de los K folds) en un
solo .tflite con pesos float16. Con --single exporta solo el fold 0 (K veces
más ligero y rápido; algo menos preciso).

Contrato de la app:
  entrada "image": float32 [1,224,224,3], RGB 0-255, recorte cuadrado del párpado
  entrada "meta" (si aplica): float32 [1,n_meta] — codificar igual que model_meta.json
  salida: logits CORAL [1,4] -> P(y>k)=sigmoid(logit_k); grado = 1 + #(P>0.5)
          P(tratar) = sigmoid(logit_2)  -> comparar con threshold_treat
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from famacha.utils import resolve, log  # noqa: E402

import numpy as np
import tensorflow as tf
import keras

from famacha import layers as _l  # noqa: F401
from famacha.data import load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()
    run = resolve(args.run, os.getcwd())
    meta = load_json(os.path.join(run, "model_meta.json"))
    paths = sorted(glob.glob(os.path.join(run, "fold_*.keras")))
    if args.single:
        paths = paths[:1]
    models = [keras.models.load_model(p, compile=False) for p in paths]
    for i, m in enumerate(models):
        m.name = f"fold_{i}"  # nombres únicos dentro del ensamble

    s = meta["image_size"]
    ins = {"image": keras.Input((s, s, 3), name="image")}
    if meta["n_meta"]:
        ins["meta"] = keras.Input((meta["n_meta"],), name="meta")
    outs = [m(ins)["famacha"] for m in models]
    logits = keras.layers.Average(name="famacha_logits")(outs) if len(outs) > 1 else outs[0]
    export_model = keras.Model(ins, logits, name="famacha_ensemble")

    # Keras 3 -> SavedModel -> TFLite (float16)
    sm_dir = os.path.join(run, "saved_model")
    export_model.export(sm_dir, format="tf_saved_model", verbose=False)
    conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tfl = conv.convert()
    out = os.path.join(run, "famacha_net.tflite")
    with open(out, "wb") as f:
        f.write(tfl)

    # Verificación: TFLite vs Keras sobre una entrada aleatoria
    interp = tf.lite.Interpreter(model_content=tfl); interp.allocate_tensors()
    x = np.random.uniform(0, 255, (1, s, s, 3)).astype("float32")
    feed = {"image": x}
    if meta["n_meta"]:
        feed["meta"] = np.zeros((1, meta["n_meta"]), "float32")
    for d in interp.get_input_details():
        key = "meta" if "meta" in d["name"] else "image"  # nombres: serving_default_image / _meta
        interp.set_tensor(d["index"], feed[key])
    interp.invoke()
    y_tfl = interp.get_tensor(interp.get_output_details()[0]["index"])
    y_k = export_model(feed, training=False).numpy()
    log(f"TFLite: {out}  ({len(tfl) / 1e6:.1f} MB, {len(models)} modelo(s))  |  max |Δlogit| vs Keras = {np.abs(y_tfl - y_k).max():.4f}")
    save_json({k: meta[k] for k in ["image_size", "n_meta", "threshold_treat", "meta_encoder"]},
              os.path.join(run, "famacha_net_app_config.json"))


if __name__ == "__main__":
    main()
