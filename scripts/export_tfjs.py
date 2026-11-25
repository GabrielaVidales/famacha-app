"""Paso 4b — Exporta el ensamble a TensorFlow.js para la página web "Análisis Rápido".

Uso:  python scripts/export_tfjs.py --run runs/exp1 [--single]

Genera runs/exp1/tfjs/ con:
  model.json + group1-shard*.bin   (modelo de grafo, pesos float16)
  famacha_net_app_config.json      (tamaño de entrada, umbral de tratamiento, codificador de metadatos)
En la página: pestaña Modelo → "Seleccionar archivos del modelo" → elige TODOS los archivos de esa carpeta.

Requiere:  pip install --no-deps tensorflowjs==4.22.0 && pip install tensorflow-hub
(Con --no-deps se evitan jax y tensorflow-decision-forests, que el convertidor no
necesita para este modelo; si faltan, este script crea sustitutos vacíos automáticamente.)
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from famacha.utils import resolve, log  # noqa: E402

import tensorflow as tf
import keras

from famacha import layers as _l  # noqa: F401
from famacha.data import load_json, save_json


def _make_stubs(d):
    """Módulos vacíos para dependencias opcionales de tensorflowjs que no se usan aquí."""
    os.makedirs(os.path.join(d, "tensorflow_decision_forests"), exist_ok=True)
    os.makedirs(os.path.join(d, "jax", "experimental"), exist_ok=True)
    open(os.path.join(d, "tensorflow_decision_forests", "__init__.py"), "w").close()
    with open(os.path.join(d, "jax", "__init__.py"), "w") as f:
        f.write("import types\nmonitoring = types.SimpleNamespace(record_scalar=lambda *a, **k: None, "
                "record_event=lambda *a, **k: None, register_event_listener=lambda *a, **k: None)\n"
                "def __getattr__(name):\n    return types.SimpleNamespace()\n")
    with open(os.path.join(d, "jax", "experimental", "__init__.py"), "w") as f:
        f.write("jax2tf = None\n")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--single", action="store_true", help="solo el fold 0 (más ligero)")
    ap.add_argument("--to-web", action="store_true", help="copiar el resultado a web/model/ para que la página lo cargue sola")
    args = ap.parse_args()
    run = resolve(args.run, os.getcwd())
    meta = load_json(os.path.join(run, "model_meta.json"))
    paths = sorted(glob.glob(os.path.join(run, "fold_*.keras")))[: 1 if args.single else None]
    models = [keras.models.load_model(p, compile=False) for p in paths]
    for i, m in enumerate(models):
        m.name = f"fold_{i}"

    s, n_meta = meta["image_size"], meta["n_meta"]
    ins = {"image": keras.Input((s, s, 3), name="image")}
    if n_meta:
        ins["meta"] = keras.Input((n_meta,), name="meta")
    outs = [m(ins)["famacha"] for m in models]
    logits = keras.layers.Average()(outs) if len(outs) > 1 else outs[0]
    ens = keras.Model(ins, logits)

    # Firma explícita con UNA salida nombrada: evita que el convertidor tome otro nodo.
    specs = [tf.TensorSpec([None, s, s, 3], tf.float32, name="image")]
    if n_meta:
        specs.append(tf.TensorSpec([None, n_meta], tf.float32, name="meta"))

    class Serving(tf.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        @tf.function(input_signature=specs)
        def serve(self, *a):
            x = {"image": a[0]}
            if n_meta:
                x["meta"] = a[1]
            return {"famacha_logits": tf.identity(self.model(x, training=False), name="famacha_logits")}

    mod = Serving(ens)
    sm_dir = os.path.join(run, "saved_model_tfjs")
    shutil.rmtree(sm_dir, ignore_errors=True)
    tf.saved_model.save(mod, sm_dir, signatures={"serving_default": mod.serve})

    out = os.path.join(run, "tfjs")
    shutil.rmtree(out, ignore_errors=True)
    cmd = [sys.executable, "-m", "tensorflowjs.converters.converter", "--input_format=tf_saved_model",
           "--output_format=tfjs_graph_model", "--signature_name=serving_default",
           "--quantize_float16=*", sm_dir, out]
    env = dict(os.environ)
    probe = subprocess.run([sys.executable, "-c", "import tensorflowjs"], capture_output=True, text=True, env=env)
    if probe.returncode != 0:
        if "No module named 'tensorflowjs'" in probe.stderr:
            sys.exit("Falta tensorflowjs: pip install --no-deps tensorflowjs==4.22.0 && pip install tensorflow-hub")
        stubs = _make_stubs(os.path.join(run, ".tfjs_stubs"))
        env["PYTHONPATH"] = stubs + os.pathsep + env.get("PYTHONPATH", "")
        log("[aviso] tensorflowjs pide jax / tensorflow-decision-forests: uso sustitutos vacíos (no se usan en esta conversión).")
    subprocess.run(cmd, check=True, env=env)
    save_json({k: meta[k] for k in ["image_size", "n_meta", "threshold_treat", "meta_encoder"]},
              os.path.join(out, "famacha_net_app_config.json"))
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(out, "*"))) / 1e6
    log(f"TF.js: {out}  ({size:.1f} MB, {len(models)} modelo(s)).")
    if args.to_web:
        web_model = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "model")
        os.makedirs(web_model, exist_ok=True)
        for f in glob.glob(os.path.join(web_model, "*")):
            if not f.endswith("README.md"):
                os.remove(f)
        for f in glob.glob(os.path.join(out, "*")):
            shutil.copy(f, web_model)
        info = {"run": os.path.basename(run), "folds": len(models)}
        rp = os.path.join(run, "report_test.json")
        if os.path.exists(rp):
            r = load_json(rp)
            o, t = r["ordinal"], r["tratar_vs_famacha_humano"]
            info["test"] = {"n_fotos": r["n_fotos"], "n_grupos": r["n_animales"],
                            "accuracy_5": o["accuracy_5_grados"], "accuracy_pm1": o["accuracy_+-1_grado"],
                            "kappa": o["kappa_cuadratica"], "tratar_acc": t["accuracy"],
                            "tratar_sens": t["sensitivity"], "tratar_spec": t["specificity"]}
        save_json(info, os.path.join(web_model, "model_info.json"))
        log(f"Copiado a {web_model}: abre la página con  python serve.py  y la red se carga sola.")
    else:
        log("En la página: pestaña Modelo → selecciona TODOS los archivos de esa carpeta (o usa --to-web).")


if __name__ == "__main__":
    main()
