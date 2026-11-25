"""Carga del ensamble entrenado y predicción."""
import glob
import os

import numpy as np
import keras

from . import layers as _custom_layers  # noqa: F401  (registra capas)
from . import ordinal
from .data import MetaEncoder, load_json, load_image


class FamachaEnsemble:
    def __init__(self, run_dir):
        self.meta = load_json(os.path.join(run_dir, "model_meta.json"))
        paths = sorted(glob.glob(os.path.join(run_dir, "fold_*.keras")))
        if not paths:
            raise FileNotFoundError(f"No hay modelos fold_*.keras en {run_dir}")
        self.models = [keras.models.load_model(p, compile=False) for p in paths]
        self.size = self.meta["image_size"]
        self.threshold = self.meta["threshold_treat"]
        self.encoder = MetaEncoder.from_dict(self.meta["meta_encoder"]) if self.meta["n_meta"] else None

    def _inputs(self, images, meta_df=None):
        x = {"image": images}
        if self.encoder is not None:
            if meta_df is None:
                import pandas as pd
                meta_df = pd.DataFrame(index=range(len(images)))
            x["meta"] = self.encoder.transform(meta_df)
        return x

    def predict_logits(self, images, meta_df=None, batch=32):
        x = self._inputs(images, meta_df)
        outs = [m.predict(x, batch_size=batch, verbose=0) for m in self.models]
        logits = np.mean([o["famacha"] for o in outs], axis=0)
        pcv = None
        if "pcv" in outs[0]:
            mu, sd = self.meta.get("pcv_norm", (25.0, 7.0))
            pcv = np.mean([o["pcv"] for o in outs], axis=0)[:, 0] * sd + mu
        # desacuerdo entre modelos del ensamble = incertidumbre epistémica
        spread = np.std([ordinal.predict_grade(o["famacha"]) for o in outs], axis=0)
        return logits, pcv, spread

    def predict_paths(self, paths, meta_df=None):
        imgs = np.stack([load_image(p, self.size).numpy() for p in paths])
        return self.predict_logits(imgs, meta_df)


def interpret(logits_row, threshold, spread=0.0, pcv=None, min_conf=0.55):
    from .utils import RECOMMENDATION
    lg = logits_row[None]
    probs = ordinal.cumprobs_to_classprobs(ordinal.logits_to_cumprobs(lg))[0]
    grade = int(ordinal.predict_grade(lg)[0])
    pt = float(ordinal.p_treat(lg)[0])
    conf = float(probs[grade - 1])
    color, msg = RECOMMENDATION[grade]
    res = {
        "grado": grade,
        "probabilidades": {str(k + 1): round(float(p), 3) for k, p in enumerate(probs)},
        "confianza": round(conf, 3),
        "p_tratar": round(pt, 3),
        "tratar": bool(pt >= threshold),
        "color": color,
        "mensaje": msg,
    }
    if pcv is not None:
        res["pcv_estimado"] = round(float(pcv), 1)
    if conf < min_conf or spread > 0.6:
        res["aviso"] = "Predicción poco confiable: repite la foto con buena luz natural y el párpado bien expuesto."
    return res
