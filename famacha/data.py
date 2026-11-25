"""Carga de datos, codificación de metadatos y pipeline tf.data."""
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers

from .ordinal import grade_to_levels


# ---------------- metadatos ----------------
class MetaEncoder:
    """Imputa con la mediana de ENTRENAMIENTO, estandariza numéricos y
    agrega un indicador de 'faltante' por columna (el dato ausente también
    es información)."""

    def __init__(self, numeric=(), binary=()):
        self.numeric, self.binary = list(numeric), list(binary)
        self.stats = {}

    @property
    def cols(self):
        return self.numeric + self.binary

    @property
    def n_features(self):
        return 2 * len(self.cols)

    def fit(self, df):
        for c in self.numeric:
            v = pd.to_numeric(df.get(c), errors="coerce")
            med = float(np.nanmedian(v)) if v.notna().any() else 0.0
            std = float(np.nanstd(v)) if v.notna().sum() > 1 else 1.0
            self.stats[c] = {"fill": med, "mean": med, "std": std or 1.0}
        for c in self.binary:
            v = pd.to_numeric(df.get(c), errors="coerce")
            self.stats[c] = {"fill": float(np.nanmedian(v)) if v.notna().any() else 0.0}
        return self

    def transform(self, df):
        out = []
        for c in self.cols:
            v = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
            miss = v.isna().astype("float32").values
            v = v.fillna(self.stats[c]["fill"]).astype("float32").values
            if c in self.numeric:
                v = (v - self.stats[c]["mean"]) / self.stats[c]["std"]
            out += [v, miss]
        return np.stack(out, axis=1).astype("float32") if out else np.zeros((len(df), 0), "float32")

    def to_dict(self):
        return {"numeric": self.numeric, "binary": self.binary, "stats": self.stats}

    @classmethod
    def from_dict(cls, d):
        enc = cls(d["numeric"], d["binary"])
        enc.stats = d["stats"]
        return enc


# ---------------- aumentación ----------------
def build_augmenter():
    """Aumentación GEOMÉTRICA libre, pero COLOR casi intacto.

    El color de la conjuntiva ES la señal diagnóstica: un hue/saturation jitter
    agresivo convertiría un grado 2 en un grado 4 sin cambiar la etiqueta.
    Solo se permite brillo leve y una pequeña ganancia por canal (simula
    diferencias de iluminante que el balance de blancos debe absorber)."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06, fill_mode="reflect"),
        layers.RandomZoom((-0.15, 0.10), fill_mode="reflect"),
        layers.RandomTranslation(0.08, 0.08, fill_mode="reflect"),
        layers.RandomBrightness(0.08, value_range=(0, 255)),
    ], name="augment")


def channel_gain(img, max_dev=0.04):
    g = tf.random.uniform([1, 1, 3], 1 - max_dev, 1 + max_dev)
    return tf.clip_by_value(img * g, 0.0, 255.0)


def load_image(path, size):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(tf.cast(img, tf.float32), (size, size), antialias=True)
    return img


def make_dataset(df, size, meta=None, with_pcv=False, class_w=None, training=False, batch=16, seed=42,
                 pcv_norm=(25.0, 7.0)):
    paths = df["proc_path"].astype(str).values
    levels = grade_to_levels(df["famacha"].values)
    sw_fam = np.array([class_w.get(int(g), 1.0) for g in df["famacha"]], "float32") if class_w else np.ones(len(df), "float32")

    x = {"path": paths}
    if meta is not None:
        x["meta"] = meta
    y = {"famacha": levels}
    sw = {"famacha": sw_fam}
    if with_pcv:
        pcv = pd.to_numeric(df["pcv"], errors="coerce").values.astype("float32")
        z = (pcv - pcv_norm[0]) / pcv_norm[1]                    # PCV estandarizado
        y["pcv"] = np.nan_to_num(z, nan=0.0).reshape(-1, 1).astype("float32")
        sw["pcv"] = (~np.isnan(pcv)).astype("float32") * 0.5     # sin PCV -> peso 0

    ds = tf.data.Dataset.from_tensor_slices((x, y, sw))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    aug = build_augmenter() if training else None

    def _map(x, y, sw):
        img = load_image(x["path"], size)
        if training:
            img = aug(img[None], training=True)[0]
            img = channel_gain(img)
        xo = {"image": img}
        if "meta" in x:
            xo["meta"] = x["meta"]
        return xo, y, sw

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def class_weights(grades):
    """Pesos inversamente proporcionales a la frecuencia (los grados 1 y 5
    suelen ser raros y son justo los que importan)."""
    vals, counts = np.unique(grades, return_counts=True)
    w = len(grades) / (len(vals) * counts)
    return {int(v): float(x) for v, x in zip(vals, w)}


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
