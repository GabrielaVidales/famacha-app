"""Explicabilidad — Grad-CAM: ¿el modelo mira la conjuntiva o los dedos/pelo?

Si el mapa de calor cae sobre el pulgar o el pelaje, el modelo aprendió un
atajo (p. ej. "pelaje negro = grado X") y su accuracy no generalizará.

Uso:  python scripts/gradcam.py --run runs/exp1 --n 12
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from famacha.utils import resolve, log  # noqa: E402

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from famacha import layers as _l  # noqa: F401
from famacha import ordinal
from famacha.data import MetaEncoder, load_image, load_json


def gradcam(model, x):
    gm = keras.Model(model.inputs, [model.get_layer("feature_map").output, model.output["famacha"]])
    with tf.GradientTape() as tape:
        fmap, logits = gm(x, training=False)
        score = tf.reduce_sum(tf.sigmoid(logits), axis=-1)  # grado esperado - 1
    grads = tape.gradient(score, fmap)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * fmap, axis=-1))
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    return cam.numpy(), logits.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    run = resolve(args.run, os.getcwd())
    meta = load_json(os.path.join(run, "model_meta.json"))
    model = keras.models.load_model(sorted(glob.glob(os.path.join(run, "fold_*.keras")))[0], compile=False)

    df = pd.read_csv(os.path.join(run, "splits.csv"))
    df = (df[df.split == "test"].sample(frac=1, random_state=0)
          .groupby("famacha").head(max(1, args.n // 5)).sort_values("famacha").reset_index(drop=True))
    imgs = np.stack([load_image(p, meta["image_size"]).numpy() for p in df.proc_path])
    x = {"image": imgs}
    if meta["n_meta"]:
        x["meta"] = MetaEncoder.from_dict(meta["meta_encoder"]).transform(df)
    cams, logits = gradcam(model, x)
    preds = ordinal.predict_grade(logits)

    cols = 4; rows = int(np.ceil(len(df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for i, ax in enumerate(np.ravel(axes)[:len(df)]):
        cam = tf.image.resize(cams[i][..., None], imgs.shape[1:3]).numpy()[..., 0]
        ax.imshow(imgs[i].astype("uint8")); ax.imshow(cam, cmap="jet", alpha=0.4)
        ax.set_title(f"real {df.famacha[i]} | pred {preds[i]}", fontsize=9)
    fig.tight_layout()
    out = os.path.join(run, "gradcam_test.png")
    fig.savefig(out, dpi=130); log(f"Guardado {out}")


if __name__ == "__main__":
    main()
