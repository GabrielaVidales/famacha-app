"""Predicción para una foto nueva.

Uso:
  python scripts/predict.py --run runs/exp1 --image foto.jpg
  python scripts/predict.py --run runs/exp1 --image foto.jpg --bbox 410 880 905 1210 --bcs 2 --bottle_jaw 1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from famacha.utils import resolve  # noqa: E402

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from famacha.inference import FamachaEnsemble, interpret
from scripts.prepare import square_crop, quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--bbox", nargs=4, type=float, help="x1 y1 x2 y2 de la conjuntiva")
    ap.add_argument("--meta", nargs="*", default=[], help="pares campo=valor, p. ej. bcs=2 bottle_jaw=1")
    args = ap.parse_args()

    ens = FamachaEnsemble(resolve(args.run, os.getcwd()))
    img = np.array(ImageOps.exif_transpose(Image.open(args.image)).convert("RGB"))
    crop = square_crop(img, tuple(args.bbox) if args.bbox else None, 0.15)
    q = quality(crop)
    x = cv2.resize(crop, (ens.size, ens.size), interpolation=cv2.INTER_AREA).astype("float32")[None]

    meta = {k: float(v) for k, v in (kv.split("=") for kv in args.meta)}
    logits, pcv, spread = ens.predict_logits(x, pd.DataFrame([meta]) if ens.encoder else None)
    res = interpret(logits[0], ens.threshold, float(spread[0]), None if pcv is None else pcv[0])
    if q["blur_var"] < 30 or q["brightness"] < 40 or q["overexposed_frac"] > 0.25:
        res["aviso"] = "Foto borrosa, oscura o sobreexpuesta: repítela."
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
