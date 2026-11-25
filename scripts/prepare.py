"""Paso 1 — Valida el CSV, recorta la conjuntiva y hace control de calidad.

Uso:  python scripts/prepare.py --config config.yaml

- Si el CSV trae bbox (x1,y1,x2,y2), recorta un cuadrado alrededor con margen.
  Es lo más recomendable: anota las cajas con CVAT / Label Studio / Roboflow.
- Si no hay bbox, toma un recorte cuadrado central (la app debe pedir al
  usuario que centre el párpado en una guía en pantalla).
- Marca fotos borrosas, oscuras o sobreexpuestas para que las revises.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from famacha.utils import load_config, resolve, log

SAVE_SIZE = 320


def square_crop(img, box, margin):
    h, w = img.shape[:2]
    if box is None:
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return img[y0:y0 + s, x0:x0 + s]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    s = min(s, h, w)
    x0 = int(np.clip(cx - s / 2, 0, w - s)); y0 = int(np.clip(cy - s / 2, 0, h - s))
    return img[y0:y0 + int(s), x0:x0 + int(s)]


def quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return {
        "blur_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "overexposed_frac": float((gray > 250).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(resolve(args.config, os.getcwd()))
    dc = cfg["data"]

    csv_path = resolve(dc["labels_csv"])
    base = os.path.dirname(csv_path)
    df = pd.read_csv(csv_path)

    # --- validación ---
    req = {"image_path", "animal_id", "famacha"}
    missing = req - set(df.columns)
    if missing:
        sys.exit(f"Faltan columnas obligatorias: {missing}")
    bad = ~df["famacha"].isin([1, 2, 3, 4, 5])
    if bad.any():
        sys.exit(f"famacha fuera de 1-5 en filas: {df.index[bad].tolist()[:20]}")
    if "pcv" in df:
        df["pcv"] = pd.to_numeric(df["pcv"], errors="coerce")
        odd = df["pcv"].notna() & ~df["pcv"].between(5, 55)
        if odd.any():
            log(f"[aviso] PCV fuera de rango fisiológico en {odd.sum()} filas")

    out_dir = resolve(dc["processed_dir"])
    os.makedirs(out_dir, exist_ok=True)
    has_box = all(c in df for c in ["x1", "y1", "x2", "y2"])

    rows = []
    for i, r in df.iterrows():
        src = r["image_path"] if os.path.isabs(r["image_path"]) else os.path.join(base, r["image_path"])
        if not os.path.exists(src):
            log(f"[omitida] no existe: {src}"); continue
        img = np.array(ImageOps.exif_transpose(Image.open(src)).convert("RGB"))
        box = None
        if has_box and pd.notna(r[["x1", "y1", "x2", "y2"]]).all():
            box = tuple(float(r[c]) for c in ["x1", "y1", "x2", "y2"])
        crop = square_crop(img, box, dc["crop_margin"])
        crop = cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
        dst = os.path.join(out_dir, f"{i:06d}.png")
        Image.fromarray(crop).save(dst)
        q = quality(crop)
        rows.append({**r.to_dict(), "proc_path": dst, "has_bbox": box is not None, **q})

    out = pd.DataFrame(rows)
    out["flag_quality"] = (out["blur_var"] < 30) | (out["brightness"] < 40) | (out["overexposed_frac"] > 0.25)
    out.to_csv(os.path.join(out_dir, "processed.csv"), index=False)

    log(f"\nFotos procesadas: {len(out)}  |  animales: {out['animal_id'].nunique()}")
    log("Distribución FAMACHA:\n" + out["famacha"].value_counts().sort_index().to_string())
    log(f"Con bbox: {out['has_bbox'].mean():.0%}  |  marcadas por calidad: {out['flag_quality'].sum()} (revísalas)")
    per_animal = out.groupby("animal_id").size()
    log(f"Fotos por animal: mediana {per_animal.median():.0f}, máx {per_animal.max()}")
    if out["famacha"].value_counts().min() < 30:
        log("[aviso] Algún grado tiene < 30 fotos: el modelo no aprenderá bien esa clase.")


if __name__ == "__main__":
    main()
