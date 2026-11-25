"""Genera un dataset SINTÉTICO solo para probar que el pipeline corre de
punta a punta. Los números que salgan con estos datos NO significan nada
sobre el desempeño con fotos reales.

Uso:  python scripts/make_synthetic.py --out data/synth --animals 40 --per-animal 6
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

GRADE_RGB = {1: (175, 30, 45), 2: (205, 75, 88), 3: (222, 128, 134), 4: (232, 172, 172), 5: (240, 214, 208)}
FUR = [(25, 25, 28), (235, 232, 225), (150, 110, 80), (200, 190, 175)]


def make_image(rng, grade, fur):
    h, w = 480, 640
    img = np.full((h, w, 3), fur, np.float32) + rng.normal(0, 18, (h, w, 3))
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    cx, cy = int(rng.uniform(220, 420)), int(rng.uniform(180, 300))
    ax, ay = int(rng.uniform(70, 120)), int(rng.uniform(25, 45))
    col = np.array(GRADE_RGB[grade], np.float32) + rng.normal(0, 7, 3)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), rng.uniform(-15, 15), 0, 180, 255, -1)
    tex = col + rng.normal(0, 10, (h, w, 3))
    img[mask > 0] = tex[mask > 0]
    cv2.ellipse(img, (cx, cy - 5), (int(ax * .5), int(ay * .5)), 0, 180, 360, (15, 15, 15), -1)  # globo ocular
    for _ in range(3):  # brillos especulares
        cv2.circle(img, (cx + int(rng.uniform(-ax, ax) * .6), cy + int(rng.uniform(3, ay * .6))), 3, (255, 255, 255), -1)
    for side in (-1, 1):  # "dedos"
        cv2.ellipse(img, (cx + int(rng.uniform(-40, 40)), cy + side * int(ay + 70)), (95, 45), 0, 0, 360,
                    (205, 160, 135), -1)
    tint = rng.uniform(0.88, 1.12, 3) * rng.uniform(0.85, 1.1)
    img = np.clip(img * tint, 0, 255).astype(np.uint8)
    return img, (cx - ax, cy - ay, cx + ax, cy + ay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--animals", type=int, default=40)
    ap.add_argument("--per-animal", type=int, default=6)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    rows = []
    for k in range(a.animals):
        aid = f"A{k:03d}"; fur = FUR[rng.integers(len(FUR))]; base = int(rng.integers(1, 6))
        goat = int(rng.integers(0, 2))
        for j in range(a.per_animal):
            g = int(np.clip(base + rng.choice([-1, 0, 0, 0, 1]), 1, 5))
            img, box = make_image(rng, g, fur)
            p = f"raw/{aid}_{j}.jpg"
            cv2.imwrite(os.path.join(a.out, p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            pcv = float(np.clip(38 - 5 * g + rng.normal(0, 2.5), 8, 45))
            rows.append({"image_path": p, "animal_id": aid, "famacha": g,
                         "famacha_2": int(np.clip(g + rng.choice([-1, 0, 0, 0, 1]), 1, 5)),
                         "pcv": round(pcv, 1) if rng.random() > 0.2 else np.nan,
                         "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                         "bcs": float(np.clip(4 - 0.4 * g + rng.normal(0, .5), 1, 5)),
                         "dag_score": int(np.clip(g - 1 + rng.integers(-1, 2), 0, 5)),
                         "age_months": int(rng.integers(4, 72)), "days_since_deworm": int(rng.integers(10, 200)),
                         "bottle_jaw": int(g == 5 and rng.random() < .5), "nasal_discharge": int(rng.random() < .1),
                         "diarrhea": int(rng.random() < .2 * g / 5), "is_goat": goat, "rainy_season": int(rng.random() < .5)})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "labels.csv"), index=False)
    print(f"{len(rows)} imágenes sintéticas en {a.out}")


if __name__ == "__main__":
    main()
