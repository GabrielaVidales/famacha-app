"""Paso 3 — Evalúa el ensamble en cualquier CSV procesado (p. ej. una granja
nueva = validación externa) y dibuja la matriz de confusión.

Uso:
  python scripts/evaluate.py --run runs/exp1                       # usa el test guardado
  python scripts/evaluate.py --run runs/exp1 --csv otra_granja/processed.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from famacha.utils import resolve, log, load_config  # noqa: E402

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from famacha import ordinal
from famacha.data import save_json
from famacha.inference import FamachaEnsemble
from famacha.metrics import full_report


def plot_confusion(cm, path, title):
    cm = np.array(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(5), [1, 2, 3, 4, 5]); ax.set_yticks(range(5), [1, 2, 3, 4, 5])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real (evaluador humano)"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--csv", help="processed.csv a evaluar (por defecto: split test)")
    args = ap.parse_args()
    run = resolve(args.run, os.getcwd())
    cfg = load_config(os.path.join(run, "config.yaml"))

    if args.csv:
        df = pd.read_csv(resolve(args.csv, os.getcwd())); name = "externo"
    else:
        df = pd.read_csv(os.path.join(run, "splits.csv")); df = df[df.split == "test"].reset_index(drop=True); name = "test"

    ens = FamachaEnsemble(run)
    logits, _, _ = ens.predict_paths(df["proc_path"].tolist(), df)
    grades, pt = ordinal.predict_grade(logits), ordinal.p_treat(logits)
    rep = full_report(df, grades, pt, ens.threshold, cfg["data"]["pcv_anemia_threshold"])
    save_json(rep, os.path.join(run, f"report_{name}.json"))
    plot_confusion(rep["ordinal"]["confusion_matrix"], os.path.join(run, f"confusion_{name}.png"),
                   f"FAMACHA — {name} (n={len(df)})")
    log(f"Accuracy 5 grados {rep['ordinal']['accuracy_5_grados']:.3f} | ±1 {rep['ordinal']['accuracy_+-1_grado']:.3f} | "
        f"tratar acc {rep['tratar_vs_famacha_humano']['accuracy']:.3f}")
    log(f"Guardado en {run}")


if __name__ == "__main__":
    main()
