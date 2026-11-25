"""Métricas de evaluación con intervalos de confianza por bootstrap a nivel ANIMAL."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score)


def binary_stats(y, yhat):
    tp = int(((y == 1) & (yhat == 1)).sum()); tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"accuracy": (tp + tn) / max(len(y), 1), "sensitivity": sens, "specificity": spec,
            "ppv": tp / (tp + fp) if tp + fp else float("nan"),
            "npv": tn / (tn + fn) if tn + fn else float("nan"),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def ordinal_report(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return {
        "accuracy_5_grados": accuracy_score(y, yhat),
        "balanced_accuracy": balanced_accuracy_score(y, yhat),
        "macro_f1": f1_score(y, yhat, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0),
        "accuracy_+-1_grado": float(np.mean(np.abs(y - yhat) <= 1)),
        "mae_grados": float(np.mean(np.abs(y - yhat))),
        "kappa_cuadratica": cohen_kappa_score(y, yhat, weights="quadratic", labels=[1, 2, 3, 4, 5]),
        "confusion_matrix": confusion_matrix(y, yhat, labels=[1, 2, 3, 4, 5]).tolist(),
    }


def full_report(df, grade_pred, p_treat, threshold, pcv_thr=19.0):
    """df: columnas famacha, animal_id, opcional pcv y famacha_2."""
    y = df["famacha"].values.astype(int)
    rep = {"n_fotos": len(df), "n_animales": int(df["animal_id"].nunique())}
    rep["ordinal"] = ordinal_report(y, grade_pred)

    # Decisión clínica: tratar (FAMACHA >= 4) vs no tratar
    y_treat = (y >= 4).astype(int)
    rep["tratar_vs_famacha_humano"] = binary_stats(y_treat, (p_treat >= threshold).astype(int))
    if len(np.unique(y_treat)) == 2:
        rep["tratar_vs_famacha_humano"]["auc"] = roc_auc_score(y_treat, p_treat)

    # Contra el estándar de oro (PCV), si existe
    if "pcv" in df and df["pcv"].notna().any():
        m = df["pcv"].notna().values
        y_an = (df.loc[m, "pcv"].values <= pcv_thr).astype(int)
        rep["modelo_vs_PCV"] = binary_stats(y_an, (p_treat[m] >= threshold).astype(int))
        rep["humano_vs_PCV"] = binary_stats(y_an, (y[m] >= 4).astype(int))

    # Techo realista: acuerdo entre dos evaluadores humanos
    if "famacha_2" in df and df["famacha_2"].notna().any():
        m = df["famacha_2"].notna().values
        rep["acuerdo_inter_evaluador"] = ordinal_report(y[m], df.loc[m, "famacha_2"].values.astype(int))

    rep["ic95_bootstrap_por_animal"] = bootstrap_ci(df, grade_pred, p_treat, threshold)
    return rep


def bootstrap_ci(df, grade_pred, p_treat, threshold, n=1000, seed=0):
    """Remuestrea ANIMALES (no fotos): las fotos de un mismo animal no son
    independientes y tratarlas como tales da intervalos falsamente estrechos."""
    rng = np.random.default_rng(seed)
    animals = df["animal_id"].values
    uniq = np.unique(animals)
    idx_by = {a: np.where(animals == a)[0] for a in uniq}
    y = df["famacha"].values.astype(int)
    acc, acc_bin, sens = [], [], []
    for _ in range(n):
        idx = np.concatenate([idx_by[a] for a in rng.choice(uniq, len(uniq), replace=True)])
        acc.append(np.mean(y[idx] == grade_pred[idx]))
        s = binary_stats((y[idx] >= 4).astype(int), (p_treat[idx] >= threshold).astype(int))
        acc_bin.append(s["accuracy"]); sens.append(s["sensitivity"])
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"accuracy_5_grados": q(acc), "accuracy_tratar": q(acc_bin), "sensibilidad_tratar": q(sens)}


def pick_threshold(y_treat, p, target_sens=0.95):
    """Umbral más alto que todavía cumple la sensibilidad objetivo
    (no detectar un animal anémico puede costarle la vida)."""
    best = 0.5
    for t in np.linspace(0.05, 0.95, 91):
        s = binary_stats(y_treat, (p >= t).astype(int))["sensitivity"]
        if not np.isnan(s) and s >= target_sens:
            best = float(t)
    return best
