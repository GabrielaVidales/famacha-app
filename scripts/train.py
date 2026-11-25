"""Paso 2 — Entrenamiento con validación cruzada AGRUPADA POR ANIMAL + ensamble.

Uso:  python scripts/train.py --config config.yaml

Por qué agrupar por animal: si fotos del mismo animal (otro ojo, otra semana)
caen en train y en test, el modelo "reconoce al animal" y la accuracy se
infla. Es el error más común en papers con 97-99 % reportado.
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from famacha.utils import load_config, resolve, log  # noqa: E402  (silencia logs de TF)

import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import StratifiedGroupKFold

from famacha import ordinal
from famacha.data import MetaEncoder, class_weights, make_dataset, save_json
from famacha.metrics import full_report, pick_threshold
from famacha.model import build_model, unfreeze_top
from famacha.inference import FamachaEnsemble


def split_test(df, frac, seed):
    n = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(sgkf.split(df, df["famacha"], df["animal_id"]))
    return df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def fold_splits(df, k, seed):
    if k > 1:
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(sgkf.split(df, df["famacha"], df["animal_id"]))
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    return [next(sgkf.split(df, df["famacha"], df["animal_id"]))]


def compile_model(model, lr, with_pcv):
    losses = {"famacha": ordinal.coral_loss}
    metrics = {"famacha": [ordinal.ordinal_accuracy, ordinal.ordinal_mae]}
    if with_pcv:
        losses["pcv"] = "mse"
    model.compile(optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4), loss=losses, metrics=metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--drop-flagged", action="store_true", help="excluir fotos marcadas por calidad")
    args = ap.parse_args()
    cfg = load_config(resolve(args.config, os.getcwd()))
    dc, tc = cfg["data"], cfg["train"]
    seed = dc["seed"]
    keras.utils.set_random_seed(seed)

    out_dir = resolve(cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(resolve(args.config, os.getcwd()), os.path.join(out_dir, "config.yaml"))

    df = pd.read_csv(os.path.join(resolve(dc["processed_dir"]), "processed.csv"))
    if args.drop_flagged:
        df = df[~df["flag_quality"]].reset_index(drop=True)
    df["famacha"] = df["famacha"].astype(int)

    # ---------- split test por animal (se guarda para reproducibilidad) ----------
    dev, test = split_test(df, dc["test_fraction"], seed)
    assert not set(dev.animal_id) & set(test.animal_id), "¡fuga de animales entre dev y test!"
    pd.concat([dev.assign(split="dev"), test.assign(split="test")]).to_csv(os.path.join(out_dir, "splits.csv"), index=False)
    log(f"dev: {len(dev)} fotos / {dev.animal_id.nunique()} animales | test: {len(test)} / {test.animal_id.nunique()}")

    # ---------- metadatos ----------
    mcfg = cfg["metadata"]
    enc = MetaEncoder(mcfg["numeric"], mcfg["binary"]).fit(dev) if mcfg["enabled"] else None
    n_meta = enc.n_features if enc else 0
    with_pcv = cfg["model"]["pcv_head"] is True or (
        cfg["model"]["pcv_head"] == "auto" and "pcv" in df and df["pcv"].notna().mean() > 0.3)

    pcv_norm = (25.0, 7.0)
    if with_pcv:
        v = pd.to_numeric(dev["pcv"], errors="coerce")
        pcv_norm = (float(v.mean()), float(v.std() or 1.0))
    size, bs = dc["image_size"], tc["batch_size"]
    oof_logits = np.zeros((len(dev), 4), "float32")

    for f, (tr_idx, va_idx) in enumerate(fold_splits(dev, tc["folds"], seed)):
        tr, va = dev.iloc[tr_idx], dev.iloc[va_idx]
        assert not set(tr.animal_id) & set(va.animal_id)
        log(f"\n===== Fold {f + 1}: train {len(tr)} / val {len(va)} =====")
        cw = class_weights(tr["famacha"].values) if tc["class_weighting"] else None
        mtr = enc.transform(tr) if enc else None
        mva = enc.transform(va) if enc else None
        ds_tr = make_dataset(tr, size, mtr, with_pcv, cw, training=True, batch=bs, seed=seed + f, pcv_norm=pcv_norm)
        ds_va = make_dataset(va, size, mva, with_pcv, None, training=False, batch=bs, pcv_norm=pcv_norm)

        keras.backend.clear_session()
        model, backbone = build_model(cfg, n_meta, with_pcv)
        ckpt = os.path.join(out_dir, f"fold_{f}.keras")
        cbs = lambda: [
            keras.callbacks.EarlyStopping("val_loss", patience=tc["patience"], restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=max(2, tc["patience"] // 3)),
        ]

        # Fase 1: solo cabezas
        compile_model(model, tc["lr_head"], with_pcv)
        model.fit(ds_tr, validation_data=ds_va, epochs=tc["epochs_head"], callbacks=cbs(), verbose=2)
        # Fase 2: fine-tuning de la parte alta del backbone
        if tc["epochs_finetune"] > 0:
            unfreeze_top(backbone, tc["unfreeze_fraction"])
            compile_model(model, tc["lr_finetune"], with_pcv)
            model.fit(ds_tr, validation_data=ds_va, epochs=tc["epochs_finetune"], callbacks=cbs(), verbose=2)

        model.save(ckpt)
        oof_logits[va_idx] = model.predict(ds_va, verbose=0)["famacha"]

    # ---------- umbral de "tratar" elegido con predicciones out-of-fold ----------
    oof_mask = np.abs(oof_logits).sum(1) > 0
    thr = pick_threshold((dev["famacha"].values[oof_mask] >= 4).astype(int),
                         ordinal.p_treat(oof_logits[oof_mask]), tc["target_sensitivity"])
    log(f"\nUmbral 'tratar' (sensibilidad objetivo {tc['target_sensitivity']:.0%}): {thr:.2f}")

    save_json({
        "image_size": size, "n_meta": n_meta, "with_pcv": bool(with_pcv),
        "meta_encoder": enc.to_dict() if enc else None, "pcv_norm": list(pcv_norm),
        "threshold_treat": thr, "folds": tc["folds"], "backbone": cfg["model"]["backbone"],
        "input": "RGB float32 0-255, recorte cuadrado de la conjuntiva, sin normalizar",
    }, os.path.join(out_dir, "model_meta.json"))

    if oof_mask.any():
        oof_rep = full_report(dev[oof_mask].reset_index(drop=True), ordinal.predict_grade(oof_logits[oof_mask]),
                              ordinal.p_treat(oof_logits[oof_mask]), thr, dc["pcv_anemia_threshold"])
        save_json(oof_rep, os.path.join(out_dir, "report_validacion_cruzada.json"))

    # ---------- evaluación final en TEST (animales nunca vistos) ----------
    ens = FamachaEnsemble(out_dir)
    logits, pcv, spread = ens.predict_paths(test["proc_path"].tolist(), test)
    grades, pt = ordinal.predict_grade(logits), ordinal.p_treat(logits)
    rep = full_report(test, grades, pt, thr, dc["pcv_anemia_threshold"])
    if pcv is not None and test["pcv"].notna().any():
        m = test["pcv"].notna().values
        rep["pcv_mae_puntos"] = float(np.mean(np.abs(pcv[m] - test.loc[m, "pcv"].values)))
    save_json(rep, os.path.join(out_dir, "report_test.json"))
    test.assign(pred=grades, p_tratar=pt, spread=spread).to_csv(os.path.join(out_dir, "predicciones_test.csv"), index=False)

    o = rep["ordinal"]; t = rep["tratar_vs_famacha_humano"]
    log("\n================ TEST (animales no vistos) ================")
    log(f"Accuracy 5 grados : {o['accuracy_5_grados']:.3f}  IC95 {rep['ic95_bootstrap_por_animal']['accuracy_5_grados']}")
    log(f"Accuracy ±1 grado : {o['accuracy_+-1_grado']:.3f}   Kappa cuadrática: {o['kappa_cuadratica']:.3f}")
    log(f"Tratar (>=4)      : acc {t['accuracy']:.3f}  sens {t['sensitivity']:.3f}  spec {t['specificity']:.3f}")
    log(f"Reportes en {out_dir}")


if __name__ == "__main__":
    main()
