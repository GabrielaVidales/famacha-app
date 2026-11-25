"""Un solo comando: preparar → entrenar → evaluar → Grad-CAM → exportar a la página web.

  python run_pipeline.py --labels data/mi_dataset/labels.csv            # completo (5 folds)
  python run_pipeline.py --labels data/mi_dataset/labels.csv --quick    # rápido para CPU (1 fold, menos épocas)
  python run_pipeline.py --demo                                         # prueba con datos sintéticos (NO es un modelo real)

Al terminar:  python serve.py   → la página abre con la red cargada.
"""
import argparse
import datetime as dt
import os
import subprocess
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
META_COLS = ["bcs", "dag_score", "age_months", "days_since_deworm", "bottle_jaw",
             "nasal_discharge", "diarrhea", "is_goat", "rainy_season"]


def run(step, *args):
    print(f"\n━━━ {step} ━━━", flush=True)
    r = subprocess.run([PY, *args], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"\n✗ Falló: {step}. Revisa el mensaje de arriba.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="CSV de etiquetas (por defecto, el de config.yaml)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true", help="1 fold y menos épocas (para CPU)")
    ap.add_argument("--demo", action="store_true", help="genera datos sintéticos y corre todo (solo para probar)")
    ap.add_argument("--ensemble-web", action="store_true", help="exportar a la web el ensamble completo (más pesado)")
    ap.add_argument("--name", help="nombre del experimento (carpeta en runs/)")
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.demo:
        run("Datos sintéticos de prueba", "scripts/make_synthetic.py", "--out", "data/synth", "--animals", "40", "--per-animal", "6")
        args.labels = "data/synth/labels.csv"
        args.quick = True
    labels = args.labels or cfg["data"]["labels_csv"]
    labels_abs = labels if os.path.isabs(labels) else os.path.join(ROOT, labels)
    if not os.path.exists(labels_abs):
        sys.exit(f"No existe {labels}. Crea tu CSV (ver data/labels_template.csv) o importa uno con scripts/import_roboflow.py")

    df = pd.read_csv(labels_abs)
    name = args.name or ("demo" if args.demo else dt.datetime.now().strftime("exp_%Y%m%d_%H%M"))
    cfg["data"]["labels_csv"] = os.path.relpath(labels_abs, ROOT)
    cfg["data"]["processed_dir"] = os.path.join(os.path.relpath(os.path.dirname(labels_abs), ROOT), "processed")
    cfg["output_dir"] = os.path.join("runs", name)

    # Si el CSV no trae datos clínicos, la rama de metadatos solo añadiría ruido.
    present = [c for c in META_COLS if c in df and df[c].notna().any()]
    if cfg["metadata"]["enabled"] and len(present) < 2:
        print(f"[info] El CSV no trae datos clínicos ({present or 'ninguno'}): se desactiva la rama de metadatos.")
        cfg["metadata"]["enabled"] = False

    n_groups = df["animal_id"].nunique()
    if args.quick:
        cfg["train"].update({"folds": 1, "epochs_head": 12, "epochs_finetune": 6, "patience": 4})
    elif n_groups < 25 and cfg["train"]["folds"] > 1:
        print(f"[info] Solo {n_groups} animales/grupos: uso 1 fold en lugar de {cfg['train']['folds']}.")
        cfg["train"]["folds"] = 1
    if n_groups < 10:
        sys.exit(f"Solo hay {n_groups} animales/grupos distintos. Con eso no se puede separar entrenamiento y prueba "
                 "de forma honesta; junta más fotos (usa el Registro de la página) o importa un dataset público.")

    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    cfg_path = os.path.join(ROOT, "runs", f"{name}.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"Experimento: runs/{name}  ·  {len(df)} fotos  ·  {n_groups} animales/grupos  ·  folds={cfg['train']['folds']}")

    run("1/5 Preparar imágenes", "scripts/prepare.py", "--config", cfg_path)
    run("2/5 Entrenar", "scripts/train.py", "--config", cfg_path)
    run("3/5 Evaluar", "scripts/evaluate.py", "--run", cfg["output_dir"])
    run("4/5 Grad-CAM", "scripts/gradcam.py", "--run", cfg["output_dir"])
    run("5/5 Exportar a la página", "scripts/export_tfjs.py", "--run", cfg["output_dir"], "--to-web",
        *([] if args.ensemble_web else ["--single"]))

    if args.demo:
        import json
        info_p = os.path.join(ROOT, "web", "model", "model_info.json")
        with open(info_p, encoding="utf-8") as f:
            info = json.load(f)
        info["demo"] = True
        with open(info_p, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Listo. Reportes y figuras en runs/{name}/")
    if args.demo:
        print("⚠ Modelo DEMO entrenado con imágenes sintéticas: sirve para probar la integración, NO para diagnosticar.")
    print("Abre la página con:  python serve.py")


if __name__ == "__main__":
    main()
