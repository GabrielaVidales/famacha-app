"""Importa un dataset público de Roboflow Universe al formato de FAMACHA-Net.

Acepta exportaciones de Roboflow en formato:
  - YOLOv8 / YOLOv5 (detección o segmentación): data.yaml + {train,valid,test}/{images,labels}
  - Folder (clasificación): {train,valid,test}/<clase>/*.jpg

Dos formas de uso:

  # A) Descarga directa con tu API key gratuita (https://app.roboflow.com → Settings → API Keys)
  pip install roboflow
  python scripts/import_roboflow.py --api-key TU_KEY --workspace WS --project PROYECTO --version 1 --format yolov8

  # B) Ya descargaste el .zip desde la web (Download Dataset → YOLOv8 o Folder) y lo descomprimiste
  python scripts/import_roboflow.py --from-dir ruta/al/dataset

Las clases se convierten a grado FAMACHA leyendo el dígito 1-5 del nombre
("Score_3", "famacha-4", "F2"...). Si tus clases se llaman distinto, usa --map:
  --map "rojo=1,rojo_rosa=2,rosa=3,rosa_blanco=4,blanco=5"
Las clases sin grado (por ejemplo "eye" o "anemic") se omiten y se reportan.

Salida: data/<nombre>/labels.csv (+ las imágenes donde ya estaban). Después:
  en config.yaml →  labels_csv: data/<nombre>/labels.csv
  python run_pipeline.py
"""
import argparse
import glob
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def group_key(path):
    """Roboflow nombra las copias aumentadas como  <original>_jpg.rf.<hash>.jpg.
    Agrupamos por <original> para que una foto y sus aumentaciones nunca queden
    repartidas entre entrenamiento y prueba (fuga de datos)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"(_(jpe?g|png|webp|bmp))?\.rf\.[0-9a-zA-Z]+$", "", stem, flags=re.I)


def parse_map(s):
    if not s:
        return {}
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k.strip().lower()] = int(v)
    return out


def class_to_grade(name, mapping):
    n = str(name).strip().lower()
    if n in mapping:
        return mapping[n]
    digits = re.findall(r"(?<!\d)([1-5])(?!\d)", n)
    return int(digits[0]) if len(set(digits)) == 1 else None


def load_yaml_names(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return names


def convert_yolo(root, mapping):
    yml = (glob.glob(os.path.join(root, "data.yaml")) + glob.glob(os.path.join(root, "*.yaml")))[0]
    names = load_yaml_names(yml)
    rows, skipped = [], Counter()
    for lbl in glob.glob(os.path.join(root, "**", "labels", "*.txt"), recursive=True):
        img_dir = os.path.join(os.path.dirname(os.path.dirname(lbl)), "images")
        stem = os.path.splitext(os.path.basename(lbl))[0]
        img = next((os.path.join(img_dir, stem + e) for e in IMG_EXT if os.path.exists(os.path.join(img_dir, stem + e))), None)
        if img is None:
            continue
        w, h = Image.open(img).size
        split = os.path.basename(os.path.dirname(os.path.dirname(lbl)))
        for line in open(lbl, encoding="utf-8").read().strip().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cls = names[int(p[0])]
            g = class_to_grade(cls, mapping)
            if g is None:
                skipped[cls] += 1
                continue
            v = list(map(float, p[1:]))
            if len(v) == 4:                      # caja: cx cy w h
                cx, cy, bw, bh = v
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            else:                                # polígono (segmentación): x1 y1 x2 y2 ...
                xs, ys = v[0::2], v[1::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            rows.append({"image_path": os.path.relpath(img, root), "animal_id": group_key(img), "famacha": g,
                         "x1": round(x1 * w), "y1": round(y1 * h), "x2": round(x2 * w), "y2": round(y2 * h),
                         "source_split": split, "source_class": cls})
    return rows, skipped


def convert_folder(root, mapping):
    rows, skipped = [], Counter()
    for img in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not img.lower().endswith(IMG_EXT):
            continue
        cls = os.path.basename(os.path.dirname(img))
        g = class_to_grade(cls, mapping)
        if g is None:
            skipped[cls] += 1
            continue
        split = os.path.basename(os.path.dirname(os.path.dirname(img)))
        rows.append({"image_path": os.path.relpath(img, root), "animal_id": group_key(img), "famacha": g,
                     "source_split": split, "source_class": cls})
    return rows, skipped


def download(args):
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("Instala el cliente:  pip install roboflow")
    dest = os.path.join(ROOT, "data", f"roboflow_{args.project}_v{args.version}")
    rf = Roboflow(api_key=args.api_key)
    rf.workspace(args.workspace).project(args.project).version(int(args.version)).download(args.format, location=dest)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-dir", help="carpeta de un export de Roboflow ya descomprimido")
    ap.add_argument("--api-key"); ap.add_argument("--workspace"); ap.add_argument("--project"); ap.add_argument("--version", default="1")
    ap.add_argument("--format", default="yolov8", choices=["yolov8", "yolov5", "folder"])
    ap.add_argument("--map", help='nombres de clase → grado, p. ej. "rojo=1,rosa=3"')
    ap.add_argument("--species", choices=["cabra", "borrego"], help="especie, si el dataset es de una sola")
    args = ap.parse_args()

    if args.from_dir:
        root = os.path.abspath(args.from_dir)
    elif args.api_key and args.workspace and args.project:
        root = download(args)
    else:
        ap.error("usa --from-dir, o --api-key + --workspace + --project")

    mapping = parse_map(args.map)
    is_yolo = bool(glob.glob(os.path.join(root, "*.yaml")))
    rows, skipped = (convert_yolo if is_yolo else convert_folder)(root, mapping)
    if skipped:
        print("Clases omitidas (no son un grado 1-5; usa --map si sí lo son):", dict(skipped))
    if not rows:
        sys.exit("No encontré imágenes con grado FAMACHA. Revisa los nombres de clase y usa --map.")

    df = pd.DataFrame(rows)
    if args.species:
        df["is_goat"] = int(args.species == "cabra")
    out = os.path.join(root, "labels.csv")
    df.to_csv(out, index=False)

    rel = os.path.relpath(out, ROOT)
    print(f"\n{len(df)} ejemplos · {df.animal_id.nunique()} fotos originales (grupos) · formato {'YOLO' if is_yolo else 'folder'}")
    print("Por grado:\n" + df.famacha.value_counts().sort_index().to_string())
    faltan = sorted(set(range(1, 6)) - set(df.famacha))
    if faltan:
        print(f"[aviso] No hay ejemplos de los grados {faltan}: el modelo no podrá predecirlos bien.")
    print("[aviso] Roboflow no trae el ID del animal: se agrupa por foto original. Si varias fotos son del\n"
          "        mismo animal, la precisión en test saldrá optimista.")
    print(f"\nListo: {rel}\nSiguiente paso:  python run_pipeline.py --labels {rel}")


if __name__ == "__main__":
    main()
