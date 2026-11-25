# data/

Pon aquí tus datasets (están en `.gitignore`: no se suben a GitHub).

Estructura esperada para cada dataset:

```
data/mi_dataset/
├── labels.csv        # una fila por foto; ver labels_template.csv
└── raw/...           # las fotos (rutas relativas a labels.csv)
```

Formas de llenarla:
1. **Registro de la página** → “Exportar dataset (.zip)” → descomprimir aquí (`famacha_dataset/`).
2. **Roboflow Universe** → `python scripts/import_roboflow.py --from-dir data/<export>`.
3. A mano, siguiendo `labels_template.csv` (obligatorias: `image_path`, `animal_id`, `famacha`).
