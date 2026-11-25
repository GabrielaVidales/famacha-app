import os
import sys
import yaml

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RECOMMENDATION = {
    1: ("verde", "Óptimo. No requiere desparasitación."),
    2: ("verde", "Aceptable. No requiere desparasitación."),
    3: ("amarillo", "Dudoso. Revisar condición corporal, edema submandibular y diarrea; tratar si hay otros signos."),
    4: ("rojo", "Anémico. Desparasitar (tratamiento selectivo)."),
    5: ("rojo", "Anemia severa. Desparasitar y consultar al veterinario de inmediato."),
}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path, base=ROOT):
    return path if os.path.isabs(path) else os.path.join(base, path)


def log(*a):
    print(*a, file=sys.stderr, flush=True)
