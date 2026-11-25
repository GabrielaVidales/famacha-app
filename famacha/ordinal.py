"""Utilidades para la salida ordinal CORAL (numpy y keras)."""
import numpy as np
import keras
from keras import ops

K = 5  # grados FAMACHA


def grade_to_levels(grades, k=K):
    """grado 1..5 -> vector acumulado [y>1, y>2, y>3, y>4]."""
    g = np.asarray(grades).reshape(-1, 1)
    return (g > np.arange(1, k)).astype("float32")


def logits_to_cumprobs(logits):
    return 1.0 / (1.0 + np.exp(-np.asarray(logits)))


def cumprobs_to_classprobs(cp):
    """P(y>k) -> P(y=k). Se fuerza monotonía por seguridad numérica."""
    cp = np.minimum.accumulate(np.asarray(cp), axis=1)
    ones = np.ones((cp.shape[0], 1))
    zeros = np.zeros((cp.shape[0], 1))
    upper = np.concatenate([ones, cp], axis=1)
    lower = np.concatenate([cp, zeros], axis=1)
    return np.clip(upper - lower, 0, 1)


def predict_grade(logits):
    """Grado = 1 + número de umbrales superados (P(y>k) > 0.5)."""
    return 1 + (logits_to_cumprobs(logits) > 0.5).sum(axis=1)


def p_treat(logits, min_grade=4):
    """P(grado >= min_grade) directamente de la salida acumulada."""
    return logits_to_cumprobs(logits)[:, min_grade - 2]


# ---------- pérdidas / métricas Keras ----------
@keras.saving.register_keras_serializable(package="famacha")
def coral_loss(y_true, logits):
    return ops.mean(ops.binary_crossentropy(y_true, logits, from_logits=True), axis=-1)


@keras.saving.register_keras_serializable(package="famacha")
def ordinal_accuracy(y_true, logits):
    t = ops.sum(y_true, axis=-1)
    p = ops.sum(ops.cast(logits > 0, "float32"), axis=-1)
    return ops.cast(ops.equal(t, p), "float32")


@keras.saving.register_keras_serializable(package="famacha")
def ordinal_mae(y_true, logits):
    t = ops.sum(y_true, axis=-1)
    p = ops.sum(ops.sigmoid(logits), axis=-1)  # valor esperado suave
    return ops.abs(t - p)
