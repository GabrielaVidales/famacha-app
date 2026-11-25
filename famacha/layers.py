"""Capas personalizadas. Todo el preprocesamiento de color vive DENTRO del grafo,
así el modelo TFLite recibe solo la imagen recortada (0-255) y la app no tiene
que reimplementar nada en Kotlin/Dart."""
import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="famacha")
class WhitePatchBalance(layers.Layer):
    """Balance de blancos tipo white-patch.

    Estima el color de la luz con los píxeles más brillantes (en la conjuntiva
    húmeda son brillos especulares, que reflejan el color del iluminante) y
    reescala cada canal. Usa avg-pooling antes del máximo para no depender de
    un solo píxel saturado. Las ganancias se recortan para no inventar color.
    """

    def __init__(self, pool=4, max_gain=1.6, **kw):
        super().__init__(**kw)
        self.pool = pool
        self.max_gain = max_gain

    def call(self, x):  # x: (B,H,W,3) en 0-255
        pooled = ops.average_pool(x, pool_size=self.pool, strides=self.pool, padding="valid")
        illum = ops.max(pooled, axis=(1, 2), keepdims=True)          # (B,1,1,3)
        ref = ops.max(illum, axis=-1, keepdims=True)                  # canal más brillante
        gain = ops.clip(ref / (illum + 1e-3), 1.0, self.max_gain)
        return ops.clip(x * gain, 0.0, 255.0)

    def get_config(self):
        return {**super().get_config(), "pool": self.pool, "max_gain": self.max_gain}


@keras.saving.register_keras_serializable(package="famacha")
class ColorStats(layers.Layer):
    """Estadísticas de color "a la FAMACHA" sobre la región central del recorte.

    Excluye pupila/sombras (muy oscuros) y brillos especulares (muy claros y
    poco saturados). Devuelve 10 features: media y desviación de cromaticidad
    r,g,b; índice de enrojecimiento (R-G)/(R+G); saturación media; luminancia
    media; y la fracción de píxeles válidos (control de calidad).
    """

    def __init__(self, center_frac=0.7, **kw):
        super().__init__(**kw)
        self.center_frac = center_frac

    def call(self, x):
        h, w = x.shape[1], x.shape[2]
        ch, cw = int(h * self.center_frac), int(w * self.center_frac)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        x = x[:, y0:y0 + ch, x0:x0 + cw, :] / 255.0

        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        s = r + g + b + 1e-6
        mx = ops.max(x, axis=-1)
        mn = ops.min(x, axis=-1)
        sat = (mx - mn) / (mx + 1e-6)
        lum = 0.299 * r + 0.587 * g + 0.114 * b

        valid = ops.cast((lum > 0.12) & ~((lum > 0.92) & (sat < 0.12)), x.dtype)
        n = ops.sum(valid, axis=(1, 2)) + 1.0

        def wmean(v):
            return ops.sum(v * valid, axis=(1, 2)) / n

        def wstd(v, m):
            return ops.sqrt(ops.sum(ops.square(v - m[:, None, None]) * valid, axis=(1, 2)) / n + 1e-6)

        cr, cg, cb = r / s, g / s, b / s
        redness = (r - g) / (r + g + 1e-6)
        m_cr, m_cg, m_cb = wmean(cr), wmean(cg), wmean(cb)
        feats = [
            m_cr, m_cg, m_cb,
            wstd(cr, m_cr), wstd(cg, m_cg), wstd(cb, m_cb),
            wmean(redness), wmean(sat), wmean(lum),
            n / float(ch * cw),
        ]
        return ops.stack(feats, axis=-1)

    def get_config(self):
        return {**super().get_config(), "center_frac": self.center_frac}


@keras.saving.register_keras_serializable(package="famacha")
class CoralHead(layers.Layer):
    """Cabeza ordinal CORAL (Cao, Mirjalili & Raschka, 2020).

    FAMACHA es ORDINAL (1<2<3<4<5): confundir 1 con 5 es mucho peor que 3 con 4.
    CORAL predice K-1 probabilidades acumuladas P(y > k) con pesos compartidos
    y sesgos distintos, lo que garantiza predicciones consistentes (rank-monotonic).
    """

    def __init__(self, num_classes=5, **kw):
        super().__init__(**kw)
        self.num_classes = num_classes

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 1), initializer="glorot_uniform", name="w")
        init = keras.initializers.Constant([float(self.num_classes - 1 - 2 * k) / 2 for k in range(self.num_classes - 1)])
        self.b = self.add_weight(shape=(self.num_classes - 1,), initializer=init, name="b")

    def call(self, x):
        # Se repite w en K-1 columnas para que el matmul ya salga (B, K-1). Es idéntico a
        # matmul(x,w)+b, pero evita que el optimizador de TF fusione un sesgo (K-1,) sobre
        # una salida (B,1), fusión que TensorFlow.js ejecuta mal.
        w = ops.tile(self.w, [1, self.num_classes - 1])
        return ops.matmul(x, w) + self.b  # logits (B, K-1)

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes}
