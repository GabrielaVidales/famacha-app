"""Arquitectura multi-entrada:

    imagen 0-255 ──► WhitePatchBalance ──┬─► Backbone CNN (ImageNet) ─► feature_map ─► GAP ─┐
                                         └─► ColorStats (10 features de color) ─► Dense ───┤
    metadatos clínicos (Five Point Check) ─────────────────────────────► Dense ───────────┤
                                                                                           ▼
                                                     concat ─► Dense ─► CORAL (FAMACHA 1-5)
                                                                     └► PCV % (opcional)
"""
import keras
from keras import layers

from .layers import WhitePatchBalance, ColorStats, CoralHead

BACKBONES = {
    "efficientnetv2b0": lambda w, s: keras.applications.EfficientNetV2B0(
        include_top=False, weights=w, input_shape=(s, s, 3), include_preprocessing=True),
    "mobilenetv3large": lambda w, s: keras.applications.MobileNetV3Large(
        include_top=False, weights=w, input_shape=(s, s, 3), include_preprocessing=True),
}


def build_model(cfg, n_meta=0, with_pcv=False):
    mc = cfg["model"]
    size = cfg["data"]["image_size"]
    weights = None if str(mc["weights"]).lower() == "none" else mc["weights"]

    img_in = keras.Input((size, size, 3), name="image")
    inputs = {"image": img_in}
    x = WhitePatchBalance(name="white_balance")(img_in) if mc["white_balance"] else img_in

    backbone = BACKBONES[mc["backbone"]](weights, size)
    backbone.trainable = False
    fmap = layers.Activation("linear", name="feature_map")(backbone(x, training=False))
    feats = [layers.Dropout(mc["dropout"])(layers.GlobalAveragePooling2D()(fmap))]

    if mc["color_features"]:
        c = ColorStats(name="color_stats")(x)
        c = layers.BatchNormalization()(c)
        feats.append(layers.Dense(32, activation="relu")(c))

    if n_meta > 0:
        meta_in = keras.Input((n_meta,), name="meta")
        inputs["meta"] = meta_in
        feats.append(layers.Dense(32, activation="relu")(meta_in))

    h = layers.Concatenate()(feats) if len(feats) > 1 else feats[0]
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(mc["dropout"])(h)

    outputs = {"famacha": CoralHead(5, name="famacha")(h)}
    if with_pcv:
        outputs["pcv"] = layers.Dense(1, name="pcv")(h)

    model = keras.Model(inputs, outputs, name="famacha_net")
    return model, backbone


def unfreeze_top(backbone, fraction):
    """Descongela la fracción final del backbone. Las BatchNorm se quedan
    congeladas: con lotes pequeños sus estadísticas se arruinan."""
    backbone.trainable = True
    cut = int(len(backbone.layers) * (1 - fraction))
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= cut and not isinstance(layer, layers.BatchNormalization)
