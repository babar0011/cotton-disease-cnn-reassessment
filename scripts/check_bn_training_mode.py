"""Minimal, dataset-free demonstration of the BatchNorm behaviour reported in the paper.

A backbone containing BatchNormalization is nested in a functional model and called with
training=False, as the Keras transfer-learning guide recommends. After the backbone is
unfrozen, one fit() step is run. If the BatchNorm layer stayed in inference mode, its moving
mean cannot change. In Keras 3.13.2 it does change, because Functional.call passes the
outer training=True into nested layers (keras/src/models/functional.py, operation_fn) and
overrides the stored training=False. Freezing the BatchNorm layer (trainable=False) forces
inference mode and is the fix used in the v3 notebook.

Run:  python scripts/check_bn_training_mode.py      (needs tensorflow / keras only)
"""
import numpy as np
import keras
from keras import layers


def build():
    keras.utils.set_random_seed(0)
    inp = keras.Input((8, 8, 3))
    x = layers.Conv2D(4, 3, padding="same")(inp)
    x = layers.BatchNormalization(name="bn")(x)
    backbone = keras.Model(inp, layers.ReLU()(x), name="backbone")
    outer_in = keras.Input((8, 8, 3))
    y = backbone(outer_in, training=False)              # recommended pattern
    y = layers.GlobalAveragePooling2D()(y)
    model = keras.Model(outer_in, layers.Dense(2, activation="softmax")(y))
    return model, backbone


x = np.random.default_rng(0).normal(3.0, 2.0, (16, 8, 8, 3)).astype("float32")
y = keras.utils.to_categorical(np.arange(16) % 2, 2)
print("keras", keras.__version__)
for freeze_bn in (False, True):
    model, backbone = build()
    backbone.trainable = True                            # unfreeze for fine-tuning
    if freeze_bn:
        backbone.get_layer("bn").trainable = False
    bn = backbone.get_layer("bn")
    before = keras.ops.convert_to_numpy(bn.moving_mean)
    model.compile("adam", "categorical_crossentropy")
    model.fit(x, y, batch_size=16, epochs=1, verbose=0)
    delta = float(np.abs(keras.ops.convert_to_numpy(bn.moving_mean) - before).max())
    mode = "TRAINING mode (statistics updated)" if delta > 0 else "inference mode"
    print(f"BatchNorm trainable={not freeze_bn!s:5}: max |change in moving mean| = {delta:.3e} -> {mode}")
