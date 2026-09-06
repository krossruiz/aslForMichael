"""
Trains the tiny ASL fingerspelling classifier used by the Android app and exports
it as a fully-integer-quantized TFLite model sized for the Note 2's Cortex-A9 CPU.

Not run automatically as part of this project -- you need a dataset and a machine
with TensorFlow installed. Recommended dataset: Kaggle "ASL Alphabet"
(https://www.kaggle.com/datasets/grassknoted/asl-alphabet), which has A-Z + space
+ nothing/blank folders that map directly onto SignClassifier.LABELS.

Usage:
    pip install tensorflow==2.13.*
    python train_asl_model.py --data_dir /path/to/asl_alphabet_train --out asl_alphabet.tflite
    cp asl_alphabet.tflite ../app/src/main/assets/

Architecture notes (why it's this small):
  - Input is 96x96x3, matching SignClassifier.INPUT_SIZE. Bigger inputs cost
    roughly O(n^2) more CPU on a device with no GPU compute path.
  - A handful of depthwise-separable conv blocks (MobileNet-style), not a full
    MobileNetV2 -- we don't need ImageNet-scale capacity for 28 classes, and
    fewer params means faster CPU inference and a smaller APK.
  - Full integer (int8) post-training quantization, so the on-device Interpreter
    never touches float ops. This is what actually makes it "GPU-free but still
    fast" on old ARM cores; XNNPACK's int8 kernels are the win, not any GPU delegate.
"""

import argparse
import pathlib

import numpy as np
import tensorflow as tf

INPUT_SIZE = 96
LABELS = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "SPACE","BLANK",
]


def build_model(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    def ds_block(x, filters, stride):
        x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(6.0)(x)
        x = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(x)  # 96->48
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = ds_block(x, 32, stride=2)   # 48->24
    x = ds_block(x, 64, stride=2)   # 24->12
    x = ds_block(x, 96, stride=2)   # 12->6
    x = ds_block(x, 128, stride=1)  # 6->6

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


def representative_dataset(train_ds):
    def gen():
        for images, _ in train_ds.take(200):
            for i in range(images.shape[0]):
                yield [tf.expand_dims(images[i], 0)]
    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="folder with one subfolder per label")
    parser.add_argument("--out", default="asl_alphabet.tflite")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.1, subset="training", seed=1337,
        image_size=(INPUT_SIZE, INPUT_SIZE), batch_size=args.batch_size, class_names=LABELS,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.1, subset="validation", seed=1337,
        image_size=(INPUT_SIZE, INPUT_SIZE), batch_size=args.batch_size, class_names=LABELS,
    )

    model = build_model(len(LABELS))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(train_ds)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    out_path = pathlib.Path(args.out)
    out_path.write_bytes(tflite_model)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
