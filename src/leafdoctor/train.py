"""Training pipeline: MobileNetV2 transfer learning on PlantVillage.

Ported from LeafDoctor.ipynb with the reproducibility gaps closed — the notebook
seeded nothing, so neither the results nor the train/val partition itself could
be reproduced, and it saved whatever the last epoch produced rather than the
best one.

Usage:
    python -m leafdoctor.train --epochs 10

Note: this is slow on CPU (hours per epoch over 16.5k images). Run it on a GPU.
"""

import argparse
import json
import random

import numpy as np

from .config import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    DATASET_DIR,
    DROPOUT,
    EPOCHS,
    IMG_SIZE,
    MODEL_PATH,
    SEED,
    VALIDATION_SPLIT,
)


def set_seeds(seed=SEED):
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_generators(data_dir, batch_size, seed=SEED):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=VALIDATION_SPLIT)
    common = dict(
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        seed=seed,  # pins the train/val partition; the notebook left this unset
    )
    train_gen = datagen.flow_from_directory(data_dir, subset="training", **common)
    val_gen = datagen.flow_from_directory(data_dir, subset="validation", **common)
    return train_gen, val_gen


def build_model(num_classes):
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    base = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False  # feature extraction only; no fine-tuning stage yet

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(DROPOUT)(x)
    preds = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=preds)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data-dir", default=None, help="defaults to data/PlantVillage")
    parser.add_argument("--output", default=str(MODEL_PATH))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    import tensorflow as tf

    set_seeds(args.seed)

    data_dir = args.data_dir
    if data_dir is None:
        from .data import ensure_dataset

        data_dir = ensure_dataset() if not DATASET_DIR.exists() else DATASET_DIR

    train_gen, val_gen = build_generators(data_dir, args.batch_size, args.seed)

    # The .h5 stores no label mapping, so persist the index order beside it.
    class_names = list(train_gen.class_indices.keys())
    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2) + "\n")
    print(f"{len(class_names)} classes -> {CLASS_NAMES_PATH}")

    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        # Keep the best epoch, not merely the last one.
        tf.keras.callbacks.ModelCheckpoint(
            args.output, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"\nValidation loss {loss:.4f} | validation accuracy {acc:.4f}")
    print(f"Best weights saved to {args.output}")
    return history


if __name__ == "__main__":
    main()
