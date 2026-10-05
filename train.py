"""
Training script for impact sound-based material classification.

Input:
    Spectrogram PNG images stored in DATA_DIR.
    File names should follow the format:
        <class_id>_<sample_id>.png

Example:
    0_1.png
    0_2.png
    1_1.png

The class ID is extracted from the first field of each filename.

Environment used in the study:
    Python 3.11
    TensorFlow 2.20.0
    Keras 3.13.2
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    concatenate,
    Input,
    GlobalAveragePooling2D,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "./data0310_G1"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
VALIDATION_RATIO = 0.30
RANDOM_STATE = 42

MODEL_PATH = "best_model.h5"

TARGET_NAMES = [
    "Plastic",
    "Metal",
    "Wood",
    "Glass",
    "Concrete",
    "Carpet",
    "Foam",
    "Paper",
    "Textile",
    "Soil",
]


# ============================================================
# Data loading
# ============================================================

def load_data(directory):
    """Load spectrogram PNG images and extract class labels."""

    images = []
    labels = []

    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".png"):
            continue

        filepath = os.path.join(directory, filename)

        image = tf.keras.preprocessing.image.load_img(
            filepath,
            target_size=IMAGE_SIZE,
        )
        image = tf.keras.preprocessing.image.img_to_array(image)

        # Class ID is stored before the first underscore.
        label = int(filename.split("_")[0])

        images.append(image)
        labels.append(label)

    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    return images, labels


# ============================================================
# Inception model
# ============================================================

def inception_module(x, filters):
    """Compact Inception-style module with four parallel branches."""

    branch_1x1 = Conv2D(
        filters,
        (1, 1),
        padding="same",
        activation="relu",
    )(x)

    branch_3x3 = Conv2D(
        filters,
        (1, 1),
        padding="same",
        activation="relu",
    )(x)
    branch_3x3 = Conv2D(
        filters,
        (3, 3),
        padding="same",
        activation="relu",
    )(branch_3x3)

    branch_5x5 = Conv2D(
        filters,
        (1, 1),
        padding="same",
        activation="relu",
    )(x)
    branch_5x5 = Conv2D(
        filters,
        (5, 5),
        padding="same",
        activation="relu",
    )(branch_5x5)

    branch_pool = MaxPooling2D(
        (3, 3),
        strides=(1, 1),
        padding="same",
    )(x)
    branch_pool = Conv2D(
        filters,
        (1, 1),
        padding="same",
        activation="relu",
    )(branch_pool)

    return concatenate(
        [
            branch_1x1,
            branch_3x3,
            branch_5x5,
            branch_pool,
        ],
        axis=-1,
    )


def build_model(num_classes):
    """Build the compact Inception-based classifier."""

    inputs = Input(shape=(224, 224, 3))

    x = inception_module(inputs, 64)
    x = MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        padding="same",
    )(x)

    x = inception_module(x, 128)
    x = MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        padding="same",
    )(x)

    x = inception_module(x, 128)
    x = MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        padding="same",
    )(x)

    x = inception_module(x, 256)
    x = MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        padding="same",
    )(x)

    x = inception_module(x, 256)

    x = GlobalAveragePooling2D()(x)

    outputs = Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
    )

    return model


# ============================================================
# Training
# ============================================================

def plot_training_history(history):
    """Save training and validation accuracy/loss curves."""

    epochs = range(len(history.history["accuracy"]))

    plt.figure(figsize=(14, 6), dpi=300)

    plt.subplot(1, 2, 1)
    plt.plot(
        epochs,
        history.history["accuracy"],
        label="Training accuracy",
    )
    plt.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        epochs,
        history.history["loss"],
        label="Training loss",
    )
    plt.plot(
        epochs,
        history.history["val_loss"],
        label="Validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        "training_results.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def save_training_history(history):
    """Save numerical training history."""

    np.save(
        "loss.npy",
        np.asarray(history.history["loss"]),
    )
    np.save(
        "val_loss.npy",
        np.asarray(history.history["val_loss"]),
    )
    np.save(
        "accuracy.npy",
        np.asarray(history.history["accuracy"]),
    )
    np.save(
        "val_accuracy.npy",
        np.asarray(history.history["val_accuracy"]),
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, X_val, y_val):
    """Calculate validation metrics and save outputs."""

    y_pred_prob = model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    y_pred = np.argmax(y_pred_prob, axis=1)

    overall_accuracy = accuracy_score(
        y_val,
        y_pred,
    )

    macro_accuracy = balanced_accuracy_score(
        y_val,
        y_pred,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_val,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    print("\nOverall accuracy:", overall_accuracy)
    print("Macro accuracy:", macro_accuracy)
    print("Macro precision:", macro_precision)
    print("Macro recall:", macro_recall)
    print("Macro F1-score:", macro_f1)

    # Classification report
    report = classification_report(
        y_val,
        y_pred,
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        "classification_report.csv",
        encoding="utf_8_sig",
    )

    # Summary metrics
    summary = pd.DataFrame(
        {
            "Metric": [
                "Overall Accuracy",
                "Macro Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1-score",
            ],
            "Value (%)": [
                overall_accuracy * 100,
                macro_accuracy * 100,
                macro_precision * 100,
                macro_recall * 100,
                macro_f1 * 100,
            ],
        }
    )

    summary.to_csv(
        "summary_metrics.csv",
        index=False,
        encoding="utf_8_sig",
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_val,
        y_pred,
    )

    plt.figure(figsize=(12, 10), dpi=300)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(
        "confusion_matrix.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    # Load data
    images, labels = load_data(DATA_DIR)

    print(f"Number of samples: {len(images)}")
    print(f"Number of classes: {len(np.unique(labels))}")

    # 70% training and 30% intra-object validation
    X_train, X_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=VALIDATION_RATIO,
        random_state=RANDOM_STATE,
    )

    # Normalize image values to [0, 1]
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
    )

    val_generator = val_datagen.flow(
        X_val,
        y_val,
        batch_size=BATCH_SIZE,
    )

    # Build model
    model = build_model(
        num_classes=len(np.unique(labels))
    )

    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        min_lr=0.0001,
        verbose=1,
    )

    # Train model
    history = model.fit(
        train_generator,
        epochs=MAX_EPOCHS,
        validation_data=val_generator,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr,
        ],
    )

    save_training_history(history)
    plot_training_history(history)

    # Load best validation checkpoint
    model.load_weights(MODEL_PATH)

    # Evaluate validation set
    evaluate_model(
        model,
        X_val,
        y_val,
    )


if __name__ == "__main__":
    main()