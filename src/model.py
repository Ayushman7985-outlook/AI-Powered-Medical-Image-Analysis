"""A compact educational binary CNN; TensorFlow is imported only when needed."""

from .config import DEFAULT_SETTINGS, Settings


def build_model(settings: Settings = DEFAULT_SETTINGS):
    """Build a sigmoid CNN where score >= 0.5 maps to PNEUMONIA (label 1)."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to build the CNN. Install requirements.txt.") from exc

    tf.keras.utils.set_random_seed(settings.seed)
    inputs = tf.keras.Input(shape=(settings.image_height, settings.image_width, settings.channels), name="xray")
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_score")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="educational_chest_xray_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model
