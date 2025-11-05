import matplotlib.pyplot as plt

# import os
import tensorflow as tf
import wandb
from tensorflow import keras
from wandb.integration.keras import WandbMetricsLogger

# Suppress TensorFlow platform/cloud warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info and warning messages
# tf.get_logger().setLevel('ERROR')  # Only show errors

# Sweep config for selecting experiment
sweep_config = {
    "method": "grid",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "batch_size": {"values": [8]},
        "learning_rate": {"values": [0.0001]},
        "hidden_nodes": {"values": [128]},
        "img_size": {"values": [16]},
        "epochs": {"values": [10]},
        "experiment": {"values": ["dropout_only", "batchnorm_only", "full"]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="5-flowers-experiments")


def train():
    with wandb.init() as run:
        config = wandb.config

        IMG_HEIGHT = config.img_size
        IMG_WIDTH = config.img_size
        IMG_CHANNELS = 3
        CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

        def read_and_decode(filename, resize_dims):
            img_bytes = tf.io.read_file(filename)
            img = tf.image.decode_jpeg(img_bytes, channels=IMG_CHANNELS)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, resize_dims)
            return img

        def parse_csvline(csv_line):
            record_default = ["", ""]
            filename, label_string = tf.io.decode_csv(csv_line, record_default)
            img = read_and_decode(filename, [IMG_HEIGHT, IMG_WIDTH])
            label = tf.where(tf.equal(CLASS_NAMES, label_string))[0, 0]
            return img, label

        # Datasets
        train_dataset = (
            tf.data.TextLineDataset(
                "gs://cloud-ml-data/img/flower_photos/train_set.csv"
            )
            .map(parse_csvline, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        eval_dataset = (
            tf.data.TextLineDataset("gs://cloud-ml-data/img/flower_photos/eval_set.csv")
            .map(parse_csvline, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Build model
        model = keras.Sequential()
        model.add(
            keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        )

        if config.experiment == "batchnorm_only":
            model.add(keras.layers.Dense(config.hidden_nodes, use_bias=False))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation("relu"))

        elif config.experiment == "dropout_only":
            model.add(keras.layers.Dense(config.hidden_nodes, activation="relu"))
            model.add(keras.layers.Dropout(0.5))

        elif config.experiment == "full":
            model.add(
                keras.layers.Dense(
                    config.hidden_nodes,
                    kernel_regularizer=keras.regularizers.l2(0.01),
                    use_bias=False,
                )
            )
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation("relu"))
            model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Dense(len(CLASS_NAMES), activation="softmax"))

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        # Train
        callbacks = [WandbMetricsLogger(log_freq=5)]
        if config.experiment == "full":
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                )
            )

        model.fit(
            train_dataset,
            validation_data=eval_dataset,
            epochs=config.epochs,
            callbacks=callbacks,
        )


wandb.agent(sweep_id, function=train)

# To run this script, ensure you have wandb installed and configured.
# You can start the sweep with: python 5_flower_dataset_Neural_Network_regularization.py
# Make sure to replace the dataset paths with your own if necessary.
# Also, ensure you have access to Google Cloud Storage if using the provided dataset paths.
