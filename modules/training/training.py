import os

from numpy import ndarray

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from datetime import datetime
import tensorflow as tf
from tensorflow import keras


def train_model(
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        batch_size, epochs,
        loss, optimizer,
        metrics,
        save_best=False, save_dir="../../static/models/saved/checkpoints/",
        log_metrics=False, log_dir="../../static/training/logs/scalars/") -> tf.keras.Model:

    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    AUTOTUNE = tf.data.AUTOTUNE

    dataset = dataset.batch(batch_size, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE)

    val_dataset = val_dataset.batch(len(val_dataset), num_parallel_calls=AUTOTUNE)

    callbacks = []
    if save_best:
        save_callback = keras.callbacks.ModelCheckpoint(
            save_dir,
            save_weights_only=True,
            save_best_only=True,
            monitor="accuracy",
            verbose=1
        )
        callbacks.append(save_callback)

    if log_metrics:
        logdir = log_dir + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard_callback)


    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    model.fit(
        dataset,
        validation_data=val_dataset,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    return model

def save_model(model: keras.Model, save_path, save_weights_only=True):
    if save_weights_only:
        model.save_weights(save_path)
    else:
        model.save(save_path)





