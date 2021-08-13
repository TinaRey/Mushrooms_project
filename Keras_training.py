import os
import logging

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import hydra
from omegaconf import DictConfig


AUTOTUNE = tf.data.AUTOTUNE
# helper functions


def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)


def decode_img(img, target_img_size):
    img = tf.io.decode_jpeg(img, channels=3)
    return smart_resize(img, target_img_size)


def process_path(file_path, class_names, target_img_size):
    label = get_label(file_path, class_names)
    img = tf.io.read_file(file_path)
    img = decode_img(img, target_img_size)
    return img, label


def simple_model(nb_classes, input_size):
    normalization_layer = layers.experimental.preprocessing.Rescaling(
        1./255, input_shape=(input_size[0], input_size[1], 3))
    model = Sequential([
        normalization_layer,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(nb_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    return model


@hydra.main(config_path="config", config_name="config.yaml")
def main_func(cfg: DictConfig):
    log = logging.getLogger(__name__)
    log.debug("Debug level message")
    log.info("Info level message")
    log.warning("Warning level message")

    orig_cwd = hydra.utils.get_original_cwd()
    data_folder = cfg.dataset.data_folder
    data_dir = os.path.join(orig_cwd, data_folder)
    class_names = os.listdir(data_dir)
    target_img_size = (cfg.dataset.target_img_size, cfg.dataset.target_img_size)
    batch_size = cfg.batch_size
    val_size = cfg.val_size
    test_size = cfg.test_size

    list_ds = tf.data.Dataset.list_files(data_dir+"/*/*.jpg", shuffle=True)
    image_count = list_ds.cardinality().numpy()
    val_size = int(image_count * val_size)
    test_size = int(image_count * test_size)

    train_ds = list_ds.skip(val_size+test_size)
    val_ds = list_ds.take(val_size)
    test_ds = list_ds.take(test_size)

    train_ds = train_ds.map(lambda x: process_path(
        x, class_names, target_img_size), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x: process_path(
        x, class_names, target_img_size), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x: process_path(
        x, class_names, target_img_size), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(test_ds.cardinality().numpy())

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    nb_classes = len(class_names)
    input_size = target_img_size
    model = simple_model(nb_classes, input_size)

    epochs = 10
    checkpoint_path = "ckpt_files/cp.ckpt"

    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  save_best_only=True,
                                  verbose=1)

    log_dir = "logs/fit/"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[cp_callback, tensorboard_callback]
    )

    # Evaluate the model
    loss, acc = model.evaluate(test_ds, verbose=2)
    print("Trained model, accuracy: {:5.2f}%".format(100 * acc))    


if __name__ == "__main__":
    main_func()