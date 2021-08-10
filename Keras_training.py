import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.models import Sequential
import numpy as np


AUTOTUNE = tf.data.AUTOTUNE
# helper functions

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  one_hot = parts[-2] == class_names
  return tf.argmax(one_hot)

def decode_img(img):
  img = tf.io.decode_jpeg(img, channels=3)
  return smart_resize(img, img_size)

def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


class_names = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']
data_dir = "./Mushrooms"
img_size = (128, 128)
batch_size = 64
list_ds = tf.data.Dataset.list_files(data_dir+"/*/*.jpg", shuffle=True)
image_count = list_ds.cardinality().numpy()

val_size = int(image_count * 0.2)
test_size = int(image_count * 0.01)
train_ds = list_ds.skip(val_size+test_size)
val_ds = list_ds.take(val_size)
test_ds = list_ds.take(test_size)


train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
test_ds = test_ds.batch(test_ds.cardinality().numpy())

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


normalization_layer = layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_size[0], img_size[1], 3))

num_classes = len(class_names)

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
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print(test_ds.cardinality().numpy())

nb_imgs = 8

plt.figure(figsize=(nb_imgs, nb_imgs))
for data, labels in test_ds.take(nb_imgs):
  predictions = model.predict(data)
  for i in range(nb_imgs):
    score = tf.nn.softmax(predictions[i])
    ax = plt.subplot(int(np.ceil(nb_imgs/2)), int(np.ceil(nb_imgs/2)), i + 1)
    plt.imshow(data[i].numpy().astype("uint8"))
    plt.title("True class: {} Class predicted: {} with {:.2f}".format(class_names[labels[i]], class_names[np.argmax(score)], 100 * np.max(score)))
    plt.axis("off")
plt.show()
