#!/usr/bin/env python
# coding: utf-8

# In[35]:


import os

# assert os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
# assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[36]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from pathlib import Path

# In[37]:


BATCH_SIZE = 8
IMG_HEIGHT, IMG_WIDTH = (256, 256)
PREPROCESS_SEED = 123
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "cp-{epoch:04d}.ckpt"


# In[38]:


base_data_dir = Path("..", "..", "input", "arch-recognizer-dataset").absolute()
val_data_dir = base_data_dir / "val"
test_data_dir = base_data_dir / "test"
train_data_dir = base_data_dir / "train"


# In[39]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_data_dir,
    labels="inferred",
    label_mode="int",
    seed=PREPROCESS_SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    labels="inferred",
    label_mode="int",
    seed=PREPROCESS_SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    labels="inferred",
    label_mode="int",
    seed=PREPROCESS_SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)
class_names = train_ds.class_names

train_ds.map(lambda i, _: tf.keras.applications.inception_resnet_v2.preprocess_input(i))
val_ds.map(lambda i, _: tf.keras.applications.inception_resnet_v2.preprocess_input(i))
test_ds.map(lambda i, _: tf.keras.applications.inception_resnet_v2.preprocess_input(i))

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = _ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[40]:


def restore_weights(model):
    latest_cp = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_cp:
        model.load_weights(latest_cp)
        _, restored_test_acc = model.evaluate(test_ds, verbose=2)
        print(f"Restored model test accuracy: {restored_test_acc}")
    return model


# In[41]:


def create_model():
    _model = tf.keras.models.Sequential(
        [
            # Preprocessing
            tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),
            # Augmentation
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
            ),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.3),
            # Convolution
            tf.keras.applications.InceptionResNetV2(
                include_top=False,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=len(class_names),
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(
                len(class_names), activation="softmax", name="predictions"
            ),
        ]
    )
    _model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return _model


# In[42]:


tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = restore_weights(create_model())
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=80,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(CHECKPOINT_PATH),
                verbose=1,
                save_weights_only=True,
                save_freq=BATCH_SIZE * 100,
            ),
        ],
    )


# In[ ]:


test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# In[ ]:


# Visualize training results
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]
val_loss_range = range(len(loss))

plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.plot(range(len(loss)), loss, label="Training Loss")
plt.plot(range(len(val_loss)), val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")

plt.subplot(2, 2, 2)
plt.plot(range(len(acc)), acc, label="Training Accuracy")
plt.plot(range(len(val_acc)), val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
# plt.show()
plt.savefig("loss-accuracy.jpg")

model.save("model-InceptionResNetV2-2")
model.save("model-InceptionResNetV2-2.h5")

# In[ ]:

import random

test_files = [
    os.path.join(path, filename)
    for path, _, files in os.walk(test_data_dir)
    for filename in files
    if filename.lower().endswith(".jpg")
]
img_path = Path(random.choice(test_files))

img = tf.keras.preprocessing.image.load_img(
    img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

pred_y = class_names[np.argmax(score)]
true_y = img_path.parent.stem

plt.figure(figsize=(10, 10))
plt.title(
    f"pred: {pred_y}" f"\ntrue: {true_y}" f"\nconf: {100 * np.max(score):.2f}%",
    backgroundcolor="green" if pred_y == true_y else "red",
    horizontalalignment="right",
)
plt.imshow(tf.keras.preprocessing.image.load_img(img_path))
plt.axis("off")
