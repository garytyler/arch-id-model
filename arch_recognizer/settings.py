import logging
from pathlib import Path
from typing import Callable, Tuple

import tensorflow as tf

APP_NAME: str = "arch-recognizer"
TIMESTAMP_FORMAT: str = r"%Y-%m-%d-%H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO
SEED = 123456
BASE_DIR: Path = Path(__file__).parent.parent.absolute()

# Metrics
WEIGHTS = ["imagenet", "none"]


class BaseCNN:
    def __init__(
        self,
        name: str,
        base_model: tf.keras.Model,
        preprocess: Callable,
        image_size: Tuple[int, int],
        batch_size: int,
    ):
        self.base_model = base_model
        self.preprocess = preprocess
        self.image_size = image_size
        self.batch_size = batch_size


BASE_CNNS = {
    tf.keras.applications.VGG19.__name__: BaseCNN(
        name=tf.keras.applications.VGG19.__name__,
        preprocess=tf.keras.applications.vgg19.preprocess_input,
        base_model=tf.keras.applications.VGG19,
        image_size=(224, 224),
        batch_size=16,
    ),
    tf.keras.applications.ResNet50V2.__name__: BaseCNN(
        name=tf.keras.applications.ResNet50V2.__name__,
        preprocess=tf.keras.applications.resnet.preprocess_input,
        base_model=tf.keras.applications.ResNet50V2,
        image_size=(224, 224),
        batch_size=16,
    ),
    tf.keras.applications.ResNet152V2.__name__: BaseCNN(
        name=tf.keras.applications.ResNet152V2.__name__,
        preprocess=tf.keras.applications.resnet.preprocess_input,
        base_model=tf.keras.applications.ResNet152V2,
        image_size=(224, 224),
        batch_size=16,
    ),
    tf.keras.applications.InceptionV3.__name__: BaseCNN(
        name=tf.keras.applications.InceptionV3.__name__,
        preprocess=tf.keras.applications.inception_v3.preprocess_input,
        base_model=tf.keras.applications.InceptionV3,
        image_size=(299, 299),
        batch_size=16,
    ),
    tf.keras.applications.InceptionResNetV2.__name__: BaseCNN(
        name=tf.keras.applications.InceptionResNetV2.__name__,
        preprocess=tf.keras.applications.inception_resnet_v2.preprocess_input,
        base_model=tf.keras.applications.InceptionResNetV2,
        image_size=(224, 224),
        batch_size=16,
    ),
    tf.keras.applications.MobileNetV2.__name__: BaseCNN(
        name=tf.keras.applications.MobileNetV2.__name__,
        preprocess=tf.keras.applications.mobilenet_v2.preprocess_input,
        base_model=tf.keras.applications.MobileNetV2,
        image_size=(224, 224),
        batch_size=32,
    ),
    tf.keras.applications.DenseNet201.__name__: BaseCNN(
        name=tf.keras.applications.DenseNet201.__name__,
        preprocess=tf.keras.applications.densenet.preprocess_input,
        base_model=tf.keras.applications.DenseNet201,
        image_size=(224, 224),
        batch_size=32,
    ),
    tf.keras.applications.EfficientNetB7.__name__: BaseCNN(
        name=tf.keras.applications.EfficientNetB7.__name__,
        preprocess=tf.keras.applications.efficientnet.preprocess_input,
        base_model=tf.keras.applications.EfficientNetB7,
        image_size=(600, 600),
        batch_size=1,
    ),
}
