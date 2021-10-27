import tensorflow as tf

# See: https://keras.io/api/applications/

CNN_APPS = {
    # tf.keras.applications.VGG19.__name__: {
    #     "image_size": (224, 224),
    #     "scale": 1.0 / 255,
    #     "offset": 0,
    #     "preprocessor": tf.keras.applications.vgg19.preprocess_input,
    #     "class": tf.keras.applications.VGG19,
    #     "batch_size": 32,
    # },
    # tf.keras.applications.ResNet50V2.__name__: {
    #     "image_size": (224, 224),
    #     "scale": 1.0 / 255,
    #     "offset": 0,
    #     "preprocessor": tf.keras.applications.resnet.preprocess_input,
    #     "class": tf.keras.applications.ResNet50V2,
    #     "batch_size": 32,
    # },
    tf.keras.applications.ResNet152V2.__name__: {
        "image_size": (224, 224),
        "scale": 1.0 / 255,
        "offset": 0,
        "preprocessor": tf.keras.applications.resnet.preprocess_input,
        "class": tf.keras.applications.ResNet152V2,
        "batch_size": 32,
    },
    # tf.keras.applications.InceptionV3.__name__: {
    #     "image_size": (299, 299),
    #     "scale": 1.0,
    #     "offset": 0,
    #     "preprocessor": tf.keras.applications.inception_v3.preprocess_input,
    #     "class": tf.keras.applications.InceptionV3,
    #     "batch_size": 16,
    # },
    tf.keras.applications.InceptionResNetV2.__name__: {
        "image_size": (299, 299),
        "scale": 1.0,
        "offset": 0,
        "preprocessor": tf.keras.applications.inception_resnet_v2.preprocess_input,
        "class": tf.keras.applications.InceptionResNetV2,
        "batch_size": 24,
    },
    # tf.keras.applications.MobileNetV2.__name__: {
    #     "image_size": (224, 224),
    #     "scale": 1.0,
    #     "offset": 0,
    #     "preprocessor": tf.keras.applications.mobilenet_v2.preprocess_input,
    #     "class": tf.keras.applications.MobileNetV2,
    #     "batch_size": 32,
    # },
    # tf.keras.applications.DenseNet201.__name__: {
    #     "image_size": (224, 224),
    #     "scale": 1.0 / 255,
    #     "offset": 0,
    #     "preprocessor": tf.keras.applications.densenet.preprocess_input,
    #     "class": tf.keras.applications.DenseNet201,
    #     "batch_size": 32,
    # },
    # tf.keras.applications.EfficientNetB7.__name__: {
    #     "image_size": (600, 600),
    #     "scale": 1.0 / 255,
    #     "offset": 0,
    #     "preprocessor": tf.keras.applications.efficientnet.preprocess_input,
    #     "class": tf.keras.applications.EfficientNetB7,
    #     "batch_size": 1,
    # },
}
