import os
import tensorflow as tf
import numpy as np
import config

def save_vgg_16_model(input_shape=config.IMAGE_SIZE):
    model = tf.keras.applications.vgg16.VGG16(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False
    )
    model.save("original_vgg_base.h5")
    print("base model is saved")

def load_base_model():
    model = tf.keras.models.load_model("original_vgg_base.h5")
    print("original base model is loaded")
    model.summary()
    return model

def custom_model(CLASSES=config.CLASSES, freeze_all=True, freeze_till=None):
    model = load_base_model()

    # freeze weights - 
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False
    elif (freeze_till is not None) and (freeze_till > 0):
        for layer in model.layers[:freeze_till]:
            layer.trainable = False

    # add custom layers -
    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(
        units=CLASSES,
        activation="softmax"
    )(flatten_in)

    full_model = tf.keras.models.Model(
        inputs=model.input,
        outputs = prediction
    )
    print("custom model summary")
    full_model.summary()

    full_model.compile(
        optimizer = tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ["accuracy"]
    )

    return full_model


if __name__ == "__main__":
    # save_vgg_16_model()
    # load_base_model()
    custom_model()