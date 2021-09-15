import os
import tensorflow as tf
import numpy as np
import utils.config as config
import time

def save_vgg_16_model(input_shape=config.IMAGE_SIZE):
    model = tf.keras.applications.vgg16.VGG16(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False
    )
    model.save("original_vgg_base.h5")
    print("base model is saved")

def load_base_model():
    save_vgg_16_model(input_shape=config.IMAGE_SIZE)
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


def callbacks(base_dir="."):

    # tensorboard callbacks
    base_log_dir = config.TENSORBOARD_ROOT_LOG_DIR
    unique_log = time.strftime("log_at_%Y%m%d_%H%M%S")
    tensorboard_log_dir = os.path.join(base_log_dir, unique_log)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)

    # checkpoint callbacks

    checkpoint_file = os.path.join(config.CHECKPOINT_DIR, "vgg_16model_checkpoint.h5")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file,
        save_best_only=True
    )

    callback_list = [tensorboard_cb, checkpoint_cb]

    return callback_list


if __name__ == "__main__":
    # save_vgg_16_model()
    # load_base_model()
    custom_model()