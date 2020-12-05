"""
author: Sunny Bhaveen Chandra
date: 05/12/2020
"""

import os
import utils.config as config
import tensorflow as tf
import utils.data_management as dm
import matplotlib.pyplot as plt


class Predict:
    def __init__(self, latest=True, model_index=None):
        if latest:
            self.get_latest_model_path()
        elif (model_index is not None) and (not latest):
            self.model_index = model_index
            self.get_other_models() 
        self.my_model = tf.keras.models.load_model(self.latest_model_path)

    def get_latest_model_path(self):
        available_models = os.listdir(config.TRAINED_MODEL_DIR)
        latest_model = sorted(available_models)[-1]
        self.latest_model_path = os.path.join(config.TRAINED_MODEL_DIR, latest_model)

    def get_other_models(self):
        available_models = os.listdir(config.TRAINED_MODEL_DIR)
        latest_model = sorted(available_models)[self.model_index]
        self.latest_model_path = os.path.join(config.TRAINED_MODEL_DIR, latest_model)

    def predict(self, input_img_path=None):
        img = plt.imread(input_img_path)
        fit_img = dm.manage_input_data(img)
        result = self.my_model.predict(fit_img)
        print("### RESULT: ", result)

if __name__ == "__main__":
    obj = Predict()
    obj.predict(input_img_path='dog.jpg')

    