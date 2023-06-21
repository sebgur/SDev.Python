""" Wrapper class for machine learning models, including scalers, and simplifying
    evaluation, history tracking, exporting to/importing from files, etc. """
import os
from sklearn.preprocessing import StandardScaler
# from keras.models import load_model
import tensorflow as tf
import joblib
import absl.logging
from sdevpy.tools import jsonmanager

class LearningModel:
    """ Wrapper class for machine learning models, including scalers, and simplifying
        evaluation, history tracking, exporting to/importing from files, etc. """
    def __init__(self, model, is_scaled=False,
                 x_scaler=StandardScaler(copy=True), y_scaler=StandardScaler(copy=True)):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.is_scaled = is_scaled
        self.topology_ = None
        self.optimizer_ = None

    def train(self, x_set, y_set, epochs, batch_size, callback=None,
              verbose=0, shuffle=True):
        """ Scale on first call, then train """
        if not self.is_scaled:
            self.x_scaler.fit(x_set)
            self.y_scaler.fit(y_set)
            self.is_scaled = True

        callbacks = []
        if callback is not None:
            callback.set_scalers(self.x_scaler, self.y_scaler)
            callback.total_epochs = epochs
            callback.batch_size = batch_size
            callback.shuffle = shuffle
            callback.set_size = x_set.shape[0]
            callbacks = [callback]

        x_scaled = self.x_scaler.transform(x_set)
        y_scaled = self.y_scaler.transform(y_set)

        history = self.model.fit(x_scaled, y_scaled, epochs=epochs, batch_size=batch_size,
                                 shuffle=shuffle, verbose=verbose, callbacks=callbacks)

        return history

    def predict(self, x_test):
        """ Predict, including scaling inputs/outputs """
        x_scaled = self.x_scaler.transform(x_test)
        y_scaled = self.model(x_scaled)
        y_test = self.y_scaler.inverse_transform(y_scaled)
        return y_test

    def save(self, path):
        """ Save model and its scalers to files """
        # Save keras model first. Turn dummy warning off temporarily.
        verbosity = absl.logging.get_verbosity()
        absl.logging.set_verbosity(absl.logging.ERROR)
        self.model.save(path)
        absl.logging.set_verbosity(verbosity)

        # Save scalers
        x_scaler_file, y_scaler_file = scaler_files(path)
        joblib.dump(self.x_scaler, x_scaler_file)
        joblib.dump(self.y_scaler, y_scaler_file)
        config_data = { 'topology': self.topology_, 'optimizer': self.optimizer_ }
        config_file = os.path.join(path, 'config.json')

        # Save additional config
        jsonmanager.serialize(config_data, config_file)


    def calculate(self, x_test, diff=False):
        """ Predict with calculation of differentials or not """
        if diff is True:
            return self.calculate_with_greeks(x_test)
        else:
            return self.predict(x_test), None

    def calculate_with_greeks(self, x_test):
        """ Predit with calculation of differentials """
        # x-scaler in TF
        x_mean = self.x_scaler.mean_
        x_scale = self.x_scaler.scale_
        tf_x_mean = tf.convert_to_tensor(x_mean)
        tf_x_scale = tf.convert_to_tensor(x_scale)
        # y-scaler in TF
        y_mean = self.y_scaler.mean_
        y_scale = self.y_scaler.scale_
        tf_y_mean = tf.convert_to_tensor(y_mean, dtype='float32')
        tf_y_scale = tf.convert_to_tensor(y_scale, dtype='float32')

        # Evaluate and record differentials
        md_x_tensor = tf.convert_to_tensor(x_test)
        with tf.GradientTape() as t:
            t.watch(md_x_tensor)
            md_x_scaled = (md_x_tensor - tf_x_mean) / tf_x_scale
            md_y_scaled = self.model(md_x_scaled)
            md_y = md_y_scaled * tf_y_scale + tf_y_mean

        # Retrieve results
        base = md_y[0].numpy()
        grads = t.gradient(md_y, md_x_tensor)
        diffs = grads.numpy()
        return base, diffs

    def scale_inputs(self, x_data):
        """ Scale inputs """
        return self.x_scaler.transform(x_data)

    def scaleback_outputs(self, y_data):
        """ Scale back outputs """
        return self.y_scaler.inverse_transform(y_data)


def scaler_files(path):
    """ Scaler files corresponding to model stored in path """
    x_scaler_file = os.path.join(path, "x_scaler.h5")
    y_scaler_file = os.path.join(path, "y_scaler.h5")
    return x_scaler_file, y_scaler_file

def load_learning_model(path, compile_=False):
    """ Load learning model from files. Note that for now, we set compile=False when loading
        the keras model as we do not know how to save and load custom components such as
        the scheduler or the callback. To restart the training after loading the model from
        file, we would have to be able to properly save and load those custom components.
         
        One possibility could be to implement additional custom saving, recreate those components
        by hand, and then compile again. """

    if os.path.exists(path) is False:
        raise RuntimeError("Model folder does not exist: " + path)

    keras_model = tf.keras.models.load_model(path, compile=compile_)

    x_scaler_file, y_scaler_file = scaler_files(path)
    if os.path.exists(x_scaler_file) and os.path.exists(y_scaler_file):
        x_scaler = joblib.load(x_scaler_file)
        y_scaler = joblib.load(y_scaler_file)
        model = LearningModel(keras_model, is_scaled=True, x_scaler=x_scaler, y_scaler=y_scaler)
    else:
        model = LearningModel(keras_model)

    config_file = os.path.join(path, 'config.json')
    if os.path.exists(config_file):
        config_data = jsonmanager.deserialize(config_file)
        model.topology_ = config_data['topology']
        model.optimizer_ = config_data['optimizer']

    return model
