""" Wrapper class for machine learning models, including scalers, and simplifying
    evaluation, history tracking, exporting to/importing from files, etc. """
from pathlib import Path
import numpy.typing as npt
import tensorflow as tf
import joblib
import absl.logging
from sdevpy.utilities import jsonmanager as jsm
from sdevpy.machinelearning.learningmodel import LearningModel, scaler_files


class KerasLearningModel(LearningModel):
    """ Keras subclass of LearningModel, using TensorFlow backend """
    def __init__(self, base_model): #, is_scaled: bool=False, x_scaler=None, y_scaler=None):
        super().__init__(base_model) #, is_scaled, x_scaler, y_scaler)
        # Now need to call set_scalers() instead
        self.callback = None
        self.history = None
        self.verbose = 0

    def train_on_scaled(self, x_scaled: npt.ArrayLike, y_scaled: npt.ArrayLike, epochs: int, batch_size: int,
                        shuffle: bool) -> None:
        """ Training on scaled data (both x and y) """
        keras_callbacks = []
        if self.callback is not None:
            self.callback.set_scalers(self.x_scaler, self.y_scaler)
            self.callback.total_epochs = epochs
            self.callback.batch_size = batch_size
            self.callback.shuffle = shuffle
            self.callback.set_size = x_scaled.shape[0]
            keras_callbacks = [self.callback]

        history = self.base_model.fit(x_scaled, y_scaled, epochs=epochs, batch_size=batch_size,
                                 shuffle=shuffle, verbose=self.verbose, callbacks=keras_callbacks)

        self.history = history

    def predict(self, x_test):
        """ Predict, including scaling inputs/outputs """
        x_scaled = self.x_scaler.transform(x_test)
        y_scaled = self.base_model(x_scaled)
        y_test = self.y_scaler.inverse_transform(y_scaled)
        return y_test

    def save(self, path: Path):
        """ Save model and its scalers to files """
        path.mkdir(parents=True, exist_ok=True)
        # Save keras model first. Turn dummy warning off temporarily.
        verbosity = absl.logging.get_verbosity()
        absl.logging.set_verbosity(absl.logging.ERROR)
        model_file = path / "model.keras"
        self.base_model.save(model_file)
        absl.logging.set_verbosity(verbosity)

        # Save scalers
        x_scaler_file, y_scaler_file = scaler_files(path)
        joblib.dump(self.x_scaler, x_scaler_file)
        joblib.dump(self.y_scaler, y_scaler_file)
        config_data = {'topology': self.topology, 'optimizer': self.optimizer}
        config_file = path / 'config.json'

        # Save additional config
        jsm.serialize(config_data, config_file)

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
            md_y_scaled = self.base_model(md_x_scaled)
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

    def set_callback(self, callback=None) -> None:
        """ Set specific callback """
        self.callback = callback


def load_learning_model(path: Path, compile_=False):
    """ Load learning model from files. Note that for now, we set compile=False when loading
        the keras model as we do not know how to save and load custom components such as
        the scheduler or the callback. To restart the training after loading the model from
        file, we would have to be able to properly save and load those custom components.

        One possibility could be to implement additional custom saving, recreate those components
        by hand, and then compile again. """
    if not path.exists():
        raise RuntimeError(f"Model folder does not exist: {path}")

    model_file = path / "model.keras"
    keras_model = tf.keras.models.load_model(model_file, compile=compile_)

    x_scaler_file, y_scaler_file = scaler_files(path)
    if x_scaler_file.exists() and y_scaler_file.exists():
        x_scaler = joblib.load(x_scaler_file)
        y_scaler = joblib.load(y_scaler_file)
        model = KerasLearningModel(keras_model, is_scaled=True, x_scaler=x_scaler, y_scaler=y_scaler)
    else:
        model = KerasLearningModel(keras_model)

    config_file = path / 'config.json'
    if config_file.exists():
        config_data = jsm.deserialize(config_file)
        model.topology = config_data['topology']
        model.optimizer = config_data['optimizer']

    return model
