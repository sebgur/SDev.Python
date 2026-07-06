""" Wrapper class for machine learning models, including scalers, and simplifying
    evaluation, history tracking, exporting to/importing from files, etc. """
from pathlib import Path
from abc import ABC, abstractmethod
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
# import absl.logging
# from sdevpy.utilities import jsonmanager
# from sdevpy.utilities import filemanager


class LearningModel(ABC):
    """ Wrapper class for machine learning models, including scalers, and simplifying
        evaluation, history tracking, exporting to/importing from files, etc. """
    def __init__(self, model, is_scaled: bool=False, x_scaler=None, y_scaler=None, verbose: bool=False):
        self.model = model
        if x_scaler is None:
            x_scaler = StandardScaler(copy=True)
        self.x_scaler = x_scaler
        if y_scaler is None:
            y_scaler = StandardScaler(copy=True)
        self.y_scaler = y_scaler
        self.is_scaled = is_scaled
        self.topology_ = None
        self.optimizer_ = None
        self.verbose = verbose

    def set_callback(self, callback=None) -> None:
        """ Set specific callback """
        self.callback = callback

    def train(self, x_set: npt.ArrayLike, y_set: npt.ArrayLike, epochs: int, batch_size: int,
              shuffle: bool=True) -> npt.ArrayLike:
        """ Scale on first call, then train """
        if not self.is_scaled:
            self.x_scaler.fit(x_set)
            self.y_scaler.fit(y_set)
            self.is_scaled = True

        x_scaled = self.x_scaler.transform(x_set)
        y_scaled = self.y_scaler.transform(y_set)

        return self.train_raw(x_scaled, y_scaled, epochs, batch_size, shuffle)

    @abstractmethod
    def train_raw(self, x_scaled: npt.ArrayLike, y_scaled: npt.ArrayLike, epochs: int, batch_size: int, shuffle: bool):
        """ Training (scaling already done) """
        pass

    def predict(self, x_test: npt.ArrayLike) -> npt.ArrayLike:
        """ Predict, including x-scaling and y-scaling """
        x_scaled = self.x_scaler.transform(x_test)
        y_scaled = self.predict_raw(x_scaled)
        y_test = self.y_scaler.inverse_transform(y_scaled)
        return y_test

    @abstractmethod
    def predict_raw(self, x_scaled: npt.ArrayLike) -> npt.ArrayLike:
        """ Predict (x-scaling already done, y-scaling not done) """
        pass

    @abstractmethod
    def save(self, path: Path):
        """ Save model and its scalers to files """
        pass

    def scale_inputs(self, x_data: npt.ArrayLike) -> npt.ArrayLike:
        """ Scale inputs """
        return self.x_scaler.transform(x_data)

    def scaleback_outputs(self, y_data: npt.ArrayLike) -> npt.ArrayLike:
        """ Scale back outputs """
        return self.y_scaler.inverse_transform(y_data)


def scaler_files(path: Path) -> tuple[Path, Path]:
    """ Scaler files corresponding to model stored in path """
    x_scaler_file = path / "x_scaler.h5"
    y_scaler_file = path / "y_scaler.h5"
    return x_scaler_file, y_scaler_file


# def load_learning_model(path, compile_=False):
#     """ Load learning model from files. Note that for now, we set compile=False when loading
#         the keras model as we do not know how to save and load custom components such as
#         the scheduler or the callback. To restart the training after loading the model from
#         file, we would have to be able to properly save and load those custom components.

#         One possibility could be to implement additional custom saving, recreate those components
#         by hand, and then compile again. """

#     if os.path.exists(path) is False:
#         raise RuntimeError("Model folder does not exist: " + path)

#     model_file = os.path.join(path, "model.keras")
#     keras_model = tf.keras.models.load_model(model_file, compile=compile_)

#     x_scaler_file, y_scaler_file = scaler_files(path)
#     if os.path.exists(x_scaler_file) and os.path.exists(y_scaler_file):
#         x_scaler = joblib.load(x_scaler_file)
#         y_scaler = joblib.load(y_scaler_file)
#         model = KerasLearningModel(keras_model, is_scaled=True, x_scaler=x_scaler, y_scaler=y_scaler)
#     else:
#         model = KerasLearningModel(keras_model)

#     config_file = os.path.join(path, 'config.json')
#     if os.path.exists(config_file):
#         config_data = jsonmanager.deserialize(config_file)
#         model.topology_ = config_data['topology']
#         model.optimizer_ = config_data['optimizer']

#     return model
