""" Wrapper class for machine learning models, including scalers, and simplifying
    evaluation, history tracking, exporting to/importing from files, etc. """
from pathlib import Path
import joblib
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sdevpy.utilities import jsonmanager as jsm


class LearningModel(ABC):
    """ Wrapper class for machine learning models, including scalers, and simplifying
        evaluation, history tracking, exporting to/importing from files, etc. """
    def __init__(self, base_model): #, is_scaled: bool=False, x_scaler=None, y_scaler=None):
        self.base_model = base_model
        # if x_scaler is None:
        x_scaler = StandardScaler(copy=True)
        self.x_scaler = x_scaler
        # if y_scaler is None:
        y_scaler = StandardScaler(copy=True)
        self.y_scaler = y_scaler
        self.is_scaled = False
        # self.is_scaled = is_scaled
        self.topology = None
        self.optimizer = None

    def set_scalers(self, x_scaler, y_scaler) -> None:
        """ Set scalers for inputs (x) and outputs (y) """
        self.x_scaler, self.y_scaler, self.is_scaled = x_scaler, y_scaler, True

    def train(self, x_set: npt.ArrayLike, y_set: npt.ArrayLike, epochs: int, batch_size: int,
              shuffle: bool=True) -> npt.ArrayLike:
        """ Scale on first call, then train """
        if not self.is_scaled:
            self.x_scaler.fit(x_set)
            self.y_scaler.fit(y_set)
            self.is_scaled = True

        x_scaled = self.x_scaler.transform(x_set)
        y_scaled = self.y_scaler.transform(y_set)

        return self.train_on_scaled(x_scaled, y_scaled, epochs, batch_size, shuffle)

    @abstractmethod
    def train_on_scaled(self, x_scaled: npt.ArrayLike, y_scaled: npt.ArrayLike, epochs: int, batch_size: int,
                        shuffle: bool) -> None:
        """ Training (scaling already done) """
        pass

    def predict(self, x_test: npt.ArrayLike) -> npt.ArrayLike:
        """ Predict, including x-scaling and y-scaling """
        # Scale inputs
        x_scaled = self.x_scaler.transform(x_test)
        # Predict scaled outputs
        y_scaled = self.predict_on_scaled(x_scaled)
        # Unscale outputs
        y_test = self.y_scaler.inverse_transform(y_scaled)
        return y_test

    @abstractmethod
    def predict_on_scaled(self, x_scaled: npt.ArrayLike) -> npt.ArrayLike:
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

    def save_scalers(self, path: Path) -> None:
        """ Save x and y scalers to path """
        x_scaler_file, y_scaler_file = scaler_files(path)
        joblib.dump(self.x_scaler, x_scaler_file)
        joblib.dump(self.y_scaler, y_scaler_file)


def scaler_files(path: Path) -> tuple[Path, Path]:
    """ Scaler files corresponding to model stored in path """
    x_scaler_file = path / "x_scaler.h5"
    y_scaler_file = path / "y_scaler.h5"
    return x_scaler_file, y_scaler_file


@dataclass
class MlpTopology:
    input_dim: int
    output_dim: int
    layers: list[str]
    neurons: int
    dropout: float

    def to_json(self, path: Path) -> None:
        """ Dump to json """
        jsm.serialize(asdict(self), path)

    @classmethod
    def from_json(cls, path: Path) -> "MlpTopology":
        return cls(**jsm.deserialize(path))

