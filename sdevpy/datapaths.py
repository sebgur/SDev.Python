import os
from pathlib import Path


def root_path() -> Path:
    """ Get root path """
    env = os.environ.get("SDEVPY_DATA")
    if env is not None:
        return Path(env)

    return Path(__file__).parent / "tests" / "data"


def marketdata_path() -> Path:
    """ Get market data path """
    path = root_path() / "marketdata"
    return path


def calibdata_path() -> Path:
    """ Get calibrated data path """
    path = root_path() / "calibdata"
    return path


def staticdata_path() -> Path:
    """ Get static data path """
    path = root_path() / "staticdata"
    return path


def dataset_path() -> Path:
    """ Get dataset path """
    path = root_path() / "datasets"
    return path


if __name__ == "__main__":
    print(marketdata_path())
