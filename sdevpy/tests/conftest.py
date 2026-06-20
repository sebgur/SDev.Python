from pathlib import Path


def root_path() -> Path:
    """ Get test root path """
    path = Path(__file__).parent / "data"
    return path


def marketdata_path() -> Path:
    """ Get test market data path """
    path = root_path() / "marketdata"
    return path


def calibdata_path() -> Path:
    """ Get test calibrated data path """
    path = root_path() / "calibdata"
    return path


def staticdata_path() -> Path:
    """ Get test static data path """
    path = root_path() / "staticdata"
    return path


if __name__ == "__main__":
    print(marketdata_path())
