from pathlib import Path


def test_root_path() -> Path:
    """ Get test root path """
    path = Path(__file__).parent / "data"
    return path


def test_marketdata_path() -> Path:
    """ Get test market data path """
    path = test_root_path() / "marketdata"
    return path


def test_calibdata_path() -> Path:
    """ Get test calibrated data path """
    path = test_root_path() / "calibdata"
    return path


if __name__ == "__main__":
    print(test_marketdata_path())
