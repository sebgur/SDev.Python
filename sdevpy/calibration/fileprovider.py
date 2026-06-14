import datetime as dt
from pathlib import Path
from sdevpy.volatility.impliedvol import impliedvol as iv_mod
from sdevpy.volatility.localvol import localvol_factory as lvf
from sdevpy.tests import testconfig
from sdevpy.utilities import jsonmanager as jsm


class CalibrationDataFileProvider:
    """ Reads calibrated data from files on disk """
    def __init__(self, root: str|Path=None):
        self.root = (Path(root) if root is not None else testconfig.calibdata_path())

    def get_impliedvol_data(self, name: str, date: dt.datetime, model_name: str) -> dict|None:
        """ Retrieve implied vol data if existing, None otherwise """
        folder = self.root / 'impliedvol'
        file = iv_mod.data_file(name, date, model_name, folder)
        return (jsm.deserialize(file) if file.exists() else None)

    def get_localvol_data(self, name: str, date: dt.datetime, model_name: str) -> dict|None:
        """ Retrieve local vol data if existing, None otherwise """
        folder = self.root / 'localvol'
        file = lvf.data_file(name, date, model_name, folder)
        return (jsm.deserialize(file) if file.exists() else None)


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
