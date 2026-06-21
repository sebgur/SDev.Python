import logging
import datetime as dt
from pathlib import Path
from sdevpy.utilities import dates as dts
from sdevpy.tests import conftest
from sdevpy.utilities import jsonmanager as jsm
log = logging.getLogger(__name__)


class CalibrationDataFileProvider:
    """ Reads calibrated data from files on disk """
    def __init__(self, root: str|Path=None):
        self.root = (Path(root) if root is not None else conftest.calibdata_path())

    def get_impliedvol_data(self, name: str, date: dt.datetime, model_name: str) -> dict|None:
        """ Retrieve implied vol data if existing, None otherwise """
        file = self.impliedvol_data_file(name, date, model_name)
        if not file.exists():
            log.debug(f'ImpliedVol file not found: {file}')

        return (jsm.deserialize(file) if file.exists() else None)

    def get_localvol_data(self, name: str, date: dt.datetime, model_name: str) -> dict|None:
        """ Retrieve local vol data if existing, None otherwise """
        file = self.localvol_data_file(name, date, model_name)
        if not file.exists():
            log.debug(f'LocalVol file not found: {file}')

        return (jsm.deserialize(file) if file.exists() else None)

    def impliedvol_data_file(self, name: str, date: dt.datetime, model_name: str) -> Path:
        """ Data file for implied vol models """
        folder = self.root / 'impliedvol'
        return Path(folder) / name / (date.strftime(dts.DATE_FILE_FORMAT) + "." + model_name + ".json")

    def localvol_data_file(self, name: str, date: dt.datetime, model_name: str) -> Path:
        """ Retrieve data file for local vol models """
        folder = self.root / 'localvol'
        return Path(folder) / name / (date.strftime(dts.DATE_FILE_FORMAT) + "." + model_name + ".json")


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
