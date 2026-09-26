import logging
import datetime as dt
from pathlib import Path
from sdevpy.utilities import dates as dts
from sdevpy.utilities import jsonmanager as jsm
from sdevpy.market import yieldcurve as ycrv
from sdevpy.market.yieldcurve import YieldCurve
from sdevpy.calibration.provider import CalibrationDataProvider
from sdevpy import datapaths
log = logging.getLogger(__name__)


class CalibrationDataFileProvider(CalibrationDataProvider):
    """ Reads calibrated data from files on disk """
    def __init__(self, root: str|Path=None):
        if root is None:
            self.root = datapaths.calibdata_path()
            log.info(f"No root given, using default data folder: {self.root}")
        else:
            self.root = Path(root)

    def get_yieldcurve(self, name: str, date: dt.datetime) -> YieldCurve:
        """ Retrieve yield curve """
        folder = self.root / 'yieldcurves'
        file = Path(folder) / name / (date.strftime(dts.DATE_FILE_FORMAT) + ".json")
        curve = ycrv.yieldcurve_from_file(file)
        return curve

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

    def get_fxvol_data(self, pair: str, date: dt.datetime) -> dict|None:
        """ Retrieve FX vol data if existing, None otherwise """
        file = self.fxvol_data_file(pair, date)
        if not file.exists():
            log.debug(f'FX Vol file not found: {file}')

        return (jsm.deserialize(file) if file.exists() else None)

    def impliedvol_data_file(self, name: str, date: dt.datetime, model_name: str) -> Path:
        """ Data file for implied vol models """
        folder = self.root / 'impliedvol' / name
        folder.mkdir(parents=True, exist_ok=True)
        return folder / (date.strftime(dts.DATE_FILE_FORMAT) + "." + model_name + ".json")

    def localvol_data_file(self, name: str, date: dt.datetime, model_name: str) -> Path:
        """ Retrieve data file for local vol models """
        folder = self.root / 'localvol' / name
        folder.mkdir(parents=True, exist_ok=True)
        return folder / (date.strftime(dts.DATE_FILE_FORMAT) + "." + model_name + ".json")

    def fxvol_data_file(self, pair: str, date: dt.datetime) -> Path:
        """ Data file for calibrated FX vol surfaces """
        folder = self.root / 'fxvol' / pair
        folder.mkdir(parents=True, exist_ok=True)
        return folder / (date.strftime(dts.DATE_FILE_FORMAT) + ".json")



if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
    cal_provider = CalibrationDataFileProvider()
    obj = cal_provider.get_yieldcurve("USD.SOFR.1D", valdate)
    print(obj)
