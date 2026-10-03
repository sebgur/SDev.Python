import logging
import datetime as dt
from pathlib import Path
from sdevpy.utilities import dates as dts
from sdevpy.utilities import jsonmanager as jsm
from sdevpy.calibration.rates import yieldcurve as ycrv
from sdevpy.calibration.rates.yieldcurve import YieldCurve
from sdevpy.calibration.source import CalibrationDataSource
from sdevpy import datapaths
log = logging.getLogger(__name__)


class CalibrationDataFileSource(CalibrationDataSource):
    """ Reads/writes calibrated data from/to files on disk """
    def __init__(self, root: str|Path=None):
        if root is None:
            self.root = datapaths.calibdata_path()
            log.info(f"No root given, using default data folder: {self.root}")
        else:
            self.root = Path(root)

    def get_yieldcurve(self, name: str, date: dt.datetime) -> YieldCurve:
        """ Retrieve yield curve """
        return ycrv.yieldcurve_from_file(self._data_file('yieldcurves', name, date))

    def get_impliedvol_data(self, name: str, date: dt.datetime, model_name: str) -> dict|None:
        """ Retrieve implied vol data if existing, None otherwise """
        return self._read(self._data_file('impliedvol', name, date, model_name))

    def get_localvol_data(self, name: str, date: dt.datetime, model_name: str) -> dict|None:
        """ Retrieve local vol data if existing, None otherwise """
        return self._read(self._data_file('localvol', name, date, model_name))

    def get_fxvoldata(self, pair: str, date: dt.datetime) -> dict|None:
        """ Retrieve FX vol data if existing, None otherwise """
        return self._read(self._data_file('fxvol', pair, date))

    def save_impliedvol_data(self, name: str, date: dt.datetime, model_name: str, data: dict) -> None:
        self._write(self._data_file('impliedvol', name, date, model_name), data)

    def save_localvol_data(self, name: str, date: dt.datetime, model_name: str, data: dict) -> None:
        self._write(self._data_file('localvol', name, date, model_name), data)

    def save_fxvoldata(self, pair: str, date: dt.datetime, data: dict) -> None:
        self._write(self._data_file('fxvol', pair, date), data)

    def _data_file(self, category: str, name: str, date: dt.datetime, model_name: str|None=None) -> Path:
        """ Data file for given category, name and date: root/category/name/yyyymmdd-hhmmss[.model].json """
        stem = date.strftime(dts.DATETIME_FILE_FORMAT)
        if model_name is not None:
            stem += "." + model_name
        return self.root / category / name / (stem + ".json")

    @staticmethod
    def _read(file: Path) -> dict|None:
        if not file.exists():
            log.debug(f'Calibration file not found: {file}')
            return None
        return jsm.deserialize(file)

    @staticmethod
    def _write(file: Path, data: dict) -> None:
        file.parent.mkdir(parents=True, exist_ok=True)
        jsm.serialize(data, file)
        log.debug(f'Calibration file written: {file}')


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
    source = CalibrationDataFileSource()
    print(source.get_yieldcurve("USD.SOFR.1D", valdate))
