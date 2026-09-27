import datetime as dt
from sdevpy.utilities.datedrepository import DatedRepository
from sdevpy.calibration.source import CalibrationDataSource
from sdevpy.calibration.dataset import CalibrationDataSet


class CalibrationDataRepository(DatedRepository[CalibrationDataSet]):
    """ Cache of CalibrationDataSets keyed by datetime, least recently used evicted first """
    def __init__(self, source: CalibrationDataSource, max_datasets: int|None=None):
        super().__init__(max_datasets)
        self.source = source

    def _new_dataset(self, date: dt.datetime) -> CalibrationDataSet:
        return CalibrationDataSet(date, self.source)
