import datetime as dt
from sdevpy.utilities.datedrepository import DatedRepository
from sdevpy.calibration.source import CalibrationDataSource
from sdevpy.calibration.dataset import CalibrationDataSet
from sdevpy.calibration.rates.yieldcurve import YieldCurve
from sdevpy.utilities import dates as dts


MIN_DATE = dt.datetime(1990, 1, 1)  # Assuming the database does not hold data before that


class CalibrationDataRepository(DatedRepository[CalibrationDataSet]):
    """ Cache of CalibrationDataSets keyed by datetime, least recently used evicted first """
    def __init__(self, source: CalibrationDataSource, max_datasets: int|None=None, min_date: dt.datetime=MIN_DATE):
        super().__init__(max_datasets)
        self.source = source
        self.min_date = min_date

    def get_available_curves(self, curve_ids: list[str], start_date: dt.datetime=None,
                             end_date: dt.datetime=None) -> dict[dt.datetime, dict[str, YieldCurve]]:
        """ For each weekday in [start_date, end_date] where every curve in curve_ids is available, retrieve them all.
            Returns a dict keyed by date in chronological order, each value a dict of curve_id -> YieldCurve.
            Dates missing any of the requested curves are skipped entirely.
            If start_date or end_date are not specified (None), the start date is the minimum date held by the repo
            and the end date is now. """
        # Default values for start and end
        start_date = self.min_date if start_date is None else start_date
        end_date = dts.today() if end_date is None else end_date

        # Find available
        result: dict[dt.datetime, dict[str, YieldCurve]] = {}
        date = start_date
        while date <= end_date:
            dataset = self[date]
            try:
                curves = {curve_id: dataset.get_yieldcurve(curve_id) for curve_id in curve_ids}
            except FileNotFoundError:
                pass
            else:
                result[date] = curves
            date += dt.timedelta(days=1)

        return result

    def _new_dataset(self, date: dt.datetime) -> CalibrationDataSet:
        return CalibrationDataSet(date, self.source)
