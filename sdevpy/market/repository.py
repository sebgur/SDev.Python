import logging
import datetime as dt
# from collections import OrderedDict
from sdevpy.utilities.datedrepository import DatedRepository
from sdevpy.market.source import MarketDataSource
from sdevpy.market.dataset import MarketDataSet
from sdevpy.market.fixings import FixingHandler
log = logging.getLogger(__name__)


class MarketDataRepository(DatedRepository[MarketDataSet]):
    """ Cache of MarketDataSets keyed by datetime, least recently used evicted first """
    def __init__(self, source: MarketDataSource, max_datasets: int|None=None):
        super().__init__(max_datasets)
        self.source = source
        self._fixing_handlers: dict[tuple, FixingHandler] = {}

    def _new_dataset(self, date: dt.datetime) -> MarketDataSet:
        return MarketDataSet(date, self.source)

    # Fixings are a time series, not a snapshot, so they live here rather than in a dataset
    def get_fixing_handler(self, name: str, **kwargs) -> FixingHandler:
        key = (name, tuple(sorted(kwargs.items())))
        if key not in self._fixing_handlers:
            self._fixing_handlers[key] = self.source.get_fixing_handler(name, **kwargs)
        return self._fixing_handlers[key]

    def get_fixings(self, name: str, dates: dt.datetime|list[dt.datetime], **kwargs) -> list[float]:
        return self.get_fixing_handler(name, **kwargs).value(dates)


# class MarketDataRepository:
#     """ Cache of MarketDataSets keyed by datetime, least recently used evicted first """
#     def __init__(self, source: MarketDataSource, max_datasets: int|None=None):
#         self.source = source
#         self.max_datasets = max_datasets
#         self._datasets: OrderedDict[dt.datetime, MarketDataSet] = OrderedDict()
#         self._fixing_handlers: dict[tuple, FixingHandler] = {}

#     def get(self, date: dt.datetime) -> MarketDataSet:
#         key = self._key(date)
#         dataset = self._datasets.get(key)
#         if dataset is None:
#             dataset = MarketDataSet(key, self.source)
#             self._store(key, dataset)
#         else:
#             self._datasets.move_to_end(key)
#         return dataset

#     __getitem__ = get

#     def put(self, dataset: MarketDataSet):
#         """ Insert a dataset directly, e.g. a bumped scenario """
#         self._store(self._key(dataset.date), dataset)

#     def _store(self, key: dt.datetime, dataset: MarketDataSet):
#         self._datasets[key] = dataset
#         self._datasets.move_to_end(key)
#         if self.max_datasets is not None and len(self._datasets) > self.max_datasets:
#             old_key, _ = self._datasets.popitem(last=False)
#             log.debug(f"Evicting market dataset {old_key}")

#     def invalidate(self, date: dt.datetime|None=None):
#         if date is None:
#             self._datasets.clear()
#         else:
#             self._datasets.pop(self._key(date), None)

#     @staticmethod
#     def _key(date: dt.datetime) -> dt.datetime:
#         return date

#     # Fixings are a time series, not a snapshot, so they live here rather than in a dataset
#     def get_fixing_handler(self, name: str, **kwargs) -> FixingHandler:
#         key = (name, tuple(sorted(kwargs.items())))
#         if key not in self._fixing_handlers:
#             self._fixing_handlers[key] = self.source.get_fixing_handler(name, **kwargs)
#         return self._fixing_handlers[key]

#     def get_fixings(self, name: str, dates: dt.datetime|list[dt.datetime], **kwargs) -> list[float]:
#         return self.get_fixing_handler(name, **kwargs).value(dates)
