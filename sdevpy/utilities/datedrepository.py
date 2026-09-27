import logging
import datetime as dt
from collections import OrderedDict
log = logging.getLogger(__name__)


class DatedRepository[T]:
    """ Cache of per-datetime datasets, least recently used evicted first.
        Subclasses implement _new_dataset(). Datasets must expose a 'date' attribute. """
    def __init__(self, max_datasets: int|None=None):
        self.max_datasets = max_datasets
        self._datasets: OrderedDict[dt.datetime, T] = OrderedDict()

    def _new_dataset(self, date: dt.datetime) -> T:
        raise NotImplementedError

    def get(self, date: dt.datetime) -> T:
        key = self._key(date)
        dataset = self._datasets.get(key)
        if dataset is None:
            dataset = self._new_dataset(key)
            self._store(key, dataset)
        else:
            self._datasets.move_to_end(key)
        return dataset

    __getitem__ = get

    def put(self, dataset: T):
        """ Insert a dataset directly, e.g. a bumped scenario """
        self._store(self._key(dataset.date), dataset)

    def _store(self, key: dt.datetime, dataset: T):
        self._datasets[key] = dataset
        self._datasets.move_to_end(key)
        if self.max_datasets is not None and len(self._datasets) > self.max_datasets:
            old_key, _ = self._datasets.popitem(last=False)
            log.debug(f"Evicting {type(self).__name__} dataset {old_key}")

    def invalidate(self, date: dt.datetime|None=None):
        if date is None:
            self._datasets.clear()
        else:
            self._datasets.pop(self._key(date), None)

    @staticmethod
    def _key(date: dt.datetime) -> dt.datetime:
        return date
