from pathlib import Path
import json
import datetime as dt
from sdevpy.utilities import dates
from sdevpy.utilities import jsonmanager as jsm


class SpotData:
    def __init__(self, valdate: dt.datetime, value: float, **kwargs):
        self.valdate = valdate
        self.snapdate = kwargs.get('snapdate', self.valdate)
        self.name = kwargs.get('name', '')
        self.value = value

    def dump(self, file: str, indent: int=2) -> None:
        """ Dump object to json file """
        data = self.dump_data()
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)

    def dump_data(self) -> dict:
        """ Dump object to dictionary """
        data = {'name': self.name, 'valdate': self.valdate.strftime(dates.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT), 'value': self.value}
        return data


def spotdata_from_file(file: str|Path) -> SpotData:
    """ Retrieve spot data object from file """
    data = jsm.deserialize(file)

    name = data.get('name')
    valdate = data.get('valdate')
    snapdate = data.get('snapdate')
    value = data.get('value')

    data = SpotData(dt.datetime.strptime(valdate, dates.DATE_FORMAT), value,
                    name=name, snapdate=dt.datetime.strptime(snapdate, dates.DATETIME_FORMAT))
    return data


if __name__ == "__main__":
    name, valdate = "ABC", dt.datetime(2026, 2, 15)
