from pathlib import Path
import json
import datetime as dt
# import numpy as np
from sdevpy.utilities import dates
from sdevpy.utilities import jsonmanager as jsm


# def get_spots(names: list[str], valdate: dt.datetime, folder: str|Path):
#     """ Get spot prices for specified names on given date """
#     spots = []
#     for name in names:
#         file = data_file(name, valdate, folder)
#         data = spotdata_from_file(file)
#         spots.append(data.value)

#     return np.asarray(spots)


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


def spotdata_from_file(file: str|Path):
    """ Retrieve spot data object from file """
    data = jsm.deserialize(file)

    name = data.get('name')
    valdate = data.get('valdate')
    snapdate = data.get('snapdate')
    value = data.get('value')

    data = SpotData(dt.datetime.strptime(valdate, dates.DATE_FORMAT), value,
                    name=name, snapdate=dt.datetime.strptime(snapdate, dates.DATETIME_FORMAT))
    return data


def data_file(name: str, date: dt.datetime, folder: str|Path) -> Path:
    """ Return the data file given the name, date and folder """
    return Path(folder) / name / (date.strftime(dates.DATE_FILE_FORMAT) + ".json")


if __name__ == "__main__":
    name, valdate = "ABC", dt.datetime(2026, 2, 15)

    # Generate a sample to start from
    spot = 100.0
    data = SpotData(valdate, spot, name=name)
    file = data_file(name, valdate)
    data.dump(file)

    # Get data from existing file
    test_data = spotdata_from_file(file)
    test_value = test_data.value
    print(test_value)
