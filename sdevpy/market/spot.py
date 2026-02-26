import os, json
import datetime as dt
from pathlib import Path
from sdevpy.tools import dates


class SpotData:
    def __init__(self, valdate, value, **kwargs):
        self.valdate = valdate
        self.snapdate = kwargs.get('snapdate', self.valdate)
        self.name = kwargs.get('name', '')
        self.value = value

    def dump(self, file, indent=2):
        data = self.dump_data()
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)

    def dump_data(self):
        data = {'name': self.name, 'valdate': self.valdate.strftime(dates.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT), 'value': self.value}
        return data


def spotdata_from_file(file):
    with open(file, 'r') as f:
        data = json.load(f)

    name = data.get('name')
    valdate = data.get('valdate')
    snapdate = data.get('snapdate')
    value = data.get('value')

    data = SpotData(dt.datetime.strptime(valdate, dates.DATE_FORMAT), value,
                    name=name, snapdate=dt.datetime.strptime(snapdate, dates.DATETIME_FORMAT))
    return data


def data_file(name, date, **kwargs):
    folder = kwargs.get('folder', test_data_folder())
    name_folder = os.path.join(folder, name)
    os.makedirs(name_folder, exist_ok=True)
    file = os.path.join(name_folder, date.strftime(dates.DATE_FILE_FORMAT) + ".json")
    return file


def test_data_folder():
    folder = Path(__file__).parent.parent.parent.resolve()
    dataset_folder = os.path.join(folder, "datasets")
    folder = os.path.join(os.path.join(dataset_folder, "marketdata"), "spot")
    os.makedirs(folder, exist_ok=True)
    return folder


if __name__ == "__main__":
    name = "ABC"
    valdate = dt.datetime(2026, 2, 15)

    # Generate a sample to start from
    spot = 100.0
    data = SpotData(valdate, spot, name=name)
    file = data_file(name, valdate)
    data.dump(file)

    # Get data from existing file
    test_data = spotdata_from_file(file)
    test_value = test_data.value
    print(test_value)
