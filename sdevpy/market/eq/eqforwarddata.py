import json
import datetime as dt
import numpy as np
from pathlib import Path
from sdevpy.utilities import dates as dts
from sdevpy.utilities import jsonmanager as jsm


class EqForwardData:
    def __init__(self, valdate, pillars, **kwargs):
        self.valdate = valdate
        self.snapdate = kwargs.get('snapdate', self.valdate)
        self.name = kwargs.get('name', '')

        # Sort by increasing date
        pillars.sort(key=lambda x: x['expiry'])

        # Extract
        self.expiries = np.asarray([s['expiry'] for s in pillars])
        self.forwards = np.asarray([s['forward'] for s in pillars])

    def dump(self, file, indent=2):
        data = self.dump_data()
        jsm.serialize(data, file, indent)

    def dump_data(self):
        pillars = []
        for expiry, forward in zip(self.expiries, self.forwards, strict=True):
            expiry_str = expiry.strftime(dts.DATE_FORMAT)
            pillar = {'expiry': expiry_str, 'forward': forward}
            pillars.append(pillar)

        data = {'name': self.name, 'valdate': self.valdate.strftime(dts.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dts.DATETIME_FORMAT), 'pillars': pillars}
        return data


def eqforwarddata_from_file(file: str|Path) -> EqForwardData:
    """ Retrieve EQ forward data from file """
    with open(file) as f:
        data = json.load(f)

    name = data.get('name')
    valdate = data.get('valdate')
    snapdate = data.get('snapdate')
    pillars = data.get('pillars')

    # Convert date strings into dates
    for pillar in pillars:
        date_str = pillar.get('expiry')
        date = dt.datetime.strptime(date_str, dts.DATE_FORMAT)
        pillar['expiry'] = date

    data = EqForwardData(dt.datetime.strptime(valdate, dts.DATE_FORMAT), pillars,
                         name=name, snapdate=dt.datetime.strptime(snapdate, dts.DATETIME_FORMAT))
    return data


if __name__ == "__main__":
    print("Hello")
