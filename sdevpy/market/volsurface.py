import os, json
import datetime as dt
from pathlib import Path


def dump_data(data, file):
    with open(file, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    # Get path to dataset folder
    folder = Path(__file__).parent.parent.parent.resolve()
    folder = os.path.join(os.path.join(folder, "datasets"), "localvol")
    name = "SomeIndex"
    valdate = dt.datetime(2025, 12, 15)
    print(folder)

    data = {'name': name, 'datetime': valdate.strftime()}
    dump_data(data, file)
    