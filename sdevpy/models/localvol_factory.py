import os
import datetime as dt
import json
# from sdevpy.models import impliedvol


DATE_FORMAT = "%d-%b-%y"

def load_lv(date, name, folder):
    file = os.path.join(folder, name)
    os.makedirs(file, exist_ok=True)
    file = os.path.join(file, "localvol_" + valdate.strftime("%Y%m%d-%H.%M.%S") + ".json")

    if not os.path.exists(file):
        raise FileNotFoundError(f"Local vol file not found: {file}")

    return file


def dump_lv(lv, date, name, folder):
    raise NotImplementedError("ToDo")


def write_example(date, name, folder):
    file = os.path.join(folder, name)
    os.makedirs(file, exist_ok=True)
    file = os.path.join(file, "localvol_" + valdate.strftime("%Y%m%d-%H.%M.%S") + ".json")
    sections = []
    for i in range(4):
        time = i / 4.0 * 2.5
        model = 'BiExp'
        params = {'a': 0.5, 'b': 0.2}
        section = {'time': time, 'model': model, 'params': params}
        sections.append(section)

    obj = {'name': name, 'datetime': valdate.strftime(DATE_FORMAT), 'sections': sections}

    with open(file, 'w') as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    folder = "C:\\temp\\sdevpy\\localvol"
    name = "SomeIndex"
    valdate = dt.datetime(2025, 12, 15)

    file = load_lv(valdate, name, folder)
    print(file)

    # Write an example to start from
    write_example(valdate, name, folder)

    # section_grid = [biexp.BiExpSection() for i in range(len(expiry_grid))]
    # lv = localvol.InterpolatedParamLocalVol(expiry_grid, section_grid)

