import os
import datetime as dt
import json
# from sdevpy.models import impliedvol


DATE_FORMAT = "%d-%b-%y"


def create_section(config):
    model = config.get('model', None)
    params = config.get('params', None)
    if model is None or params is None:
        raise TypeError("Invalid section input in local vol file")

    match model.lower():
        case 'biexp':
            section = biexp.create_section(params)
        case 'svivol':
            section = svivol.create_section(params)
        case _:
            section = None
            raise TypeError(f"Unknown section type: {model}")

    return section


def load_lv(date, name, folder):
    file = os.path.join(folder, name)
    os.makedirs(file, exist_ok=True)
    file = os.path.join(file, "localvol_" + valdate.strftime("%Y%m%d-%H.%M.%S") + ".json")

    # Check file existence
    if not os.path.exists(file):
        raise FileNotFoundError(f"Local vol file not found: {file}")

    # Retrieve LV definition
    with open(file, 'r') as f:
        data = json.load(f)

    # Create sections
    sections = data['sections']
    if sections is None:
        raise KeyError("Sections node not valid in LV data")

    expiry_grid = []
    section_grid = []
    for section_config in sections:
        time = section_config['time']
        model = section_config['model']
        params = section_config['params']
        section = create_section(section_config)
        expiry_grid.append(time)
        section_grid.append(section)

    # Create LV
    lv = localvol.InterpolatedParamLocalVol(expiry_grid, section_grid)
    return lv


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

    # # Write an example to start from
    # write_example(valdate, name, folder)

    lv = load_lv(valdate, name, folder)
    print(lv)

