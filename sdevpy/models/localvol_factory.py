import os
import datetime as dt
import json
from sdevpy.models import biexp, svivol
from sdevpy.models import localvol


DATE_FORMAT = "%d-%b-%y"

# * Test load
# * Work out the question of the time: should we use
#   them as is? But they'll slightly change every day. Or we could
#   put the tenors instead? Or both? What happens if a tenor is not found?


def create_section(time, config):
    model = config.get('model', None)
    param_config = config.get('params', None)
    if model is None or param_config is None:
        raise TypeError("Invalid section input in local vol file")

    match model.lower():
        case 'biexp':
            section = biexp.create_section(time, param_config)
        case 'svivol':
            section = svivol.create_section(time, param_config)
        case _:
            section = None
            raise TypeError(f"Unknown section type: {model}")

    return section


def load_lv_from_folder(date, name, folder):
    file = os.path.join(folder, name)
    os.makedirs(file, exist_ok=True)
    file = os.path.join(file, "localvol_" + valdate.strftime("%Y%m%d-%H.%M.%S") + ".json")

    # Check file existence
    if not os.path.exists(file):
        raise FileNotFoundError(f"Local vol file not found: {file}")

    # Retrieve LV definition
    with open(file, 'r') as f:
        data = json.load(f)

    # Load LV
    lv = load_lv_from_data(data)
    return lv

def load_lv_from_data(data):
    # Create sections
    sections = data['sections']
    if sections is None:
        raise KeyError("Sections node not valid in LV data")

    expiry_grid = []
    section_grid = []
    for section_config in sections:
        time = section_config['time']
        section = create_section(time, section_config)
        expiry_grid.append(time)
        section_grid.append(section)

    # Create LV
    lv = localvol.InterpolatedParamLocalVol(expiry_grid, section_grid)

    return lv


def write_example(date, name, folder):
    file = os.path.join(folder, name)
    os.makedirs(file, exist_ok=True)
    file = os.path.join(file, "localvol_" + valdate.strftime("%Y%m%d-%H.%M.%S") + ".json")
    sections = []
    for i in range(4):
        time = i / 4.0 * 2.5
        model = 'BiExp'
        params = {'f0': 0.5, 'fl': 0.2, 'fr': 0.2, 'taul': 0.2, 'taur': 0.2, 'fp': 0.2}
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

    lv = load_lv_from_folder(valdate, name, folder)
    print(lv)

    lv_data = lv.dump_data()
    data = {'name': "SomeIndex2", 'datetime': valdate.strftime(DATE_FORMAT),
            'sections': lv_data}
    file = os.path.join(folder, 'test.json')
    with open(file, 'w') as f:
        json.dump(data, f, indent=2)

