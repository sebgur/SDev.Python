import os
from pathlib import Path
import datetime as dt
import json
from sdevpy.models import biexp, svivol
from sdevpy.models import localvol
from sdevpy.tools import timegrids
from sdevpy.maths import interpolation as itp


DATE_FORMAT = "%d-%b-%y"


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


def load_lv_from_folder(new_date, new_expiries, store_date, store_name, folder):
    file = os.path.join(folder, store_name)
    os.makedirs(file, exist_ok=True)
    file = os.path.join(file, "localvol_" + store_date.strftime("%Y%m%d-%H.%M.%S") + ".json")

    # Check file existence
    if not os.path.exists(file):
        raise FileNotFoundError(f"Local vol file not found: {file}")

    # Retrieve LV definition
    with open(file, 'r') as f:
        data = json.load(f)

    # Load LV
    lv = load_lv_from_data(new_date, new_expiries, data)
    return lv

def load_lv_from_data(new_date, new_expiries, data):
    # Create sections
    sections = data['sections']
    if sections is None:
        raise KeyError("Sections node not valid in LV data")

    # Collected stored grids
    expiry_grid = []
    # section_grid = []
    model_names = []
    parameters = []
    for section_config in sections:
        time = section_config['time']
        # section = create_section(time, section_config)
        expiry_grid.append(time)
        # section_grid.append(section)
        model_names.append(section_config['model'])
        parameters.append(section_config['params'])

    # Check section consistency
    model_name = model_names[0]
    param_names = parameters[0].keys()
    param_size = len(param_names)
    same_models = all(x == model_name for x in model_names)
    same_sizes = all(len(x.keys()) == param_size for x in parameters)

    if not same_models or not same_sizes:
        raise RuntimeError("Impossible to set LV from previous data due to inconsistent sections")

    # Collect parameter vectors
    param_vectors = {}
    for name in param_names:
        param_vectors[name] = []

    for p in parameters:
        for name in param_names:
            param_vectors[name].append(p[name])

    # Prepare new time grid
    t_grid = []
    for new_expiry in new_expiries:
        t_grid.append(timegrids.model_time(new_date, new_expiry))

    # Define interpolations
    interps = {}
    for name in param_names:
        interps[name] = itp.create_interpolation(interp='linear', l_extrap='flat', r_extrap='flat',
                                                 x_grid=expiry_grid, y_grid=param_vectors[name])

    # Interpolate
    new_section_grid = []
    for time in t_grid:
        section_config = {'time': time, 'model': model_name}
        params = {}
        for name in param_names:
            params[name] = float(interps[name].value(time))

        section_config['params'] = params
        section = create_section(time, section_config)
        new_section_grid.append(section)

    # Create LV
    print(t_grid)
    print(new_section_grid)
    lv = localvol.InterpolatedParamLocalVol(t_grid, new_section_grid)
    # lv = localvol.InterpolatedParamLocalVol(expiry_grid, section_grid)

    return lv


def write_example(date, name, folder):
    file = os.path.join(folder, name)
    os.makedirs(file, exist_ok=True)
    file = os.path.join(file, "localvol_" + valdate.strftime("%Y%m%d-%H.%M.%S") + ".json")
    sections = []
    print(file)
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
    # Get path to dataset folder
    folder = Path(__file__).parent.parent.parent.resolve()
    folder = os.path.join(os.path.join(folder, "datasets"), "localvol")
    name = "SomeIndex"
    new_date = dt.datetime(2025, 12, 15)
    new_expiries = [dt.datetime(2026, 12, 15), dt.datetime(2027, 12, 15)]

    # # Write an example to start from
    # write_example(valdate, name, folder)

    # Load
    lv = load_lv_from_folder(new_date, new_expiries, new_date, name, folder)

    # Dump
    lv_data = lv.dump_data()
    print(lv_data)
    data = {'name': "SomeIndex2", 'datetime': new_date.strftime(DATE_FORMAT),
            'sections': lv_data}
    file = os.path.join(folder, 'test.json')
    print(file)
    print(data)
    with open(file, 'w') as f:
        json.dump(data, f, indent=2)

