import json
from pathlib import Path
import datetime as dt
import logging
from sdevpy.volatility.impliedvol.models import biexp, vsvi, cubicvol
from sdevpy.volatility.localvol import localvol
from sdevpy.volatility.localvol.localvol import InterpolatedParamLocalVol, LocalVolSection
from sdevpy.utilities import dates
from sdevpy.maths import interpolation as itp
from sdevpy.utilities import jsonmanager as jsm
log = logging.getLogger(__name__)


############ TODO #######################################################
# * Plug in matrix local vols. Should we implement calibration on the fly?
# * Maybe we try to find a file and use it if it's there, and if it's not
#   there then we trigger the calibration?

name_model_map = {'ABC': 'BiExp', 'KLM': 'VSVI', 'XYZ': 'Matrix'}


def get_local_vols(names: list[str], valdate: dt.datetime, **kwargs) -> list[localvol.LocalVol]:
    """ Retrieve local vols assuming calibration has already been done """
    lv_map = kwargs.get('lv_map', None)
    lvs = []
    if lv_map is None: # Then read from folder
        folder = kwargs.get('folder', test_data_folder())
        for name in names:
            lvs.append(load_param_lv(valdate, name, folder=folder))
    else: # Read from map
        for name in names:
            name_lv = lv_map.get(name, None)
            if name_lv is not None:
                lvs.append(name_lv)
            else:
                raise ValueError(f"Could not find LV object in map for name: {name}")

    return lvs


def create_sections(t_grid: list[float], model: str) -> list[LocalVolSection]:
    """ Create sections along the time grid for the chosen model """
    sections = []
    for t in t_grid:
        config = {'time': t, 'model': model, 'params': None}
        section = create_section(config)
        sections.append(section)

    return sections


def create_section(config: dict) -> LocalVolSection:
    """ Section factory: create a LocalVolSection from a config dictionary """
    time = config.get('time', None)
    model = config.get('model', None)
    param_config = config.get('params', None)
    if time is None or model is None: # or param_config is None:
        raise ValueError("Invalid section input in local vol file")

    match model.lower():
        case 'biexp':
            section = biexp.create_section(time, param_config)
        case 'cubicvol':
            section = cubicvol.create_section(time, param_config)
        case 'vsvi':
            section = vsvi.create_section(time, param_config)
        case _:
            raise ValueError(f"Unknown section type: {model}")

    return section


# def load_lv_new(t_grid: list[float], model: str) -> InterpolatedParamLocalVol:
#     """ Load new LV by sections of given model on time grid """
#     sections = create_sections(t_grid, model)
#     lv = localvol.InterpolatedParamLocalVol(sections)
#     return lv


# def load_lv_from_folder(store_date: dt.datetime, name: str, folder: str,
#                         t_grid: list[float]=None, model_name: str=None) -> InterpolatedParamLocalVol:
#     """ Create InterpolatedLocalVol from folder.
#         If t_grid is specified, we interpolate the model parameters from their
#         stored grid to t_grid. If it is not specified, we use the stored grid.
#         If model_name is specified, we retrieve the file for that model. If it
#         is not specified, we get the model name from the (name x model) map.
#     """
#     file = Path(folder) / name
#     file.mkdir(parents=True, exist_ok=True)
#     store_date_str = store_date.strftime(dates.DATE_FILE_FORMAT)
#     eff_name = (name_model_map.get(name, None) if model_name is None else model_name)
#     if eff_name is None:
#         raise ValueError(f"No model name specified for name: {eff_name}")
#     file = file / (store_date_str + "." + eff_name + ".json")

#     # Check file existence
#     if not file.exists():
#         raise FileNotFoundError(f"Local vol file not found: {file}")

#     # Retrieve LV definition
#     with open(file) as f:
#         data = json.load(f)

#     # Load LV
#     lv = load_lv_from_data(data, t_grid)
#     return lv


def load_param_lv(date: dt.datetime, name: str, folder: str=None, model_name: str=None,
                  t_grid: list[float]=None) -> InterpolatedParamLocalVol:
    """ Load InterpolatedParamLocalVol for given date and name.
        If the model name is not given, we infer it from the (name, model) map.
        t_grid, if given, is used to define the time grid of the model, interpolating
        from the stored grid if a stored model is present.
        If t_grid is not given and there is a stored model, the grid of the stored model
        is used. If t_grid is not given and there is no stored model, an error is thrown.
    """
    folder = (test_data_folder() if folder is None else folder)
    model_name = (name_model_map.get(name, None) if model_name is None else model_name)
    if model_name is None:
        raise ValueError(f"No model name specified for name: {name}")

    # Look for an existing model file
    file = Path(folder) / name
    file.mkdir(parents=True, exist_ok=True)
    date_str = date.strftime(dates.DATE_FILE_FORMAT)
    file = file / (date_str + "." + model_name + ".json")
    if file.exists():
        log.info(f"Loading existing LV for {name} from file: {file}")
        lv_data = jsm.deserialize(file)
        lv = load_lv_from_data(lv_data, t_grid)
    else:
        log.info(f"Initializing new LV for {name}")
        sections = create_sections(t_grid, model_name)
        lv = localvol.InterpolatedParamLocalVol(sections)

    return lv


def load_lv_from_data(data: dict, new_t_grid: list[float]=None) -> localvol.LocalVol:
    """ Create InterpolatedLocalVol from data """
    # Create sections
    sections = data['sections']
    if sections is None:
        raise KeyError("Sections node not valid in LV data")

    if new_t_grid is None: # Just use the data
        new_section_grid = []
        for section_config in sections:
            section = create_section(section_config)
            new_section_grid.append(section)
    else:
        # Collect stored grids
        old_t_grid, model_names, parameters = [], [], []
        for section in sections:
            old_t_grid.append(section['time'])
            model_names.append(section['model'])
            parameters.append(section['params'])

        # Check section consistency
        model_name = model_names[0]
        param_names = parameters[0].keys()
        param_size = len(param_names)
        same_models = all(x == model_name for x in model_names)
        same_sizes = all(len(x.keys()) == param_size for x in parameters)

        if not same_models or not same_sizes:
            raise RuntimeError("Impossible to set LV from previous data due to inconsistent sections")

        # Collect parameter vectors per parameter name
        param_vectors = {}
        for name in param_names:
            param_vectors[name] = []

        for p in parameters:
            for name in param_names:
                param_vectors[name].append(p[name])

        # Define interpolations
        interps = {}
        for name in param_names:
            interps[name] = itp.create_interpolation(interp='linear', l_extrap='flat', r_extrap='flat',
                                                    x_grid=old_t_grid, y_grid=param_vectors[name])

        # Interpolate
        new_section_grid = []
        for time in new_t_grid:
            params = {}
            for name in param_names:
                params[name] = float(interps[name].value(time))

            section_config = {'time': time, 'model': model_name, 'params': params}
            section = create_section(section_config)
            new_section_grid.append(section)

    # Create LV
    lv = localvol.InterpolatedParamLocalVol(new_section_grid)
    return lv


def write_example(date: dt.datetime, name: str, folder: str) -> None:
    """ Write an example LocalVol to file """
    file = Path(folder) / name
    file.mkdir(parents=True, exist_ok=True)
    file = file / (date.strftime(dates.DATE_FILE_FORMAT) + ".json")
    sections = []
    print(file)
    for i in range(4):
        time = i / 4.0 * 2.5
        model = 'BiExp'
        params = {'f0': 0.5, 'fl': 0.2, 'fr': 0.2, 'taul': 0.2, 'taur': 0.2, 'fp': 0.2}
        section = {'time': time, 'model': model, 'params': params}
        sections.append(section)

    obj = {'name': name, 'datetime': date.strftime(dates.DATE_FORMAT), 'sections': sections}

    with open(file, 'w') as f:
        json.dump(obj, f, indent=2)


def test_data_folder() -> str:
    """ Test data folder for Local Vol """
    folder = Path(__file__).parent.parent.parent.parent.resolve()
    folder = folder / "datasets" / "localvol"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    name = "ABC"
    valdate = dt.datetime(2025, 12, 15)
    folder = test_data_folder()

    # # Generate a sample to start from
    # write_example(valdate, name, folder)

    # Load
    store_date = valdate
    # new_date = valdate
    new_expiries = None # [1.0, 2.0]# [dt.datetime(2026, 12, 15), dt.datetime(2027, 12, 15)]
    lv = load_param_lv(store_date, name, folder=folder, t_grid=new_expiries)
    lv.name = name
    lv.valdate = valdate

    # Dump
    lv_data = lv.dump_data()
    print(lv_data)
    data = lv_data
    # data = {'valdate': valdate.strftime(dates.DATE_FORMAT),
    #         'snapdate': valdate.strftime(dates.DATETIME_FORMAT),
    #         'sections': lv_data}
    file = Path(folder) / 'test_dump.json'
    print(file)
    print(data)
    with open(file, 'w') as f:
        json.dump(data, f, indent=2)

    # View
    expiries = lv.t_grid
    n_expiries = len(expiries)
    x = np.linspace(-0.5, 0.5, 100)
    exp_idx = n_expiries - 1
    lparams = lv.params(exp_idx)
    print(lparams)
    lvols = lv.value(expiries[exp_idx], x)

    plt.plot(x, lvols)
    plt.show()
