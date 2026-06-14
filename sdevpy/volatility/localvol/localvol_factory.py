from pathlib import Path
import datetime as dt
import logging
from sdevpy.volatility.impliedvol.models import biexp, vsvi, cubicvol
from sdevpy.volatility.localvol import localvol
from sdevpy.volatility.localvol.localvol import InterpolatedParamLocalVol, LocalVolSection
from sdevpy.utilities import dates as dts
from sdevpy.maths import interpolation as itp
log = logging.getLogger(__name__)



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
    if time is None or model is None:
        raise ValueError("Invalid section input in local vol file")

    match model.lower():
        case 'biexp':
            section = biexp.create_section(time, param_config)
        case 'cubicvol':
            section = cubicvol.create_section(time, param_config)
        case 'vsvi':
            section = vsvi.create_section(time, param_config)
        case 'cubicspline' | 'linear':
            section = localvol.create_interpolated_section(time, config)
        case _:
            raise ValueError(f"Unknown section type: {model}")

    return section


# def load_param_lv(name: str, date: dt.datetime, cal_prov: CalibrationDataProvider, model_name: str=None,
#                   t_grid: list[float]=None, force_new: bool=False) -> InterpolatedParamLocalVol:
#     """ Load InterpolatedParamLocalVol for given date and name.
#         If the model name is not given, we infer it from the (name, model) map.
#         t_grid, if given, is used to define the time grid of the model, interpolating
#         from the stored grid if a stored model is present.
#         If t_grid is not given and there is a stored model, the grid of the stored model
#         is used. If t_grid is not given and there is no stored model, an error is thrown.
#         Args:
#             - force_new: force new initialization even if a suitable file exists
#     """
#     model_name = (name_model_map.get(name, None) if model_name is None else model_name)
#     if model_name is None:
#         raise ValueError(f"No model name specified for name: {name}")

#     # Look for an existing model file
#     lv = None
#     if not force_new: # Try to get it from calibration provider, None if absent
#         lv = cal_prov.get_localvol(name, date, model_name)

#     if lv is None:
#         log.info(f"Initializing new LV for {name}")
#         lv = get_localvol_new(t_grid, model_name)
#         # sections = create_sections(t_grid, model_name)
#         # lv = localvol.InterpolatedParamLocalVol(sections)

#     # date_str = date.strftime(dts.DATE_FILE_FORMAT)
#     # file = Path(folder) / name / (date_str + "." + model_name + ".json")
#     # if file.exists() and not force_new:
#     #     log.info(f"Loading existing LV for {name} from file: {file}")
#     #     lv_data = jsm.deserialize(file)
#     #     lv = get_localvol_from_data(lv_data, t_grid)
#     # else:
#     #     log.info(f"Initializing new LV for {name}")
#     #     sections = create_sections(t_grid, model_name)
#     #     lv = localvol.InterpolatedParamLocalVol(sections)

#     return lv


def get_localvol_new(t_grid: list[float], model_name: str) -> InterpolatedParamLocalVol:
    """ Create new local vol """
    sections = create_sections(t_grid, model_name)
    lv = localvol.InterpolatedParamLocalVol(sections)
    return lv


def get_localvol_from_data(data: dict, new_t_grid: list[float]=None) -> localvol.LocalVol:
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


def data_file(name: str, date: dt.datetime, model_name: str, folder: str|Path) -> Path:
    """ Retrieve data file for local vol models """
    return Path(folder) / name / (date.strftime(dts.DATE_FILE_FORMAT) + "." + model_name + ".json")


if __name__ == "__main__":
    print("Hello")
