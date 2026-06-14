import logging
from sdevpy.volatility.impliedvol.models import biexp, vsvi, cubicvol
from sdevpy.volatility.localvol import localvol
from sdevpy.volatility.localvol.localvol import InterpolatedParamLocalVol, LocalVolSection
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


def get_localvol_new(t_grid: list[float], model_name: str) -> InterpolatedParamLocalVol:
    """ Create new local vol """
    sections = create_sections(t_grid, model_name)
    lv = localvol.InterpolatedParamLocalVol(sections)
    return lv


def get_localvol_from_data(data: dict, new_t_grid: list[float]=None) -> localvol.LocalVol:
    """ Create InterpolatedLocalVol from data """
    if data is None:
        raise ValueError('LocalVol data is None')

    # Create sections
    sections = data.get('sections', None)
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


if __name__ == "__main__":
    print("Hello")
