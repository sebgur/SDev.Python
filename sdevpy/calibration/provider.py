import datetime as dt
import logging
from typing import Protocol, runtime_checkable
from sdevpy.volatility.impliedvol import impliedvol as iv_mod
from sdevpy.volatility.impliedvol import impliedvol_factory as ivf
from sdevpy.volatility.localvol import localvol as lv_mod
from sdevpy.volatility.localvol import localvol_factory as lvf
log = logging.getLogger(__name__)


@runtime_checkable
class CalibrationDataProvider(Protocol):
    def get_impliedvol_data(self, name: str, date: dt.datetime, model_name: str) -> dict|None: ...
    def get_localvol_data(self, name: str, date: dt.datetime, model_name: str) -> dict|None: ...


def get_impliedvol(name: str, date: dt.datetime, model_name: str,
                   cal_prov: CalibrationDataProvider) -> iv_mod.ImpliedVol:
    """ Retrieve implied vol knowing name, date and model name """
    data = cal_prov.get_impliedvol_data(name, date, model_name)
    return ivf.get_impliedvol_from_data(data)


def get_localvol(name: str, date: dt.datetime, model_name: str, cal_prov: CalibrationDataProvider,
                 t_grid: list[float]=None) -> lv_mod.LocalVol:
    """ Retrieve local vol knowing name, date and model name """
    data = cal_prov.get_localvol_data(name, date, model_name)
    return lvf.get_localvol_from_data(data, t_grid)


name_model_map = {'ABC': 'BiExp', 'KLM': 'VSVI', 'XYZ': 'Matrix'}


def get_localvol_or_new(name: str, date: dt.datetime, model_name: str, cal_prov: CalibrationDataProvider,
                        t_grid: list[float]=None, force_new: bool=False) -> lv_mod.LocalVol:
    """ Retrieve local vol for given name, date and model name.
        If the model name is None, we infer it from the (name, model) map.
        t_grid, if given, is used to define the time grid of the model, interpolating
        from the stored grid if a stored model is present.
        If t_grid is not given and there is a stored model, the grid of the stored model
        is used. If t_grid is not given and there is no stored model, an error is thrown.
        Args:
            - force_new: return new local vol even if there is an existing one
    """
    model_name = (name_model_map.get(name, None) if model_name is None else model_name)
    if model_name is None:
        raise ValueError(f"No model name specified for name: {name}")

    # Look for an existing model file
    lv = None
    if not force_new: # Try to get it from calibration provider, None if absent
        lv = get_localvol(name, date, model_name, cal_prov, t_grid=t_grid)

    if lv is None:
        log.info(f"Initializing new LV for {name}")
        lv = lvf.get_localvol_new(t_grid, model_name)

    return lv


def get_local_vols(names: list[str], valdate: dt.datetime, cal_prov: CalibrationDataProvider,
                   **kwargs) -> list[lv_mod.LocalVol]:
    """ Retrieve local vols assuming calibration has already been done """
    lv_map = kwargs.get('lv_map', None)
    lvs = []
    if lv_map is None: # Get from CalibrationDataProvider
        model_name = kwargs.get('model_name', None)
        for name in names:
            lvs.append(get_localvol_or_new(name, valdate, model_name, cal_prov))
    else: # Read from map
        for name in names:
            name_lv = lv_map.get(name, None)
            if name_lv is not None:
                lvs.append(name_lv)
            else:
                raise ValueError(f"Could not find LV object in map for name: {name}")

    return lvs


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
