import datetime as dt
from sdevpy.utilities import jsonmanager as jsm
from sdevpy.volatility.impliedvol.impliedvol import ImpliedVol, data_file
from sdevpy.volatility.impliedvol.models.tssvi1 import TsSvi1
from sdevpy.volatility.impliedvol.models.tssvi2 import TsSvi2
from sdevpy.volatility.impliedvol.models.logmix import LogMix


def get_impliedvol_from_data(data: dict) -> ImpliedVol:
    """ Create implied vol model from its data dictionary """
    type_ = data.get('type', None)
    if type_ is None:
        raise ValueError("No type found in implied vol data")

    type_lw = type_.lower()
    match type_lw:
        case "tssvi1":
            params = data['params']
            ivol = TsSvi1()
            ivol.update_params(params)
        case "tssvi2":
            params = data['params']
            ivol = TsSvi2()
            ivol.update_params(params)
        case "logmix":
            n_mix = data['n_mix']
            ivol = LogMix(n_mix)
            params = data['params']
            ivol.update_params(params)
        case _:
            raise ValueError(f"Unknown implied vol model: {type_}")

    return ivol


def get_impliedvol(name: str, date: dt.datetime, model_name: str, **kwargs) -> ImpliedVol:
    """ Retrieve implied vol model knowing name, date and model name """
    file = data_file(name, date, model_name, **kwargs)

    # Check file existence
    if not file.exists():
        raise FileNotFoundError(f"Implied vol file not found: {file}")

    data = jsm.deserialize(file)
    return get_impliedvol_from_data(data)
