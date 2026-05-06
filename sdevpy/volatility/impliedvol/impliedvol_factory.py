from pathlib import Path
import datetime as dt
from sdevpy.utilities import jsonmanager as jsm
from sdevpy.utilities import dates as dts
from sdevpy.volatility.impliedvol.impliedvol import ImpliedVol


def get_impliedvol_from_data(data: dict) -> ImpliedVol:
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
            n_dim = data['dim']
            ivol = LogMix(n_dim)
            params = data['params']
            ivol.update_params(params)

    return ivol


def get_impliedvol(name: str, date: dt.datetime, model_name: str, **kwargs) -> ImpliedVol:
    """ Retrieve ImpliedVol model knowing name, date and model name """
    file = data_file(name, date, model_name, **kwargs)

    # Check file existence
    if not file.exists():
        raise FileNotFoundError(f"Implied vol file not found: {file}")

    data = jsm.deserialize(file)
    return get_impliedvol_from_data(data)


def data_file(name: str, date: dt.datetime, model_name: str, **kwargs) -> str:
    """ Data file for implied vol models """
    folder = kwargs.get('folder', test_data_folder())
    name_folder = folder / name
    name_folder.mkdir(parents=True, exist_ok=True)
    filename = date.strftime(dts.DATE_FILE_FORMAT) + "_" + model_name + ".json"
    return name_folder / filename


def test_data_folder() -> str:
    """ Test data folder for implied vol models """
    path = Path(__file__).parent.parent.parent.parent / "datasets" / "impliedvol"
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    print(test_data_folder())
