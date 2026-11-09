import copy
import pytest
from sdevpy.conventions.repository import (ConventionRepository, ConventionError, convention_repository,
                                           CONVENTIONS_FILE)
from sdevpy.conventions.rateconventions import IndexKind
from sdevpy.utilities.calendars import BDC
from sdevpy.utilities.daycounts import DayCount
from sdevpy.tests.conftest import staticdata_path

SOFR = {"notes": "", "currency": "USD", "kind": "OVERNIGHT", "tenor": None, "daycount": "ACT360",
        "calendar": "USGOV", "fixing_lag": 0, "publication_lag": 1, "bdc": "F", "eom": False}
EURIBOR_6M = {"notes": "", "currency": "EUR", "kind": "TERM", "tenor": "6M", "daycount": "ACT360",
              "calendar": "EUR", "fixing_lag": 2, "publication_lag": 0, "bdc": "MF", "eom": True}
BASE = {"indices": {"USD-SOFR": SOFR, "EUR-EURIBOR-6M": EURIBOR_6M}}


def modified(group: str, conv_id: str, **changes) -> dict:
    """ Copy of BASE with fields changed on one entry. A value of ... removes the field """
    data = copy.deepcopy(BASE)
    entry = data[group][conv_id]
    for key, value in changes.items():
        if value is ...:
            del entry[key]
        else:
            entry[key] = value
    return data


def test_default_repository_loads():
    repo = convention_repository()
    assert repo is convention_repository()  # loaded once
    assert 'USD-SOFR' in repo.ids('indices')


def test_index_parsing():
    repo = ConventionRepository.from_dict(BASE)
    sofr = repo.index('USD-SOFR')
    assert sofr.id == 'USD-SOFR'
    assert sofr.kind == IndexKind.OVERNIGHT and sofr.tenor is None
    assert sofr.daycount == DayCount.ACT360 and sofr.bdc == BDC.F
    euribor = repo.index('EUR-EURIBOR-6M')
    assert euribor.tenor == '6M' and euribor.fixing_lag == 2 and euribor.eom


def test_unknown_id_suggests():
    repo = ConventionRepository.from_dict(BASE)
    with pytest.raises(KeyError, match="EUR-EURIBOR-6M"):
        repo.index('EUR-EURIBOR-6N')


def test_duplicate_id(tmp_path):
    text = (staticdata_path() / CONVENTIONS_FILE).read_text()
    dup = text.replace('"indices": {', '"indices": {\n"USD-SOFR": {},', 1)
    path = tmp_path / "dup.json"
    path.write_text(dup)
    with pytest.raises(ConventionError, match="Duplicate key 'USD-SOFR'"):
        ConventionRepository.from_file(path)


@pytest.mark.parametrize('data, message', [
    ({}, "missing \\['indices'\\]"),
    ({**BASE, "swapz": {}}, "unknown \\['swapz'\\]"),
    ({"indices": {"usd-sofr": SOFR}}, "invalid id"),
    (modified('indices', 'USD-SOFR', notes=...), "missing \\['notes'\\]"),
    (modified('indices', 'USD-SOFR', pay_lg=2), "unknown \\['pay_lg'\\]"),
    (modified('indices', 'USD-SOFR', daycount="ACT/360"), "not a DayCount"),
    (modified('indices', 'USD-SOFR', fixing_lag=True), "expected int"),
    (modified('indices', 'USD-SOFR', eom=0), "expected bool"),
    (modified('indices', 'USD-SOFR', currency="usd"), "invalid currency"),
    (modified('indices', 'USD-SOFR', calendar="USD,XYZ"), "unknown calendar 'XYZ'"),
    (modified('indices', 'USD-SOFR', fixing_lag=-1), "fixing_lag: must be >= 0"),
    (modified('indices', 'USD-SOFR', tenor="1D"), "overnight index must have tenor null"),
    (modified('indices', 'EUR-EURIBOR-6M', tenor=None), "term index needs a tenor"),
    (modified('indices', 'EUR-EURIBOR-6M', tenor="6X"), "invalid tenor"),
])
def test_validation(data, message):
    with pytest.raises(ConventionError, match=message):
        ConventionRepository.from_dict(data)
