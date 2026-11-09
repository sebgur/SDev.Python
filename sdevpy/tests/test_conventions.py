import copy
import datetime as dt
import numpy as np
import pytest
from sdevpy.conventions.repository import (ConventionRepository, ConventionError, convention_repository,
                                           CONVENTIONS_FILE)
from sdevpy.conventions.rateconventions import IndexKind, SwapKind, RateMethod, FutureDateRule
from sdevpy.utilities.calendars import BDC
from sdevpy.utilities.daycounts import DayCount
from sdevpy.utilities import schedules as sch
from sdevpy.tests.conftest import staticdata_path

SOFR = {"notes": "", "currency": "USD", "kind": "OVERNIGHT", "tenor": None, "daycount": "ACT360",
        "calendar": "USGOV", "fixing_lag": 0, "publication_lag": 1, "bdc": "F", "eom": False}
EURIBOR_6M = {"notes": "", "currency": "EUR", "kind": "TERM", "tenor": "6M", "daycount": "ACT360",
              "calendar": "EUR", "fixing_lag": 2, "publication_lag": 0, "bdc": "MF", "eom": True}
SOFR_LEG = {"frequency": "1Y", "daycount": "ACT360", "calendar": "USGOV", "bdc": "MF", "adjust_accrual": True,
            "pay_lag": 2, "fixing_lag": None, "fixing_calendar": None, "eom": False}
SOFR_OIS = {"notes": "", "kind": "OIS", "currency": "USD", "float_index": "USD-SOFR", "fixed_leg": SOFR_LEG,
            "float_leg": dict(SOFR_LEG), "spot_lag": 2, "spot_calendar": None}
EUR_FIXED_LEG = {"frequency": "1Y", "daycount": "D30_360", "calendar": "EUR", "bdc": "MF", "adjust_accrual": True,
                 "pay_lag": 0, "fixing_lag": None, "fixing_calendar": None, "eom": False}
EURIBOR_6M_LEG = {"frequency": "6M", "daycount": "ACT360", "calendar": "EUR", "bdc": "MF", "adjust_accrual": True,
                  "pay_lag": 0, "fixing_lag": 2, "fixing_calendar": None, "eom": False}
EURIBOR_6M_IRS = {"notes": "", "kind": "IRS", "currency": "EUR", "float_index": "EUR-EURIBOR-6M",
                  "fixed_leg": EUR_FIXED_LEG, "float_leg": EURIBOR_6M_LEG, "spot_lag": 2, "spot_calendar": None}
USD_DEPO = {"notes": "", "currency": "USD", "daycount": "ACT360", "calendar": "USD", "spot_lag": 2, "bdc": "MF",
            "eom": True}
EURIBOR_6M_FRA = {"notes": "", "currency": "EUR", "index": "EUR-EURIBOR-6M", "spot_lag": 2}
SOFR_3M_FUT = {"notes": "", "currency": "USD", "index": "USD-SOFR", "rate_method": "COMPOUNDED", "period": "3M",
               "date_rule": "IMM", "exchange": "CME", "notional": 1000000}
SOFR_1M_FUT = {"notes": "", "currency": "USD", "index": "USD-SOFR", "rate_method": "AVERAGED", "period": "1M",
               "date_rule": "MONTH", "exchange": "CME", "notional": 5000000}
EURIBOR_6M_FUT = {"notes": "", "currency": "EUR", "index": "EUR-EURIBOR-6M", "rate_method": "TERM",
                  "period": "6M", "date_rule": "IMM", "exchange": "ICE", "notional": 1000000}
BASE = {"indices": {"USD-SOFR": SOFR, "EUR-EURIBOR-6M": EURIBOR_6M},
        "deposits": {"USD-DEPO": USD_DEPO},
        "fras": {"EUR-EURIBOR-6M-FRA": EURIBOR_6M_FRA},
        "futures": {"USD-SOFR-3M-FUT": SOFR_3M_FUT, "USD-SOFR-1M-FUT": SOFR_1M_FUT,
                    "EUR-EURIBOR-6M-FUT": EURIBOR_6M_FUT},
        "swaps": {"USD-SOFR-OIS": SOFR_OIS, "EUR-EURIBOR-6M-IRS": EURIBOR_6M_IRS}}


def modified(group: str, conv_id: str, **changes) -> dict:
    """ Copy of BASE with fields changed on one entry. A value of ... removes the field.
        A dict value on a dict field (e.g. a leg) updates that field instead of replacing it """
    data = copy.deepcopy(BASE)
    _apply(data[group][conv_id], changes)
    return data


def _apply(entry: dict, changes: dict) -> None:
    for key, value in changes.items():
        if value is ...:
            del entry[key]
        elif isinstance(value, dict) and isinstance(entry.get(key), dict):
            _apply(entry[key], value)
        else:
            entry[key] = value


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
    ({}, "missing \\['deposits', 'fras', 'futures', 'indices', 'swaps'\\]"),
    ({**BASE, "swapz": {}}, "unknown \\['swapz'\\]"),
    ({**BASE, "indices": {"usd-sofr": SOFR}}, "invalid id"),
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


def test_swap_parsing():
    repo = ConventionRepository.from_dict(BASE)
    ois = repo.swap('USD-SOFR-OIS')
    assert ois.kind == SwapKind.OIS and ois.float_index == 'USD-SOFR'
    assert repo.index(ois.float_index).kind == IndexKind.OVERNIGHT
    sofr_leg = sch.LegConvention('1Y', DayCount.ACT360, 'USGOV', pay_lag=2)
    assert ois.fixed_leg == sofr_leg and ois.float_leg == sofr_leg
    assert ois.schedule_convention == sch.SwapConvention(sofr_leg, sofr_leg, spot_lag=2)


def test_swap_schedule_from_repository():
    """ Same schedule as the hand-built EUR 6M convention used in test_swaps """
    eur_6m = sch.SwapConvention(sch.LegConvention('1Y', DayCount.D30_360, 'EUR'),
                                sch.LegConvention('6M', DayCount.ACT360, 'EUR', fixing_lag=2))
    conv = convention_repository().swap('EUR-EURIBOR-6M-IRS').schedule_convention
    assert conv == eur_6m
    trade_date = dt.date(2026, 10, 23)
    expected = sch.make_spot_swap_scheduler(trade_date, '5Y', eur_6m)
    actual = sch.make_spot_swap_scheduler(trade_date, '5Y', conv)
    assert np.array_equal(actual.floating.fixing, expected.floating.fixing)
    assert np.array_equal(actual.fixed.payment, expected.fixed.payment)


@pytest.mark.parametrize('data, message', [
    (modified('swaps', 'USD-SOFR-OIS', float_index="USD-SOFRR"), "float_index: no 'USD-SOFRR' in indices"),
    (modified('swaps', 'USD-SOFR-OIS', currency="EUR"), "currency EUR but index USD-SOFR is USD"),
    (modified('swaps', 'USD-SOFR-OIS', kind="IRS"), "float_index: needs a TERM index"),
    (modified('swaps', 'EUR-EURIBOR-6M-IRS', kind="OIS"), "float_index: needs a OVERNIGHT index"),
    (modified('swaps', 'EUR-EURIBOR-6M-IRS', float_leg={"frequency": "3M"}), "does not match index tenor 6M"),
    (modified('swaps', 'USD-SOFR-OIS', fixed_leg={"frequency": "5M"}), "fixed_leg.frequency: invalid"),
    (modified('swaps', 'USD-SOFR-OIS', fixed_leg={"bdc": "XX"}), "fixed_leg.bdc: 'XX' is not a BDC"),
    (modified('swaps', 'USD-SOFR-OIS', float_leg={"eom": ...}), "float_leg: missing \\['eom'\\]"),
    (modified('swaps', 'USD-SOFR-OIS', float_leg={"calendar": "XYZ"}), "float_leg.calendar: unknown calendar"),
    (modified('swaps', 'USD-SOFR-OIS', float_leg={"pay_lag": -1}), "float_leg.pay_lag: must be >= 0"),
    (modified('swaps', 'USD-SOFR-OIS', spot_calendar="XYZ"), "spot_calendar: unknown calendar"),
    (modified('swaps', 'USD-SOFR-OIS', spot_lag=-2), "spot_lag: must be >= 0"),
])
def test_swap_validation(data, message):
    with pytest.raises(ConventionError, match=message):
        ConventionRepository.from_dict(data)


def test_curve_instrument_parsing():
    repo = ConventionRepository.from_dict(BASE)
    depo = repo.deposit('USD-DEPO')
    assert depo.daycount == DayCount.ACT360 and depo.spot_lag == 2 and depo.bdc == BDC.MF
    fra = repo.fra('EUR-EURIBOR-6M-FRA')
    assert repo.index(fra.index).tenor == '6M'
    sr3 = repo.future('USD-SOFR-3M-FUT')
    assert sr3.rate_method == RateMethod.COMPOUNDED and sr3.date_rule == FutureDateRule.IMM
    assert isinstance(sr3.notional, float) and sr3.notional == 1.0e6


def test_default_repository_groups():
    repo = convention_repository()
    for group in ('indices', 'deposits', 'fras', 'futures', 'swaps'):
        assert repo.ids(group)


@pytest.mark.parametrize('data, message', [
    (modified('deposits', 'USD-DEPO', calendar="XYZ"), "deposits.USD-DEPO: unknown calendar"),
    (modified('deposits', 'USD-DEPO', spot_lag=-1), "spot_lag: must be >= 0"),
    (modified('fras', 'EUR-EURIBOR-6M-FRA', index="USD-SOFR"), "currency EUR but index USD-SOFR is USD"),
    (modified('fras', 'EUR-EURIBOR-6M-FRA', currency="USD", index="USD-SOFR"), "index: needs a TERM index"),
    (modified('fras', 'EUR-EURIBOR-6M-FRA', index="EUR-EURIBOR-3M"), "index: no 'EUR-EURIBOR-3M' in indices"),
    (modified('futures', 'USD-SOFR-3M-FUT', rate_method="TERM"), "index: needs a TERM index"),
    (modified('futures', 'EUR-EURIBOR-6M-FUT', rate_method="COMPOUNDED"), "index: needs a OVERNIGHT index"),
    (modified('futures', 'EUR-EURIBOR-6M-FUT', period="3M"), "period: 3M does not match index tenor 6M"),
    (modified('futures', 'USD-SOFR-1M-FUT', period="3M"), "MONTH rule needs a 1M period"),
    (modified('futures', 'USD-SOFR-3M-FUT', period="3X"), "period: invalid tenor"),
    (modified('futures', 'USD-SOFR-3M-FUT', date_rule="SERIAL"), "'SERIAL' is not a FutureDateRule"),
    (modified('futures', 'USD-SOFR-3M-FUT', exchange=""), "exchange: must not be empty"),
    (modified('futures', 'USD-SOFR-3M-FUT', notional=0), "notional: must be > 0"),
    (modified('futures', 'USD-SOFR-3M-FUT', notional="1e6"), "notional: expected a number"),
])
def test_curve_instrument_validation(data, message):
    with pytest.raises(ConventionError, match=message):
        ConventionRepository.from_dict(data)
