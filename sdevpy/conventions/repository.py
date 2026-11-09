""" Convention repository: loads all instrument conventions once from a single JSON file, validates them
    strictly (stops at the first error) and serves them by id. Objects are immutable and refer to each other
    by id. The JSON file is grouped by type, then keyed by id:
        {"indices": {"USD-SOFR": {...}, ...}, ...} """
import json
import re
import types
import typing
from dataclasses import fields, is_dataclass
from difflib import get_close_matches
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any
from sdevpy.datapaths import staticdata_path
from sdevpy.utilities import dates as dts
from sdevpy.utilities.calendars import CCY_CALENDARS, MCAL_ALIASES, list_mcal_calendars
from sdevpy.utilities.schedules import LegConvention, coupons_per_year
from sdevpy.conventions.rateconventions import IndexConvention, IndexKind, FixFloatSwapConvention, SwapKind

CONVENTIONS_FILE = "conventions.json"
_ID_PATTERN = re.compile(r"[A-Z0-9]+(-[A-Z0-9]+)*")
_CCY_PATTERN = re.compile(r"[A-Z]{3}")

# JSON group name -> convention type. Ids are unique within a group
GROUPS: dict[str, type] = {
    'indices': IndexConvention,
    'swaps': FixFloatSwapConvention,
}

# Float index kind required by each swap kind
_SWAP_INDEX_KIND = {SwapKind.IRS: IndexKind.TERM, SwapKind.OIS: IndexKind.OVERNIGHT}


class ConventionError(ValueError):
    """ Invalid convention data """


class ConventionRepository:
    def __init__(self, tables: dict[str, dict[str, Any]]):
        self._tables = tables

    @classmethod
    def from_file(cls, path: Path) -> "ConventionRepository":
        with open(path, encoding='utf-8') as f:
            data = json.load(f, object_pairs_hook=_no_duplicate_keys)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "ConventionRepository":
        """ Parse and validate raw data. Raises ConventionError at the first problem """
        _check_keys(data, set(GROUPS), "<root>")
        tables = {}
        for group, conv_type in GROUPS.items():
            entries = data[group]
            if not isinstance(entries, dict):
                raise ConventionError(f"{group}: expected an object keyed by id")
            tables[group] = {conv_id: _parse_entry(conv_type, conv_id, raw, f"{group}.{conv_id}")
                             for conv_id, raw in entries.items()}
        repo = cls(tables)
        repo._validate()
        return repo

    def index(self, conv_id: str) -> IndexConvention:
        return self._get('indices', conv_id)

    def swap(self, conv_id: str) -> FixFloatSwapConvention:
        return self._get('swaps', conv_id)

    def ids(self, group: str) -> list[str]:
        return sorted(self._table(group))

    def _table(self, group: str) -> dict[str, Any]:
        if group not in self._tables:
            raise KeyError(f"Unknown convention group '{group}', expected one of {sorted(self._tables)}")
        return self._tables[group]

    def _get(self, group: str, conv_id: str) -> Any:
        table = self._table(group)
        try:
            return table[conv_id]
        except KeyError:
            close = get_close_matches(conv_id.upper(), table, n=3)
            hint = f" Did you mean {close}?" if close else ""
            raise KeyError(f"No convention '{conv_id}' in {group}.{hint}") from None

    def _validate(self) -> None:
        """ Business rules and cross-references """
        for conv in self._tables['indices'].values():
            _check_index(conv, f"indices.{conv.id}")
        for conv in self._tables['swaps'].values():
            self._check_swap(conv, f"swaps.{conv.id}")

    def _check_swap(self, conv: FixFloatSwapConvention, path: str) -> None:
        _check_currency(conv.currency, path)
        _check_non_negative(conv, ('spot_lag',), path)
        if conv.spot_calendar is not None:
            _check_calendar(conv.spot_calendar, f"{path}.spot_calendar")
        _check_leg(conv.fixed_leg, f"{path}.fixed_leg")
        _check_leg(conv.float_leg, f"{path}.float_leg")

        index = self._reference('indices', conv.float_index, f"{path}.float_index")
        if index.currency != conv.currency:
            raise ConventionError(f"{path}: currency {conv.currency} but index {index.id} is {index.currency}")
        if index.kind != _SWAP_INDEX_KIND[conv.kind]:
            raise ConventionError(f"{path}: {conv.kind.name} needs a {_SWAP_INDEX_KIND[conv.kind].name} index, "
                                  f"{index.id} is {index.kind.name}")
        if conv.kind == SwapKind.IRS and dts.period(conv.float_leg.frequency) != dts.period(index.tenor):
            raise ConventionError(f"{path}.float_leg.frequency: {conv.float_leg.frequency} does not match "
                                  f"index tenor {index.tenor}")

    def _reference(self, group: str, conv_id: str, path: str) -> Any:
        """ Referenced convention, which must exist """
        if conv_id not in self._tables[group]:
            raise ConventionError(f"{path}: no '{conv_id}' in {group}")
        return self._tables[group][conv_id]


@lru_cache(maxsize=1)
def convention_repository() -> ConventionRepository:
    """ Default repository, loaded once from the static data folder """
    return ConventionRepository.from_file(staticdata_path() / CONVENTIONS_FILE)


################## PARSING ########################################################################################
def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """ json.load silently keeps the last of duplicate keys: refuse them instead """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConventionError(f"Duplicate key '{key}'")
        result[key] = value
    return result


def _check_keys(raw: Any, expected: set[str], path: str) -> None:
    """ Explicit fields: every expected key present, no other key """
    if not isinstance(raw, dict):
        raise ConventionError(f"{path}: expected an object")
    missing = sorted(expected - raw.keys())
    if missing:
        raise ConventionError(f"{path}: missing {missing}")
    unknown = sorted(raw.keys() - expected)
    if unknown:
        raise ConventionError(f"{path}: unknown {unknown}")


def _parse_entry(conv_type: type, conv_id: str, raw: Any, path: str) -> Any:
    """ Top-level entry: the id is the JSON key, not a field in the entry """
    if not _ID_PATTERN.fullmatch(conv_id):
        raise ConventionError(f"{path}: invalid id, expected upper case words separated by '-'")
    return _parse_dataclass(conv_type, raw, path, id=conv_id)


def _parse_dataclass(cls: type, raw: Any, path: str, **given: Any) -> Any:
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls) if f.name not in given]
    _check_keys(raw, set(names), path)
    values = {name: _convert(raw[name], hints[name], f"{path}.{name}") for name in names}
    return cls(**given, **values)


def _convert(value: Any, tp: Any, path: str) -> Any:
    if isinstance(tp, types.UnionType):  # only X | None is supported
        if value is None and types.NoneType in tp.__args__:
            return None
        inner = [a for a in tp.__args__ if a is not types.NoneType]
        if len(inner) != 1:
            raise TypeError(f"{path}: unsupported field type {tp}")
        return _convert(value, inner[0], path)

    if is_dataclass(tp):
        return _parse_dataclass(tp, value, path)

    if isinstance(tp, type) and issubclass(tp, Enum):
        if not isinstance(value, str) or value not in tp.__members__:
            raise ConventionError(f"{path}: {value!r} is not a {tp.__name__}, expected one of "
                                  f"{list(tp.__members__)}")
        return tp[value]

    if tp is float:  # accept JSON integers, but not booleans
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConventionError(f"{path}: expected a number, got {value!r}")
        return float(value)

    if tp in (str, int, bool):
        if type(value) is not tp:  # strict: no bool for int, no int for bool
            raise ConventionError(f"{path}: expected {tp.__name__}, got {value!r}")
        return value

    raise TypeError(f"{path}: unsupported field type {tp}")


################## CHECKS #########################################################################################
def _check_index(conv: IndexConvention, path: str) -> None:
    _check_currency(conv.currency, path)
    _check_calendar(conv.calendar, path)
    _check_non_negative(conv, ('fixing_lag', 'publication_lag'), path)
    if conv.kind == IndexKind.OVERNIGHT and conv.tenor is not None:
        raise ConventionError(f"{path}: overnight index must have tenor null")
    if conv.kind == IndexKind.TERM:
        if conv.tenor is None:
            raise ConventionError(f"{path}: term index needs a tenor")
        _check_tenor(conv.tenor, f"{path}.tenor")


def _check_leg(leg: LegConvention, path: str) -> None:
    try:
        coupons_per_year(leg.frequency)
    except ValueError:
        raise ConventionError(f"{path}.frequency: invalid frequency '{leg.frequency}'") from None
    _check_calendar(leg.calendar, f"{path}.calendar")
    if leg.fixing_calendar is not None:
        _check_calendar(leg.fixing_calendar, f"{path}.fixing_calendar")
    _check_non_negative(leg, ('pay_lag',), path)
    if leg.fixing_lag is not None:
        _check_non_negative(leg, ('fixing_lag',), path)

@lru_cache(maxsize=1)
def _known_calendars() -> frozenset[str]:
    return frozenset(CCY_CALENDARS) | frozenset(MCAL_ALIASES) | frozenset(list_mcal_calendars())


def _check_calendar(name: str, path: str) -> None:
    """ Joint calendars are comma-separated, e.g. 'EUR,GBP' """
    for part in name.split(','):
        if part not in _known_calendars():
            raise ConventionError(f"{path}: unknown calendar '{part}'")


def _check_currency(ccy: str, path: str) -> None:
    if not _CCY_PATTERN.fullmatch(ccy):
        raise ConventionError(f"{path}: invalid currency '{ccy}'")


def _check_tenor(tenor: str, path: str) -> None:
    try:
        dts.period(tenor)
    except ValueError:
        raise ConventionError(f"{path}: invalid tenor '{tenor}'") from None


def _check_non_negative(conv: Any, names: tuple[str, ...], path: str) -> None:
    for name in names:
        if getattr(conv, name) < 0:
            raise ConventionError(f"{path}.{name}: must be >= 0")
