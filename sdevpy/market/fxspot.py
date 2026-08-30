""" Conventions and utilities for the FX market """


USD_IS_QUOTE = {'EUR', 'GBP', 'AUD', 'NZD', 'XAU', 'XAG', 'XPT', 'XPD'} # ZZZ/USD convention
USD = "USD"

# Premium-currency seniority hierarchy (Reiswich & Wystup, "FX Volatility Smile Construction",
# CPQF Working Paper No. 20, 2009/2010, Table 2/eq. 14): within a pair, the option premium is
# conventionally denominated in whichever currency ranks higher here. Currencies not listed fall
# in an unranked tail. Only comparisons against a currency ABOVE them in this list are reliable.
# The paper itself notes "exceptions may occur, so in case of doubt it is advisable to check."
PREMIUM_CCY_TIERS = [
    {'USD'}, {'EUR'}, {'GBP'}, {'AUD'}, {'NZD'}, {'CAD'}, {'CHF'},
    {'NOK', 'SEK', 'DKK'}, {'CZK', 'PLN', 'TRY', 'MXN'}, {'JPY'},
]

# Currency pair naming seniority. Given two currencies, the one with the lower score is quoted
# first. 0 = not yet verified either way -- two unranked currencies can't be resolved.
SENIOR_TIERS = [{'EUR'}, {'GBP'}, {'AUD'}, {'NZD'}, {'USD'}]  # most to least senior -- beats
                                                              # any unranked currency
JUNIOR_TIERS = [{'JPY'}]  # least to most junior -- loses to any unranked currency


# # Pair-naming seniority: the higher-ranked currency is listed first (e.g. EUR ranks above GBP,
# # so EUR-GBP not GBP-EUR). Verified top tier only -- extend below USD as you encounter real
# # pairs, sourced from your actual data vendor's naming, not a general reference; there is no
# # single authoritative total order for the full currency universe.
# PAIR_NAMING_TIERS = [
#     {'EUR'}, {'GBP'}, {'AUD'}, {'NZD'}, {'USD'}, {'JPY'}
#     # Extend below USD only against a verified pair
# ]


def parse_fx_pair(name: str) -> tuple[str, str]:
    """ Parse a 6-letter FX pair name into (forccy, domccy), e.g. EURUSD -> (EUR, USD) """
    if len(name) != 6:
        raise ValueError(f"Expected a 6-letter FX pair name, got: {name}")

    return name[:3], name[3:]


def usd_conventional_pair(ccy: str) -> tuple[str, str]:
    """ Market-conventional (forccy, domccy) ordering for ccy quoted against USD """
    if ccy in USD_IS_QUOTE:
        return ccy, USD
    elif ccy == USD:
        raise ValueError("Requested USD conventional pair for USD")
    else:
        return USD, ccy


def is_inverted_quote(ccy: str) -> bool:
    """ True if ccy is conventionally quoted as USD/ccy (USD is base), i.e. the quote is
        "inverted" relative to the ccy/USD convention used by EUR, GBP, AUD, NZD. """
    forccy, _ = usd_conventional_pair(ccy)
    return forccy == USD


def premium_currency(forccy: str, domccy: str) -> str:
    """ Conventional premium currency for the pair, per the hierarchy. Raises if neither
        currency is ranked, or if both fall in the same (unordered) tier. """
    for tier in PREMIUM_CCY_TIERS:
        if forccy in tier and domccy not in tier:
            return forccy
        if domccy in tier and forccy not in tier:
            return domccy
    raise ValueError(f"Cannot determine premium currency for {forccy}/{domccy} from the "
                     f"documented hierarchy -- verify manually against a primary source")


def is_premium_adjusted(forccy: str, domccy: str) -> bool:
    """ True iff forccy/domccy conventionally uses premium-adjusted delta. Holds exactly when
        the premium currency is the foreign (base) side of the pair. Verified against all 12
        example pairs in Reiswich & Wystup (2010), Table 2. """
    return premium_currency(forccy, domccy) == forccy


# def conventional_pair_name(ccy1: str, ccy2: str) -> tuple[str, str]:
#     """ (forccy, domccy) in conventional naming order, for the currencies covered by
#         PAIR_NAMING_TIERS. Raises for anything not yet verified, rather than guess. """
#     for tier in PAIR_NAMING_TIERS:
#         if ccy1 in tier and ccy2 not in tier:
#             return ccy1, ccy2
#         if ccy2 in tier and ccy1 not in tier:
#             return ccy2, ccy1
#     raise ValueError(f"Naming order for {ccy1}/{ccy2} not yet verified -- check your data "
#                      f"vendor's convention and add it to PAIR_NAMING_TIERS")


def _seniority_score(ccy: str) -> int:
    for i, tier in enumerate(SENIOR_TIERS):
        if ccy in tier:
            return i - len(SENIOR_TIERS)  # EUR=-5, GBP=-4, ..., USD=-1
    for i, tier in enumerate(JUNIOR_TIERS):
        if ccy in tier:
            return i + 1  # JPY=+1
    return 0


def conventional_pair_name(ccy1: str, ccy2: str) -> tuple[str, str]:
    """ (forccy, domccy) in conventional naming order. Raises if neither currency has a
        verified rank -- e.g. two unranked currencies, which likely don't trade as a direct
        named pair at all (route through fxspot's USD triangulation instead). """
    s1, s2 = _seniority_score(ccy1), _seniority_score(ccy2)
    if s1 == s2 == 0:
        raise ValueError(f"Naming order for {ccy1}/{ccy2} not yet verified -- check your data "
                         f"vendor's convention and add an entry to SENIOR_TIERS/JUNIOR_TIERS")
    return (ccy1, ccy2) if s1 < s2 else (ccy2, ccy1)


if __name__ == "__main__":
    ccy1, ccy2 = "JPY", "USD"

    print(f"Given order: {ccy1}/{ccy2}")
    forccy, domccy = conventional_pair_name(ccy1, ccy2)
    print(f"Conventional: {forccy}/{domccy}")
    print(f"Premium ajusted: {is_premium_adjusted(forccy, domccy)}")
