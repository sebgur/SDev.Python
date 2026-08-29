

USD_IS_QUOTE = {'EUR', 'GBP', 'AUD', 'NZD', 'XAU', 'XAG', 'XPT', 'XPD'} # ZZZ/USD convention
USD = "USD"

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
