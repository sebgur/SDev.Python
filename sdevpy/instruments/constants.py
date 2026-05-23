from enum import Enum



class OptionType(Enum):
    CALL = 0
    PUT = 1
    STRADDLE = 2


def string_to_optiontype(s: str) -> OptionType:
    """ Convert string to OptionType """
    match s.lower():
        case 'call':
            return OptionType.CALL
        case 'put':
            return OptionType.PUT
        case 'straddle':
            return OptionType.STRADDLE
        case _:
            raise ValueError(f"Invalid option type: {s}")
