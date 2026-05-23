from enum import Enum



class VanillaOptionType(Enum):
    CALL = 0
    PUT = 1
    STRADDLE = 2


def string_to_optiontype(s: str) -> VanillaOptionType:
    """ Convert string to VanillaOptionType """
    match s.lower():
        case 'call':
            return VanillaOptionType.CALL
        case 'put':
            return VanillaOptionType.PUT
        case 'straddle':
            return VanillaOptionType.STRADDLE
        case _:
            raise ValueError(f"Invalid option type: {s}")
