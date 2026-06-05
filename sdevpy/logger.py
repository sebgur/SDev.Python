import logging, colorlog


LOG_COLORS = {"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red",
              "CRITICAL": "bold_red"}


def string_to_logging_level(level_str: str):
    level_str = level_str.lower()
    match level_str:
        case 'debug':
            return logging.DEBUG
        case 'info':
            return logging.INFO
        case 'warning':
            return logging.WARNING
        case 'error':
            return logging.ERROR
        case 'critical':
            return logging.CRITICAL
        case _:
            raise ValueError(f"Unsupported logging level: {level_str}")


def configure(root_level: str='warning', sdevpy_level: str='debug',
              module_display: str='none') -> None:
    """ Configure logger. Levels can be: debug, info, warning, error and critical. """
    handler = colorlog.StreamHandler()

    match module_display.lower():
        case 'none':
            module_str = "%(log_color)s%(levelname)-1s%(reset)s | %(message)s"
        case 'partial':
            module_str = "%(log_color)s%(levelname)-1s%(reset)s | %(module)s | %(message)s"
        case 'full':
            module_str = "%(log_color)s%(levelname)-1s%(reset)s | %(name)s | %(message)s"
        case _:
            raise ValueError(f"Unsupported module display mode: {module_display}")

    handler.setFormatter(colorlog.ColoredFormatter(module_str, log_colors=LOG_COLORS))
    logging.basicConfig(level=string_to_logging_level(root_level), handlers=[handler], force=True)
    logging.getLogger("sdevpy").setLevel(string_to_logging_level(sdevpy_level))
