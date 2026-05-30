import logging, colorlog


LOG_COLORS = {"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red",
              "CRITICAL": "bold_red"}


def configure(root_level: int=logging.WARNING, sdevpy_level: int=logging.DEBUG,
              module_display: str='none') -> None:
    """ Configure logger """
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
    logging.basicConfig(level=root_level, handlers=[handler], force=True)
    logging.getLogger("sdevpy").setLevel(sdevpy_level)
