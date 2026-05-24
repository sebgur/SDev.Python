import logging, colorlog


def configure(root_level: int = logging.WARNING, sdevpy_level: int = logging.DEBUG) -> None:
    """ Configure logger """
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(name)s - %(message)s",
        log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow",
                    "ERROR": "red", "CRITICAL": "bold_red"},
    ))
    logging.basicConfig(level=root_level, handlers=[handler], force=True)
    logging.getLogger("sdevpy").setLevel(sdevpy_level)
