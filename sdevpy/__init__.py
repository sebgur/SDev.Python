__version__ = '1.0.5'

import logging
import colorlog

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(name)s - %(message)s",
    log_colors={
        "DEBUG":    "cyan",
        "INFO":     "green",
        "WARNING":  "yellow",
        "ERROR":    "red",
        "CRITICAL": "bold_red",
    }
))

root = logging.getLogger()
root.addHandler(handler)
root.setLevel(logging.WARNING)
logging.getLogger("sdevpy").setLevel(logging.DEBUG)
