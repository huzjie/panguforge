"""Central logging setup."""
import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def get_logger(name="panguforge", level="INFO", stream=True):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(h)
    return logger
