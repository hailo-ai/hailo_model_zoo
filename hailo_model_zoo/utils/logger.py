import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_g_logger = None


class HailoExamplesFormatter(logging.Formatter):
    COLOR_PREFIX = "\x1b["
    COLOR_SUFFIX = "\x1b[0m"
    COLORS = {
        logging.DEBUG: "34m",  # blue
        logging.INFO: "36m",  # Cyan
        logging.WARNING: "33;1m",  # bold yellow
        logging.ERROR: "31;1m",  # bold red
        logging.CRITICAL: "41;1m",  # bold white on red
        # This has to match hailo_sdk_common.logger.logger.DEPRECATION_WARNING.
        # Unfortunately we can't import it here because it's too slow.
        logging.WARNING - 1: "33;21m",  # yellow
    }

    def format(self, record):
        # record.levelname = record.levelname.title()
        level_name = record.levelname
        level_no = record.levelno
        message = "%(message)s"
        level_fmt = (
            f"{self.COLOR_PREFIX}{self.COLORS[level_no]}<Hailo Model Zoo " f"{level_name}> {message}{self.COLOR_SUFFIX}"
        )
        formatter = logging.Formatter(f"{level_fmt}")
        return formatter.format(record)


def namer(name):
    return name.replace(".log", "") + ".log"


def get_logger():
    # for faster loading
    from hailo_sdk_common.logger.logger import DFC_FOLDER_PATH, create_custom_logger

    MZ_FOLDER_PATH = str(Path(DFC_FOLDER_PATH, "..", "modelzoo"))

    global _g_logger
    if _g_logger is None:
        # setting console to False to set a custom console logger
        Path(MZ_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
        log_file_name = MZ_FOLDER_PATH + "/hailo_examples.log"
        _g_logger = create_custom_logger("hailo_examples.log", fmt=None, console=False)
        rotate_handler = RotatingFileHandler(log_file_name, maxBytes=10000000, backupCount=10)
        rotate_handler.namer = namer
        _g_logger.addHandler(rotate_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(HailoExamplesFormatter())
        _g_logger.addHandler(console_handler)
    return _g_logger
