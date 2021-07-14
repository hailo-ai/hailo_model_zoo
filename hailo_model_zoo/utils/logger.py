import sys
import logging
from termcolor import colored
from hailo_sdk_common.logger.logger import create_custom_logger

_g_logger = None


class HailoExamplesFormatter(logging.Formatter):
    def __init__(self):
        logging.Formatter.__init__(self, colored('<Hailo Model Zoo %(levelname)s> %(message)s', 'cyan'))

    def format(self, record):
        record.levelname = record.levelname.title()
        return logging.Formatter.format(self, record)


def get_logger():
    global _g_logger
    if _g_logger is None:
        # setting console to False to set a custom console logger
        _g_logger = create_custom_logger('hailo_examples.log', fmt=None, console=False)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(HailoExamplesFormatter())
        _g_logger.addHandler(console_handler)
    return _g_logger
