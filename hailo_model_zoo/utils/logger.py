import sys
import logging
from hailo_sdk_common.logger.logger import create_custom_logger

_g_logger = None


class HailoExamplesFormatter(logging.Formatter):
    COLOR_PREFIX = '\x1b['
    COLOR_SUFFIX = '\x1b[0m'
    COLORS = {
        logging.DEBUG: '34m',  # blue
        logging.INFO: '36m',  # Cyan
        logging.WARNING: '33;1m',  # bold yellow
        logging.ERROR: '31;1m',  # bold red
        logging.CRITICAL: '41;1m',  # bold white on red
    }

    def format(self, record):
        # record.levelname = record.levelname.title()
        level_name = record.levelname
        level_no = record.levelno
        message = '%(message)s'
        level_fmt = f'{self.COLOR_PREFIX}{self.COLORS[level_no]}<Hailo Model Zoo '\
                    f'{level_name}> {message}{self.COLOR_SUFFIX}'
        formatter = logging.Formatter(f'{level_fmt}')
        return formatter.format(record)


def get_logger():
    global _g_logger
    if _g_logger is None:
        # setting console to False to set a custom console logger
        _g_logger = create_custom_logger('hailo_examples.log', fmt=None, console=False)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(HailoExamplesFormatter())
        _g_logger.addHandler(console_handler)
    return _g_logger
