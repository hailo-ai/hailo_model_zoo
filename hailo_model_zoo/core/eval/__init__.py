import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
except Exception:
    logger.debug("pyximport not available — Cython eval extensions will fail at runtime if used.")
