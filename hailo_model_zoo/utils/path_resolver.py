from pathlib import Path
import os

# /path/to/hailo_networks/utils/path_resolver.py ->  /path/to/hailo_networks
_MODEL_ZOO_ROOT = Path(__file__).parent.parent.absolute()
_DATA_DEFAULT_DIRECTORY = 'data'
BASE_CFG_DIR = _MODEL_ZOO_ROOT / 'cfg'
NETWORK_CFG_DIR = BASE_CFG_DIR / 'networks'
ALLS_DIR = BASE_CFG_DIR / 'alls'
MULTI_NETWORKS_DIR = BASE_CFG_DIR / 'multi-networks'


def resolve_alls_path(path):
    return ALLS_DIR / path


def resolve_cfg_path(path):
    return BASE_CFG_DIR / path


def resolve_model_path(path_list):
    path = Path(path_list[0])
    if path.suffix in ['.onnx', '.pb']:
        return resolve_data_path(path)

    # ckpt files have strange extensions, but we want up to .ckpt
    return resolve_data_path(path.with_suffix(''))


def resolve_data_path(path):
    data_dir = os.getenv('HMZ_DATA', _DATA_DEFAULT_DIRECTORY)
    return Path(data_dir) / path


def get_network_names():
    return sorted([name.with_suffix('').name for name in NETWORK_CFG_DIR.glob('*.yaml')])
