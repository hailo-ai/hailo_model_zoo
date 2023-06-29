from pathlib import Path
import os

# /path/to/hailo_networks/utils/path_resolver.py ->  /path/to/hailo_networks
_MODEL_ZOO_ROOT = Path(__file__).parent.parent.absolute()
_USER_DATA_DIR = Path.home() / ".hailomz"
_DATA_DEFAULT_DIRECTORY = _USER_DATA_DIR / 'data'
BASE_CFG_DIR = _MODEL_ZOO_ROOT / 'cfg'
NETWORK_CFG_DIR = BASE_CFG_DIR / 'networks'
ALLS_DIR = BASE_CFG_DIR / 'alls'
ALLS_PERFORMANCE_DIR = ALLS_DIR / 'performance'
MULTI_NETWORKS_DIR = BASE_CFG_DIR / 'multi-networks'

# ensure env var HMZ_DATA exists
os.environ['HMZ_DATA'] = os.environ.get('HMZ_DATA', str(_DATA_DEFAULT_DIRECTORY))


def resolve_alls_path(path):
    alls_path = ALLS_DIR / path
    if alls_path.exists():
        return alls_path
    elif (alls_path.parents[1] / Path("base") / alls_path.name).exists():
        return alls_path.parents[1] / Path("base") / alls_path.name
    elif (ALLS_DIR / Path('generic') / alls_path.name).exists():
        return ALLS_DIR / Path('generic') / alls_path.name
    raise Exception("no alls found for requested hw_arch")


def resolve_cfg_path(path):
    return BASE_CFG_DIR / path


def resolve_model_path(path_list):
    path = Path(path_list[0])
    if path.suffix in ['.onnx', '.pb', '.tflite']:
        return resolve_data_path(path)

    # ckpt files have strange extensions, but we want up to .ckpt
    return resolve_data_path(path.with_suffix(''))


def resolve_data_path(path):
    data_dir = os.getenv('HMZ_DATA', _DATA_DEFAULT_DIRECTORY)
    return Path(data_dir) / path


def get_network_names():
    return sorted([name.with_suffix('').name for name in NETWORK_CFG_DIR.glob('*.yaml')])


def get_network_peformance():
    return sorted([name.with_suffix('').name for name in ALLS_PERFORMANCE_DIR.glob('*.alls')])
