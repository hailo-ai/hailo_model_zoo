from omegaconf import OmegaConf
from pathlib import Path
from functools import lru_cache

from hailo_model_zoo.utils import path_resolver


@lru_cache(maxsize=128)
def _load_cfg(cfg_file):
    cfg = OmegaConf.load(cfg_file)
    base = cfg.get('base')
    if not base:
        return cfg
    del cfg['base']
    # if extension exists make a recursive call for each extension
    config = OmegaConf.create()
    for f in base:
        cfg_path = path_resolver.resolve_cfg_path(f)
        config = OmegaConf.merge(config, _load_cfg(cfg_path))
    # override with cfg fields
    config = OmegaConf.merge(config, cfg)
    return config


def get_network_info(model_name, read_only=False, yaml_path=None):
    '''
    Args:
        model_name: The network name to load.
        read_only: If set return read-only object.
                   The read_only mode save run-time and memroy.
        yaml_path: Path to external YAML file for network configuration
    Return:
        OmegaConf object that represent network configuration.
    '''
    if model_name is None and yaml_path is None:
        raise ValueError("Either model_name or yaml_path must be given")
    net = f'networks/{model_name}.yaml'
    cfg_path = Path(yaml_path) if yaml_path is not None else path_resolver.resolve_cfg_path(net)
    if not cfg_path.is_file():
        raise ValueError('cfg file is missing in {}'.format(cfg_path))
    cfg = _load_cfg(cfg_path)
    if read_only:
        OmegaConf.set_readonly(cfg, True)
        return cfg
    else:
        cfg_copy = cfg.copy()
        OmegaConf.set_readonly(cfg_copy, False)
        return cfg_copy
