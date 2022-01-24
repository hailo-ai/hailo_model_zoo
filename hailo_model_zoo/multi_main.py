#!/usr/bin/env python
import argparse
import functools

from pathlib import Path
from omegaconf import OmegaConf

from hailo_sdk_common.targets.inference_targets import ParamsKinds
from hailo_sdk_client import ClientRunner
from hailo_model_zoo.core.main_utils import (
    parse_model,
    get_network_info,
    quantize_model,
)
from hailo_model_zoo.utils.logger import get_logger
from hailo_model_zoo.utils.path_resolver import MULTI_NETWORKS_DIR


DEFAULT_CONFIG = Path(MULTI_NETWORKS_DIR) / "default_multi.yaml"


def get_quantized_model(model_name, network_info, results_dir):
    logger = get_logger()
    logger.info("Start run for network {} ...".format(model_name))

    logger.info("Initializing the runner...")
    runner = ClientRunner()

    parse_model(runner, network_info, results_dir=results_dir, logger=logger)

    logger.info("Initializing the dataset ...")
    quantize_model(runner, logger, network_info, calib_path=None, results_dir=results_dir)
    return runner


def main(cfg_path):
    with open(DEFAULT_CONFIG) as cfg_file:
        cfg = OmegaConf.load(cfg_file)

    extension = '.yaml'
    if cfg_path.endswith(extension):
        final_cfg_path = cfg_path
        cfg_dir = cfg_path[:-len(extension)]
    else:
        final_cfg_path = next(Path(MULTI_NETWORKS_DIR).glob(f'{cfg_path}/*.yaml'))
        cfg_dir = cfg_path

    with open(final_cfg_path) as cfg_file:
        cfg.update(OmegaConf.load(cfg_file))

    results_dir = Path(cfg.results_dir)
    runners = []
    for model_name in cfg.models:
        network_info = get_network_info(model_name)

        runner = get_quantized_model(model_name, network_info, results_dir)
        runners.append(runner)

    def join(lhs, rhs):
        lhs.join(rhs)
        return lhs

    functools.reduce(join, runners)
    runner = runners[0]

    hn = runner.get_hn_model()
    final_name = hn.name
    with open(results_dir / f"{final_name}.hn", "w") as alls_fp:
        alls_fp.write(hn.to_hn(final_name))

    runner.save_params(results_dir / f'{final_name}_quant.npz', ParamsKinds.TRANSLATED)

    alls_script = None if not cfg.alls_script else str(MULTI_NETWORKS_DIR.joinpath(cfg_dir, cfg.alls_script))
    hef = runner.get_hw_representation(
        fps=cfg.fps,
        mapping_timeout=1200,
        allocator_script_filename=alls_script
    )

    with open(results_dir / f"{final_name}.hef", "wb") as f:
        f.write(hef)

    return cfg, cfg_dir, runner


def get_argparser():
    parser = argparse.ArgumentParser(description="Compile multiple networks together",
                                     epilog="Example: multi_main.py fast_depth_ssd")
    configurations = [d.name for d in Path(MULTI_NETWORKS_DIR).iterdir() if d.is_dir()]
    config_string = ', '.join(configurations)
    parser.add_argument('cfg', type=str,
                        help=('Which configuration to run. Can be full path to a .yaml'
                              f' OR the name of an existing configuration: {config_string}'))
    return parser


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args.cfg)
