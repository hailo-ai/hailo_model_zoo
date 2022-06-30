#!/usr/bin/env python
import argparse
import functools

from pathlib import Path
from omegaconf import OmegaConf

from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import States
from hailo_model_zoo.core.main_utils import (
    parse_model,
    get_network_info,
    quantize_model,
    load_model
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


def join_models(models, results_dir):
    runners = []
    for model_name in models:
        network_info = get_network_info(model_name)

        runner = get_quantized_model(model_name, network_info, results_dir)
        runners.append(runner)

    def join(lhs, rhs):
        lhs.join(rhs)
        return lhs

    functools.reduce(join, runners)
    runner = runners[0]
    runner.save_har(results_dir / f'{runner.model_name}.har')
    return runner


def main(cfg_path, har_path=None):
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
    if not har_path:
        runner = join_models(cfg.models, results_dir)
    else:
        logger = get_logger()
        runner = ClientRunner()
        load_model(runner, har_path, logger)
        if (runner.state not in [States.QUANTIZED_MODEL, States.COMPILED_MODEL]):
            raise ValueError(
                f'HAR file {har_path} does not seem to be quantized')
    final_name = runner.model_name

    alls_script = None if not cfg.alls_script else str(MULTI_NETWORKS_DIR.joinpath(cfg_dir, cfg.alls_script))
    hef = runner.get_hw_representation(
        fps=cfg.fps,
        allocator_script_filename=alls_script
    )
    runner.save_har(results_dir / f'{final_name}.har')

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
    parser.add_argument('--har', type=str, default=None,
                        help='Path to quantized HAR to compile from')
    return parser


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args.cfg, har_path=args.har)
