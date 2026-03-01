#!/usr/bin/env python
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from hailo_model_zoo.utils.cli_utils import register_command
from hailo_model_zoo.utils.logger import get_logger
from hailo_model_zoo.utils.path_resolver import CASCADES_DIR

DEFAULT_CONFIG = Path(CASCADES_DIR) / "default_cascade.yaml"


def get_argparser():
    parser = argparse.ArgumentParser(add_help=False, description="Run multiple networks in a cascade")
    configurations = [d.name for d in Path(CASCADES_DIR).iterdir() if d.is_dir()]
    config_string = ", ".join(configurations)
    subparsers = parser.add_subparsers(dest="cascade_command")
    eval_parser = subparsers.add_parser("eval")

    eval_parser.add_argument(
        "cfg",
        type=str,
        help=(
            "Which configuration to run. Can be full path to a .yaml"
            f" OR the name of an existing configuration: {config_string}"
        ),
    )
    eval_parser.add_argument("--override", type=str, nargs="+")

    return parser


@register_command(get_argparser, name="cascade")
def main(args):
    assert args.cascade_command == "eval"
    # importing here to prevent plugin from slowing down everything
    from hailo_model_zoo.core.cascades.cascade_factory import get_cascade
    from hailo_model_zoo.core.cascades.utils import Model

    cfg_path = args.cfg

    with open(DEFAULT_CONFIG) as cfg_file:
        cfg = OmegaConf.load(cfg_file)

    extension = ".yaml"
    if cfg_path.endswith(extension):
        final_cfg_path = cfg_path
    else:
        final_cfg_path = next(Path(CASCADES_DIR).glob(f"{cfg_path}/*.yaml"))

    with open(final_cfg_path) as cfg_file:
        cfg.merge_with(OmegaConf.load(cfg_file))

    OmegaConf.set_struct(cfg, True)
    if args.override:
        cfg.merge_with(OmegaConf.from_dotlist(args.override))

    logger = get_logger()
    models = cfg.models
    runners = {stage_name: Model(**model_config) for stage_name, model_config in models.items()}
    cascade_function = get_cascade(cfg.cascade)
    cascade_function(runners, logger=logger, cfg=cfg)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
