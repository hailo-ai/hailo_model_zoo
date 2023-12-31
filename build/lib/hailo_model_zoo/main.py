#!/usr/bin/env python
import argparse
import importlib
from pathlib import Path

import hailo_model_zoo.plugin
# we try to minimize imports to make 'main.py --help' responsive. So we only import definitions.

from hailo_model_zoo.utils.cli_utils import HMZ_COMMANDS, OneResizeValueAction, add_model_name_arg
from hailo_model_zoo.utils.constants import DEVICE_NAMES, PROFILER_MODE_NAMES, TARGETS
from hailo_model_zoo.utils.plugin_utils import iter_namespace


discovered_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in iter_namespace(hailo_model_zoo.plugin)
}


def _make_parsing_base():
    parsing_base_parser = argparse.ArgumentParser(add_help=False)
    config_group = parsing_base_parser.add_mutually_exclusive_group()
    add_model_name_arg(config_group, optional=True)
    config_group.add_argument(
        '--yaml', type=str, default=None, dest='yaml_path',
        help='Path to YAML for network configuration. By default using the default configuration')
    parsing_base_parser.add_argument(
        '--ckpt', type=str, default=None, dest='ckpt_path',
        help='Path to onnx or ckpt to use for parsing. By default using the model cache location')
    parsing_base_parser.add_argument(
        '--hw-arch', type=str, default='hailo8', metavar='', choices=['hailo8', 'hailo8l', 'hailo15h'],
        help='Which hw arch to run: hailo8 / hailo8l/ hailo15h. By default using hailo8.')
    parsing_base_parser.set_defaults(results_dir=Path('./'))
    return parsing_base_parser


def _make_optimization_base():
    optimization_base_parser = argparse.ArgumentParser(add_help=False)
    optimization_base_parser.add_argument(
        '--har', type=str, default=None, help='Use external har file', dest='har_path')
    optimization_base_parser.add_argument(
        '--calib-path', type=Path,
        help='Path to external tfrecord for calibration or a directory containing \
            images in jpg or png format',
    )
    optimization_base_parser.add_argument(
        '--model-script', type=str, default=None, dest='model_script_path',
        help='Path to model script to use. By default using the model script specified'
        ' in the network YAML configuration')
    optimization_base_parser.add_argument(
        '--performance', action='store_true',
        help='Enable flag for benchmark performance')
    optimization_base_parser.add_argument(
        '--resize', type=int, nargs='+', action=OneResizeValueAction,
        help='Add input resize from given [h,w]')
    optimization_base_parser.add_argument(
        '--input-conversion', type=str,
        choices=['nv12_to_rgb', 'yuy2_to_rgb', 'rgbx_to_rgb'],
        help='Add input conversion from given type')

    return optimization_base_parser


def _make_hef_base():
    hef_base_parser = argparse.ArgumentParser(add_help=False)
    hef_base_parser.add_argument(
        '--hef', type=str, default=None, help='Use external HEF files', dest='hef_path')
    return hef_base_parser


def _make_profiling_base():
    profile_base_parser = argparse.ArgumentParser(add_help=False)
    profile_base_parser.add_argument(
        '--mode', help='Profiling mode', dest='profile_mode',
        type=str, default='pre_placement',
        choices=PROFILER_MODE_NAMES)
    return profile_base_parser


def _make_evaluation_base():
    evaluation_base_parser = argparse.ArgumentParser(add_help=False)
    targets = TARGETS
    devices = ', '.join(DEVICE_NAMES)
    evaluation_base_parser.add_argument(
        '--target', type=str, choices=targets, metavar='', default='full_precision',
        help='Which target to run: full_precision (GPU) / emulator (GPU) / hailo8 (PCIe).\n'
        f'A specific hailo8 device may be specified. Available devices: {devices}')

    evaluation_base_parser.add_argument(
        '--batch-size', type=int,
        help='Batch size for INFERENCE (evaluation and pre-quant stats collection) only '
        '(feel free to increase to whatever your GPU can handle). '
        ' the quant-aware optimizers s.a. QFT & IBC use the calibration batch size parameter from the ALLS'
    )

    evaluation_base_parser.add_argument(
        '--data-count', type=int, default=None, dest='eval_num_examples',
        help='Maximum number of images to use for evaluation')

    evaluation_base_parser.add_argument(
        '--visualize', action='store_true', dest='visualize_results',
        help='Run visualization without evaluation. The default value is False',
    )
    evaluation_base_parser.add_argument(
        '--video-outpath',
        help='Make a video from the visualizations and save it to this path',
    )
    evaluation_base_parser.add_argument(
        '--data-path', type=Path,
        help='Path to external tfrecord for evaluation. In case you use --visualize \
            you can give a directory of images in jpg or png format',
    )
    evaluation_base_parser.set_defaults(print_num_examples=1e9,
                                        visualize_results=False)
    return evaluation_base_parser


def _create_args_parser():
    # --- create shared arguments parsers
    parsing_base_parser = _make_parsing_base()
    optimization_base_parser = _make_optimization_base()
    hef_base_parser = _make_hef_base()
    profile_base_parser = _make_profiling_base()
    evaluation_base_parser = _make_evaluation_base()

    # --- create per action subparser
    parser = argparse.ArgumentParser(epilog='Example: hailomz parse resnet_v1_50')
    # can't set the entry point for each subparser as it forces us to add imports which slow down the startup time.
    # instead we'll check the 'command' argument after parsing
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('parse', parents=[parsing_base_parser],
                          help="model translation of the input model into Hailo's internal representation.")

    subparsers.add_parser('optimize', parents=[parsing_base_parser, optimization_base_parser],
                          help="run model optimization which includes numeric translation of \
                                the input model into a compressed integer representation.")

    compile_help = ("run the Hailo compiler to generate the Hailo Executable Format file (HEF)"
                    " which can be executed on the Hailo hardware.")
    subparsers.add_parser('compile', parents=[parsing_base_parser, optimization_base_parser],
                          help=compile_help)

    profile_help = ("generate profiler report of the model."
                    " The report contains information about your model and expected performance on the Hailo hardware.")
    subparsers.add_parser('profile', parents=[
        parsing_base_parser, optimization_base_parser, hef_base_parser, profile_base_parser],
        help=profile_help)

    subparsers.add_parser('eval', parents=[
        parsing_base_parser, optimization_base_parser, hef_base_parser, evaluation_base_parser],
        help="infer the model using the Hailo Emulator or the Hailo hardware and produce the model accuracy.")

    # add parsers for plugins
    for command in HMZ_COMMANDS:
        command_parser = command.parser_fn()
        subparsers.add_parser(command.name, parents=[command_parser], help=command_parser.description)
    return parser


def run(args):
    # search for commands from plugins
    command_to_handler = {command.name: command.fn for command in HMZ_COMMANDS}
    if args.command in command_to_handler:
        return command_to_handler[args.command](args)

    # we make sure to only import these now to keep loading & plugins fast
    from hailo_model_zoo.main_driver import parse, optimize, compile, profile, evaluate
    handlers = {
        'parse': parse,
        'optimize': optimize,
        'compile': compile,
        'profile': profile,
        'eval': evaluate,
    }

    return handlers[args.command](args)


def main():
    parser = _create_args_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # from this point we can import heavy modules
    run(args)


if __name__ == '__main__':
    main()
