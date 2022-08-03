#!/usr/bin/env python
import argparse

from pathlib import Path

# we try to minize imports to make 'main.py --help' responsive. So we only import definitions.
from hailo_sdk_common.profiler.profiler_common import ProfilerModes

from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.hw_utils import TARGETS, DEVICE_NAMES


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
        '--model-script', type=str, default=None, dest='model_script_path',
        help='Path to model script to use. By default using the model script specified'
        ' in the network YAML configuration')
    parsing_base_parser.set_defaults(results_dir=Path('./'))
    return parsing_base_parser


def _make_quantization_base():
    quantization_base_parser = argparse.ArgumentParser(add_help=False)
    quantization_base_parser.add_argument(
        '--har', type=str, default=None, help='Use external har file', dest='har_path')
    quantization_base_parser.add_argument(
        '--calib-path', type=Path,
        help='Path to external tfrecord for calibration',
    )
    return quantization_base_parser


def _make_hef_base():
    hef_base_parser = argparse.ArgumentParser(add_help=False)
    hef_base_parser.add_argument(
        '--hef', type=str, default=None, help='Use external HEF files', dest='hef_path')
    return hef_base_parser


def _make_profiling_base():
    profile_base_parser = argparse.ArgumentParser(add_help=False)
    profiler_mode_names = {profiler_mode.value for profiler_mode in ProfilerModes}
    profile_base_parser.add_argument(
        '--mode', help='Profiling mode', dest='profile_mode',
        type=str, default=ProfilerModes.PRE_PLACEMENT.value,
        choices=profiler_mode_names)
    return profile_base_parser


def _make_evaluation_base():
    evaluation_base_parser = argparse.ArgumentParser(add_help=False)
    targets = list(TARGETS.keys())
    devices = ', '.join(DEVICE_NAMES)
    evaluation_base_parser.add_argument(
        '--target', type=str, choices=targets, metavar='', default='full_precision',
        help='Which target to run: full_precision (GPU) / emulator (GPU) / hailo8 (PCIe).\n'
        f'A specific hailo8 device may be specified. Available devices: {devices}')

    evaluation_base_parser.add_argument(
        '--eval-batch-size', type=int,
        help='Batch size for INFERENCE (evaluation and pre-quant stats collection) only '
        '(feel free to increase to whatever your GPU can handle). '
        ' the quant-aware optimizers s.a. QFT & IBC use the calibration batch size parameter from the ALLS'
    )

    evaluation_base_parser.add_argument(
        '--eval-num', type=int, default=2 ** 20, dest='eval_num_examples',
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
        help='Path to external tfrecord for evaluation',
    )
    evaluation_base_parser.set_defaults(print_num_examples=1e9,
                                        required_fps=None, visualize_results=False)
    return evaluation_base_parser


def _make_info_base():
    info_base_parser = argparse.ArgumentParser(add_help=False)
    add_model_name_arg(info_base_parser)
    return info_base_parser


def add_model_name_arg(parser, optional=False):
    network_names = list(path_resolver.get_network_names())
    # Setting empty metavar in order to prevent listing the models twice
    nargs = '?' if optional else None
    parser.add_argument('model_name', type=str, nargs=nargs, choices=network_names, metavar='model_name',
                        help='Which network to run. Choices: ' + ', '.join(network_names))


def _create_args_parser():
    # --- create shared arguments parsers
    parsing_base_parser = _make_parsing_base()
    quantization_base_parser = _make_quantization_base()
    hef_base_parser = _make_hef_base()
    profile_base_parser = _make_profiling_base()
    evaluation_base_parser = _make_evaluation_base()
    information_base_parser = _make_info_base()

    # --- create per action subparser
    parser = argparse.ArgumentParser(epilog='Example: main.py parse resnet_v1_50')
    # can't set the entry point for each subparser as it forces us to add imports which slow down the startup time.
    # instead we'll check the 'command' argument after parsing
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('parse', parents=[parsing_base_parser],
                          help="model translation of the input model into Hailo's internal representation.")

    subparsers.add_parser('optimize', parents=[parsing_base_parser, quantization_base_parser],
                          help="run model optimization which includes numeric translation of \
                                the input model into a compressed integer representation.")

    compile_help = ("run the Hailo compiler to generate the Hailo Executable Format file (HEF)"
                    " which can be executed on the Hailo hardware.")
    subparsers.add_parser('compile', parents=[parsing_base_parser, quantization_base_parser],
                          help=compile_help)

    profile_help = ("generate profiler report of the model."
                    " The report contains information about your model and expected performance on the Hailo hardware.")
    subparsers.add_parser('profile', parents=[
        parsing_base_parser, quantization_base_parser, hef_base_parser, profile_base_parser],
        help=profile_help)

    subparsers.add_parser('eval', parents=[
        parsing_base_parser, quantization_base_parser, hef_base_parser, evaluation_base_parser],
        help="infer the model using the Hailo Emulator or the Hailo hardware and produce the model accuracy.")

    subparsers.add_parser('info', parents=[information_base_parser],
                          help="Print model information.")

    return parser


def run(args):
    from hailo_model_zoo.main_driver import parse, optimize, compile, profile, evaluate, info
    handlers = {
        'parse': parse,
        'optimize': optimize,
        'compile': compile,
        'profile': profile,
        'eval': evaluate,
        'info': info,
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
