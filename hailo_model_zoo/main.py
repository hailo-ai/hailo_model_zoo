#!/usr/bin/env python
import argparse
import io
from pathlib import Path


from hailo_platform.drivers.hailort.pyhailort import HEF
from hailo_platform.drivers.hw_object import PcieDevice

from hailo_sdk_common.profiler.profiler_common import ProfilerModes
from hailo_sdk_client import ClientRunner, SdkNative, SdkPartialNumeric
from hailo_sdk_client.exposed_definitions import States
from hailo_sdk_client.tools.profiler.report_generator import ReportGenerator
from hailo_model_zoo.core.main_utils import (get_network_info, parse_model, load_model,
                                             make_preprocessing, make_calibset_callback,
                                             quantize_model, infer_model, compile_model, get_hef_path,
                                             get_network_names, resolve_alls_path)
from hailo_model_zoo.utils.logger import get_logger


TARGETS = {
    'hailo8': PcieDevice,
    'full_precision': SdkNative,
    'emulator': SdkPartialNumeric
}


def _make_parsing_base():
    parsing_base_parser = argparse.ArgumentParser(add_help=False)
    network_names = list(get_network_names())
    # Setting empty metavar in order to prevent listing the models twice
    parsing_base_parser.add_argument('model_name', type=str, choices=network_names, metavar='model_name',
                                     help='Which network to run. Choices: ' + ', '.join(network_names))
    parsing_base_parser.add_argument(
        '--ckpt', type=str, default=None, dest='ckpt_path',
        help='Path to onnx or ckpt to use for parsing. By default using the model cache location')
    parsing_base_parser.add_argument(
        '--yaml', type=str, default=None, dest='yaml_path',
        help='Path to YAML for network configuration. By default using the default configuration')
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


def _str_to_profiling_mode(name):
    return ProfilerModes[name.upper()]


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
    evaluation_base_parser.add_argument(
        '--target', type=str, choices=targets, metavar='', default='full_precision',
        help='Which target to run: full_precision (GPU) / emulator (GPU) / hailo8 (PCIe)')

    evaluation_base_parser.add_argument(
        '--eval-batch-size', type=int, default=8,
        help='Batch size for INFERENCE (evaluation and pre-quant stats collection) only '
        '(feel free to increase to whatever your GPU can handle). '
        ' the quant-aware optimizers s.a. QFT & IBC use the <quantization_batch_size> field from yaml'
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


def _ensure_quantized(runner, logger, args, network_info):
    _ensure_parsed(runner, logger, network_info, args)

    if runner.state != States.HAILO_MODEL:
        return

    _quantization(runner, logger, network_info, args.calib_path, args.results_dir)


def _ensure_parsed(runner, logger, network_info, args):
    if runner.state != States.UNINITIALIZED:
        return

    parse_model(runner, network_info, ckpt_path=args.ckpt_path, results_dir=args.results_dir, logger=logger)


def _quantization(runner, logger, network_info, calib_path, results_dir):
    quant_batch_size = network_info.quantization.quantization_batch_size

    logger.info('Using batch size of {} for quantization'.format(quant_batch_size))
    preproc_callback = make_preprocessing(runner, network_info)
    calib_feed_callback = make_calibset_callback(network_info, quant_batch_size, preproc_callback, calib_path)
    quantize_model(runner, network_info, calib_feed_callback, results_dir)


def _ensure_runnable_state(args, logger, network_info, runner, target):
    if target.name == 'sdk_native':
        _ensure_parsed(runner, logger, network_info, args)
        return

    if not target.is_hardware:
        _ensure_quantized(runner, logger, args, network_info)
        return

    if args.hef_path:
        # we already have hef, just need .hn
        _ensure_parsed(runner, logger, network_info, args)
    else:
        _ensure_quantized(runner, logger, args, network_info)
        if runner.state == States.COMPILED_MODEL:
            return
        logger.info("Compiling the model (without inference) ...")
        compile_model(runner, network_info, args.results_dir)


def parse(args):
    logger = get_logger()
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path)
    logger.info('Start run for network {} ...'.format(args.model_name))

    logger.info('Initializing the runner...')
    runner = ClientRunner()

    parse_model(runner, network_info, ckpt_path=args.ckpt_path, results_dir=args.results_dir, logger=logger)


def quantize(args):
    logger = get_logger()
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path)
    logger.info('Start run for network {} ...'.format(args.model_name))

    logger.info('Initializing the runner...')
    runner = ClientRunner()

    if args.har_path:
        load_model(runner, args.har_path, logger=logger)

    _ensure_parsed(runner, logger, network_info, args)

    _quantization(runner, logger, network_info, args.calib_path, args.results_dir)


def compile(args):
    logger = get_logger()
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path)
    logger.info('Start run for network {} ...'.format(args.model_name))

    logger.info('Initializing the runner...')
    runner = ClientRunner()

    if args.har_path:
        load_model(runner, args.har_path, logger=logger)

    _ensure_quantized(runner, logger, args, network_info)

    compile_model(runner, network_info, args.results_dir)

    logger.info(f'HEF file written to {get_hef_path(args.results_dir, network_info.network.network_name)}')


def profile(args):
    profile_mode = _str_to_profiling_mode(args.profile_mode)
    if args.hef_path and profile_mode is ProfilerModes.PRE_PLACEMENT:
        raise ValueError(
            "hef is not used when profiling in pre_placement mode. use --mode post_placement for profiling with a hef.")
    logger = get_logger()
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path)
    logger.info('Start run for network {} ...'.format(args.model_name))

    logger.info('Initializing the runner...')
    runner = ClientRunner()

    if args.har_path:
        load_model(runner, args.har_path, logger=logger)

    if args.hef_path or profile_mode is ProfilerModes.PRE_PLACEMENT:
        # we already have hef (or don't need one), just need .hn
        _ensure_parsed(runner, logger, network_info, args)
    else:
        # Quantize the model so profile_hn_model could compile & profile it
        _ensure_quantized(runner, logger, args, network_info)

    alls_script_path = resolve_alls_path(network_info.paths.alls_script)
    stats, csv_data = runner.profile_hn_model(network_info.allocation.required_fps,
                                              profiling_mode=profile_mode,
                                              hef_filename=args.hef_path,
                                              allocator_script=alls_script_path)
    mem_file = io.StringIO()
    outpath = args.results_dir / f'{args.model_name}.html'
    report_generator = ReportGenerator(mem_file, csv_data, outpath, stats, hw_arch='hailo8')
    csv_data = report_generator.create_report(should_open_web_browser=False)

    logger.info(f'Profiler report generated in {outpath}')


def evaluate(args):
    if args.hef_path and args.target != 'hailo8':
        raise ValueError(
            f"hef is not used when evaluating with {args.target}. use --target hailo8 for evaluating with a hef.")

    logger = get_logger()
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path)
    model_name = network_info.network.network_name
    logger.info(f'Start run for network {model_name} ...')

    logger.info('Initializing the runner...')
    runner = ClientRunner()
    network_groups = None

    if args.har_path:
        load_model(runner, args.har_path, logger=logger)

    logger.info(f'Chosen target is {args.target}')
    hailo_target = TARGETS[args.target]
    with hailo_target() as target:
        if args.hef_path:
            hef = HEF(args.hef_path)
            network_groups = target.configure(hef)

        _ensure_runnable_state(args, logger, network_info, runner, target)

        result = infer_model(runner, network_info, target, logger,
                             args.eval_num_examples, args.data_path, args.eval_batch_size,
                             args.print_num_examples, args.visualize_results, args.video_outpath,
                             dump_results=False, network_groups=network_groups)

        return result


def _create_args_parser():
    # --- create shared arguments parsers
    parsing_base_parser = _make_parsing_base()
    quantization_base_parser = _make_quantization_base()
    hef_base_parser = _make_hef_base()
    profile_base_parser = _make_profiling_base()
    evaluation_base_parser = _make_evaluation_base()

    # --- create per action subparser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parse_parser = subparsers.add_parser('parse', parents=[parsing_base_parser])
    parse_parser.set_defaults(func=parse)

    quantize_parser = subparsers.add_parser('quantize', parents=[parsing_base_parser, quantization_base_parser])
    quantize_parser.set_defaults(func=quantize)

    compile_parser = subparsers.add_parser('compile', parents=[parsing_base_parser, quantization_base_parser])
    compile_parser.set_defaults(func=compile)

    profile_parser = subparsers.add_parser('profile', parents=[
        parsing_base_parser, quantization_base_parser, hef_base_parser, profile_base_parser])
    profile_parser.set_defaults(func=profile)

    eval_parser = subparsers.add_parser('eval', parents=[
        parsing_base_parser, quantization_base_parser, hef_base_parser, evaluation_base_parser])
    eval_parser.set_defaults(func=evaluate)

    # --- parse and run
    parser.set_defaults(func=lambda args: parser.print_help())
    return parser


def main():
    parser = _create_args_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
