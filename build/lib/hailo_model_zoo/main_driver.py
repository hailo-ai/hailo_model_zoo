from pathlib import Path

try:
    from hailo_platform import HEF, PcieDevice
    HEF_EXISTS = True
except ModuleNotFoundError:
    HEF_EXISTS = False

from hailo_sdk_client import ClientRunner, InferenceContext
from hailo_sdk_client.exposed_definitions import States
from hailo_sdk_client.tools.profiler.react_report_generator import ReactReportGenerator
from hailo_sdk_common.profiler.profiler_common import ProfilerModes
from hailo_sdk_common.targets.inference_targets import SdkFPOptimized, SdkPartialNumeric
from hailo_model_zoo.core.main_utils import (compile_model, get_hef_path, get_integrated_postprocessing,
                                             get_network_info, infer_model_tf1, infer_model_tf2,
                                             optimize_full_precision_model, optimize_model, parse_model,
                                             resolve_alls_path)
from hailo_model_zoo.utils.hw_utils import DEVICE_NAMES, DEVICES, INFERENCE_TARGETS, TARGETS
from hailo_model_zoo.utils.logger import get_logger
from hailo_model_zoo.utils.path_resolver import get_network_performance


def _ensure_performance(model_name, model_script, performance, logger):
    if not performance and model_name in get_network_performance():
        # Check whether the model has a performance
        logger.info(f'Running {model_name} with default model script.\n\
                      To obtain maximum performance use --performance:\n\
                      hailomz <command> {model_name} --performance')
    if performance and model_script:
        if model_script.parent.name == "base":
            logger.info(f'Using base alls script found in {model_script} because there is no performance alls')
        if model_script.parent.name == "generic":
            logger.info(f'Using generic alls script found in {model_script} because there is no specific hardware alls')
        elif model_script.parent.name != "performance" and model_name in get_network_performance():
            logger.info(f'Using alls script {model_script}')


def _extract_model_script_path(networks_alls_script, model_script_path, hw_arch, performance):
    return Path(model_script_path) if model_script_path \
        else resolve_alls_path(networks_alls_script, hw_arch=hw_arch, performance=performance)


def _ensure_compiled(runner, logger, args, network_info):
    if runner.state == States.COMPILED_MODEL or runner.hef:
        return
    logger.info("Compiling the model (without inference) ...")
    compile_model(runner, network_info, args.results_dir, allocator_script_filename=args.model_script_path)


def _ensure_optimized(runner, logger, args, network_info):
    _ensure_parsed(runner, logger, network_info, args)

    integrated_postprocessing = get_integrated_postprocessing(network_info)
    if integrated_postprocessing and integrated_postprocessing.enabled and args.model_script_path is not None:
        raise ValueError(f"Network {network_info.network.network_name} joins several networks together\n"
                         "and cannot get a user model script")

    if runner.state != States.HAILO_MODEL:
        return
    model_script = _extract_model_script_path(network_info.paths.alls_script,
                                              args.model_script_path,
                                              args.hw_arch,
                                              args.performance)
    _ensure_performance(network_info.network.network_name, model_script, args.performance, logger)
    optimize_model(runner, logger, network_info, args.calib_path, args.results_dir,
                   model_script=model_script, resize=args.resize, input_conversion=args.input_conversion)


def _ensure_parsed(runner, logger, network_info, args):
    _hailo8l_warning(args.hw_arch, logger)

    if runner.state != States.UNINITIALIZED:
        return

    parse_model(runner, network_info, ckpt_path=args.ckpt_path, results_dir=args.results_dir, logger=logger)


def configure_hef_tf1(hef_path, target):
    hef = HEF(hef_path)
    network_groups = target.configure(hef)
    return network_groups


def configure_hef_tf2(runner, hef_path):
    if hef_path:
        runner.hef = hef_path
    return


def _ensure_runnable_state_tf1(args, logger, network_info, runner, target):
    _ensure_parsed(runner, logger, network_info, args)
    if isinstance(target, SdkFPOptimized) or (isinstance(target, PcieDevice) and args.hef_path is not None):
        if runner.state == States.HAILO_MODEL:
            integrated_postprocessing = get_integrated_postprocessing(network_info)
            if integrated_postprocessing and integrated_postprocessing.enabled:
                runner.optimize_full_precision()
                return None if not args.hef_path else configure_hef_tf1(args.hef_path, target)
            # We intenionally use base model script and assume its modifications
            # compatible to the performance model script
            model_script = _extract_model_script_path(network_info.paths.alls_script,
                                                      args.model_script_path,
                                                      args.hw_arch,
                                                      performance=False)

            optimize_full_precision_model(runner, model_script=model_script, resize=args.resize,
                                          input_conversion=args.input_conversion)

        return None if not args.hef_path else configure_hef_tf1(args.hef_path, target)

    if args.hef_path:
        return configure_hef_tf1(args.hef_path, target)

    _ensure_optimized(runner, logger, args, network_info)

    if isinstance(target, SdkPartialNumeric):
        return None

    assert isinstance(target, PcieDevice)
    _ensure_compiled(runner, logger, args, network_info)
    return None


def _ensure_runnable_state_tf2(args, logger, network_info, runner, target):
    _ensure_parsed(runner, logger, network_info, args)
    if target == InferenceContext.SDK_FP_OPTIMIZED or \
            (target == InferenceContext.SDK_HAILO_HW and args.hef_path is not None):
        if runner.state != States.HAILO_MODEL:
            configure_hef_tf2(runner, args.hef_path)
            return

        # We intenionally use base model script and assume its modifications
        # compatible to the performance model script
        model_script = _extract_model_script_path(network_info.paths.alls_script,
                                                  args.model_script_path,
                                                  args.hw_arch,
                                                  False)
        optimize_full_precision_model(runner, model_script=model_script, resize=args.resize,
                                      input_conversion=args.input_conversion)
        configure_hef_tf2(runner, args.hef_path)
        return

    configure_hef_tf2(runner, args.hef_path)

    _ensure_optimized(runner, logger, args, network_info)

    if target == InferenceContext.SDK_QUANTIZED:
        return

    _ensure_compiled(runner, logger, args, network_info)
    return


def _str_to_profiling_mode(name):
    return ProfilerModes[name.upper()]


def _hailo8l_warning(hw_arch, logger):
    if hw_arch == "hailo8l":
        logger.warning("Hailo8L support is currently at Preview on Hailo Model Zoo")


def parse(args):
    logger = get_logger()
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path)
    model_name = network_info.network.network_name
    logger.info(f'Start run for network {model_name} ...')

    logger.info('Initializing the runner...')
    runner = ClientRunner(hw_arch=args.hw_arch)
    parse_model(runner, network_info, ckpt_path=args.ckpt_path, results_dir=args.results_dir, logger=logger)


def optimize(args):
    logger = get_logger()
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path)
    model_name = network_info.network.network_name
    logger.info(f'Start run for network {model_name} ...')

    if args.calib_path is None and network_info.quantization.calib_set is None:
        raise ValueError(
            "Cannot run optimization without dataset. use --calib-path to provide external dataset.")

    logger.info(f'Initializing the {args.hw_arch} runner...')
    runner = ClientRunner(hw_arch=args.hw_arch, har_path=args.har_path)

    _ensure_parsed(runner, logger, network_info, args)

    model_script = _extract_model_script_path(network_info.paths.alls_script,
                                              args.model_script_path,
                                              args.hw_arch,
                                              args.performance)
    _ensure_performance(model_name, model_script, args.performance, logger)
    optimize_model(runner, logger, network_info, args.calib_path, args.results_dir,
                   model_script=model_script, resize=args.resize, input_conversion=args.input_conversion)


def compile(args):
    logger = get_logger()
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path)
    model_name = network_info.network.network_name
    logger.info(f'Start run for network {model_name} ...')

    logger.info(f'Initializing the {args.hw_arch} runner...')
    runner = ClientRunner(hw_arch=args.hw_arch, har_path=args.har_path)

    _ensure_optimized(runner, logger, args, network_info)

    model_script = _extract_model_script_path(network_info.paths.alls_script,
                                              args.model_script_path, args.hw_arch, args.performance)
    _ensure_performance(model_name, model_script, args.performance, logger)
    compile_model(runner, network_info, args.results_dir, model_script)

    logger.info(f'HEF file written to {get_hef_path(args.results_dir, network_info.network.network_name)}')


def profile(args):
    profile_mode = _str_to_profiling_mode(args.profile_mode)
    if args.hef_path and profile_mode is ProfilerModes.PRE_PLACEMENT:
        raise ValueError(
            "hef is not used when profiling in pre_placement mode. use --mode post_placement for profiling with a hef.")
    logger = get_logger()
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path)
    model_name = network_info.network.network_name
    logger.info(f'Start run for network {model_name} ...')

    logger.info(f'Initializing the {args.hw_arch} runner...')
    runner = ClientRunner(hw_arch=args.hw_arch, har_path=args.har_path)

    if profile_mode is ProfilerModes.PRE_PLACEMENT or args.hef_path:
        # we already have hef (or don't need one), just need .hn
        _ensure_parsed(runner, logger, network_info, args)
        if runner.state == States.HAILO_MODEL:
            if args.hef_path:
                model_script = _extract_model_script_path(network_info.paths.alls_script,
                                                          args.model_script_path,
                                                          args.hw_arch,
                                                          args.performance)
                _ensure_performance(model_name, model_script, args.performance, logger)
            else:
                model_script = _extract_model_script_path(network_info.paths.alls_script,
                                                          args.model_script_path,
                                                          args.hw_arch,
                                                          False)
            optimize_full_precision_model(runner, model_script=model_script, resize=args.resize,
                                          input_conversion=args.input_conversion)
    else:
        # Optimize the model so profile_hn_model could compile & profile it
        _ensure_optimized(runner, logger, args, network_info)

    export = runner._profile(profiling_mode=profile_mode, should_use_logical_layers=True,
                             hef_filename=args.hef_path)
    outpath = args.results_dir / f'{model_name}.html'
    report_generator = ReactReportGenerator(export, outpath)
    csv_data = report_generator.create_report(should_open_web_browser=False)
    logger.info(f'Profiler report generated in {outpath}')

    return export['stats'], csv_data, export['latency_data']


def evaluate(args):
    if args.target == 'hailo8' and not HEF_EXISTS:
        raise ModuleNotFoundError(
            f"HailoRT is not available, in case you want to run on {args.target} you should install HailoRT first")

    if args.hef_path and not HEF_EXISTS:
        raise ModuleNotFoundError(
            "HailoRT is not available, in case you want to evaluate with hef you should install HailoRT first")

    hardware_targets = set(DEVICE_NAMES)
    hardware_targets.add('hailo8')
    if args.hef_path and args.target not in hardware_targets:
        raise ValueError(
            f"hef is not used when evaluating with {args.target}. use --target hailo8 for evaluating with a hef.")

    if args.video_outpath and not args.visualize_results:
        raise ValueError(
            "The --video-output argument requires --visualize argument")

    logger = get_logger()
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path)

    if args.data_path is None and network_info.evaluation.data_set is None:
        raise ValueError(
            "Cannot run evaluation without dataset. use --data-path to provide external dataset.")
    model_name = network_info.network.network_name
    if args.hw_arch == "hailo15h" and args.target == "hailo8":
        raise ValueError(
            "Cannot run hailo15h compiled hef on hailo8.")
    logger.info(f'Start run for network {model_name} ...')

    logger.info('Initializing the runner...')
    runner = ClientRunner(hw_arch=args.hw_arch, har_path=args.har_path)
    network_groups = None

    logger.info(f'Chosen target is {args.target}')
    batch_size = args.batch_size or __get_batch_size(network_info, args.target)
    infer_type = network_info.evaluation.infer_type
    # legacy tf1 inference flow
    if infer_type not in ['runner_infer', 'model_infer', 'np_infer', 'facenet_infer',
                          'np_infer_lite', 'model_infer_lite']:
        hailo_target = TARGETS[args.target]
        with hailo_target() as target:
            network_groups = _ensure_runnable_state_tf1(args, logger, network_info, runner, target)
            return infer_model_tf1(runner, network_info, target, logger,
                                   args.eval_num_examples, args.data_path, batch_size,
                                   args.print_num_examples, args.visualize_results, args.video_outpath,
                                   dump_results=False, network_groups=network_groups)

    else:
        # new tf2 inference flow
        target = INFERENCE_TARGETS[args.target]
        _ensure_runnable_state_tf2(args, logger, network_info, runner, target)

        device_info = DEVICES.get(args.target)
        context = runner.infer_context(target, device_info)
        return infer_model_tf2(runner, network_info, context, logger, args.eval_num_examples, args.data_path,
                               batch_size, args.print_num_examples, args.visualize_results, args.video_outpath,
                               dump_results=False)


def __get_batch_size(network_info, target):
    if target == 'full_precision':
        return network_info.inference.full_precision_batch_size
    return network_info.inference.emulator_batch_size
