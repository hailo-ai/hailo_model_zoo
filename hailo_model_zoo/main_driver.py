import io

try:
    from hailo_platform import HEF, PcieDevice
    HEF_EXISTS = True
except ModuleNotFoundError:
    HEF_EXISTS = False

from hailo_sdk_common.targets.inference_targets import SdkNative, SdkPartialNumeric
from hailo_sdk_common.profiler.profiler_common import ProfilerModes
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import States
from hailo_sdk_client.tools.profiler.react_report_generator import ReactReportGenerator
from hailo_model_zoo.core.main_utils import (get_network_info, parse_model,
                                             optimize_model, infer_model, compile_model, get_hef_path,
                                             resolve_alls_path, _get_integrated_postprocessing)
from hailo_model_zoo.utils.hw_utils import DEVICE_NAMES, TARGETS
from hailo_model_zoo.utils.logger import get_logger


def _ensure_compiled(runner, logger, args, network_info):
    if runner.state == States.COMPILED_MODEL:
        return
    logger.info("Compiling the model (without inference) ...")
    compile_model(runner, network_info, args.results_dir, model_script_path=args.model_script_path)


def _ensure_optimized(runner, logger, args, network_info):
    _ensure_parsed(runner, logger, network_info, args)

    integrated_postprocessing = _get_integrated_postprocessing(network_info)
    if integrated_postprocessing and integrated_postprocessing.enabled and args.model_script_path is not None:
        raise ValueError(f"Network {network_info.network.network_name} joins several networks together\n"
                         "and cannot get a user model script")

    if runner.state != States.HAILO_MODEL:
        return

    optimize_model(runner, logger, network_info, args.calib_path, args.results_dir,
                   model_script_path=args.model_script_path)


def _ensure_parsed(runner, logger, network_info, args):
    _hailo8l_warning(args.hw_arch, logger)

    if runner.state != States.UNINITIALIZED:
        return

    parse_model(runner, network_info, ckpt_path=args.ckpt_path, results_dir=args.results_dir, logger=logger)


def _ensure_runnable_state(args, logger, network_info, runner, target):
    _ensure_parsed(runner, logger, network_info, args)
    if isinstance(target, SdkNative):
        if runner.state == States.HAILO_MODEL:
            integrated_postprocessing = _get_integrated_postprocessing(network_info)
            if integrated_postprocessing and integrated_postprocessing.enabled:
                runner.apply_model_modification_commands()
                return None

            model_script = args.model_script_path if args.model_script_path \
                else resolve_alls_path(network_info.paths.alls_script)
            runner.load_model_script(model_script)
            runner.apply_model_modification_commands()
        return None

    if args.hef_path:
        hef = HEF(args.hef_path)
        network_groups = target.configure(hef)
        return network_groups

    _ensure_optimized(runner, logger, args, network_info)

    if isinstance(target, SdkPartialNumeric):
        return None

    assert isinstance(target, PcieDevice)
    _ensure_compiled(runner, logger, args, network_info)
    return None


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
    runner = ClientRunner()
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

    optimize_model(runner, logger, network_info, args.calib_path, args.results_dir,
                   model_script_path=args.model_script_path)


def compile(args):
    logger = get_logger()
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path)
    model_name = network_info.network.network_name
    logger.info(f'Start run for network {model_name} ...')

    logger.info(f'Initializing the {args.hw_arch} runner...')
    runner = ClientRunner(hw_arch=args.hw_arch, har_path=args.har_path)

    _ensure_optimized(runner, logger, args, network_info)

    compile_model(runner, network_info, args.results_dir, model_script_path=args.model_script_path)

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

    if profile_mode is ProfilerModes.PRE_PLACEMENT:
        _ensure_parsed(runner, logger, network_info, args)
    elif args.hef_path:
        # we already have hef (or don't need one), just need .hn
        _ensure_parsed(runner, logger, network_info, args)
        if runner.state == States.HAILO_MODEL:
            model_script = args.model_script_path if args.model_script_path \
                else resolve_alls_path(network_info.paths.alls_script)
            runner.load_model_script(model_script)
            runner.apply_model_modification_commands()
    else:
        # Optimize the model so profile_hn_model could compile & profile it
        _ensure_optimized(runner, logger, args, network_info)

    alls_script_path = (args.model_script_path if args.model_script_path else
                        resolve_alls_path(network_info.paths.alls_script)) \
        if profile_mode is not ProfilerModes.PRE_PLACEMENT else None

    stats, csv_data, latency_data = runner.profile_hn_model(profiling_mode=profile_mode,
                                                            should_use_logical_layers=True,
                                                            allocator_script=alls_script_path,
                                                            hef_filename=args.hef_path)

    mem_file = io.StringIO()
    outpath = args.results_dir / f'{model_name}.html'
    report_generator = ReactReportGenerator(mem_file=mem_file, csv_data=csv_data, latency_data=latency_data,
                                            runtime_data=latency_data["runtime_data"], out_path=outpath,
                                            stats=stats, hw_arch="hailo8")

    csv_data = report_generator.create_report(should_open_web_browser=False)
    logger.info(f'Profiler report generated in {outpath}')

    return stats, csv_data, latency_data


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
    logger.info(f'Start run for network {model_name} ...')

    logger.info('Initializing the runner...')
    runner = ClientRunner(hw_arch=args.hw_arch, har_path=args.har_path)
    network_groups = None

    logger.info(f'Chosen target is {args.target}')
    hailo_target = TARGETS[args.target]
    with hailo_target() as target:
        network_groups = _ensure_runnable_state(args, logger, network_info, runner, target)

        batch_size = args.eval_batch_size or __get_batch_size(network_info, target)
        result = infer_model(runner, network_info, target, logger,
                             args.eval_num_examples, args.data_path, batch_size,
                             args.print_num_examples, args.visualize_results, args.video_outpath,
                             dump_results=False, network_groups=network_groups)

        return result


def __get_batch_size(network_info, target):
    if target.name == 'sdk_native':
        return network_info.inference.full_precision_batch_size
    return network_info.inference.emulator_batch_size
