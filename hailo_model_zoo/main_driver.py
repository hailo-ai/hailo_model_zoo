from pathlib import Path

try:
    from hailo_platform import HEF, Device

    HEF_EXISTS = True
except ModuleNotFoundError:
    HEF_EXISTS = False

from hailo_sdk_client import ClientRunner, InferenceContext
from hailo_sdk_client.exposed_definitions import States
from hailo_sdk_client.tools.profiler.react_report_generator import ReactReportGenerator
from hailo_sdk_common.logger.logger import DeprecationVersion
from hailo_sdk_common.targets.inference_targets import SdkFPOptimized, SdkPartialNumeric

from hailo_model_zoo.core.main_utils import (
    compile_model,
    get_hef_path,
    get_integrated_postprocessing,
    get_network_info,
    infer_model_tf2,
    is_network_performance,
    optimize_full_precision_model,
    optimize_model,
    parse_model,
    prepare_calibration_data,
    resolve_alls_path,
)
from hailo_model_zoo.utils.hw_utils import DEVICE_NAMES, DEVICES, INFERENCE_TARGETS
from hailo_model_zoo.utils.logger import get_logger


def _ensure_performance(model_name, model_script, performance, hw_arch, logger):
    if not performance and is_network_performance(model_name, hw_arch):
        # Check whether the model has a performance
        logger.info(f"Running {model_name} with default model script.\n\
                       To obtain maximum performance use --performance:\n\
                       hailomz <command> {model_name} --performance")
    if performance and model_script:
        if model_script.parent.name == "base":
            logger.info(f"Using base alls script found in {model_script} because there is no performance alls")
        elif model_script.parent.name == "generic":
            logger.info(f"Using generic alls script found in {model_script} because there is no specific hardware alls")
        elif model_script.parent.name == "performance" and is_network_performance(model_name, hw_arch):
            logger.info(f"Using performance alls script found in {model_script}")
        else:
            logger.info(f"Using alls script from {model_script}")


def _extract_model_script_path(networks_alls_script, model_script_path, hw_arch, performance):
    return (
        Path(model_script_path)
        if model_script_path
        else resolve_alls_path(networks_alls_script, hw_arch=hw_arch, performance=performance)
    )


def _ensure_compiled(runner, logger, args, network_info):
    if runner.state == States.COMPILED_MODEL or runner.hef:
        return
    logger.info("Compiling the model (without inference) ...")
    compile_model(
        runner,
        network_info,
        args.results_dir,
        allocator_script_filename=args.model_script_path,
        performance=args.performance,
    )


def _ensure_optimized(runner, logger, args, network_info):
    _ensure_parsed(runner, logger, network_info, args)

    integrated_postprocessing = get_integrated_postprocessing(network_info)
    if integrated_postprocessing and integrated_postprocessing.enabled and args.model_script_path is not None:
        raise ValueError(
            f"Network {network_info.network.network_name} joins several networks together\n"
            "and cannot get a user model script"
        )

    if runner.state != States.HAILO_MODEL:
        return
    model_script = _extract_model_script_path(
        network_info.paths.alls_script, args.model_script_path, args.hw_arch, args.performance
    )
    _ensure_performance(network_info.network.network_name, model_script, args.performance, args.hw_arch, logger)
    calib_feed_callback = prepare_calibration_data(
        runner, network_info, args.calib_path, logger, args.input_conversion, args.resize
    )
    optimize_model(
        runner,
        calib_feed_callback,
        logger,
        network_info,
        args.results_dir,
        model_script,
        args.resize,
        args.input_conversion,
        args.classes,
    )


def _ensure_parsed(runner, logger, network_info, args):
    if runner.state != States.UNINITIALIZED:
        return

    if args.hw_arch not in network_info.info.supported_hw_arch:
        msg = (
            f"Model {args.model_name} is not supported with hw_arch: {args.hw_arch}. "
            f"Supported hw_arch values: {network_info.info.supported_hw_arch}"
        )
        raise ValueError(msg)

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
    if isinstance(target, SdkFPOptimized) or (isinstance(target, Device) and args.hef_path is not None):
        if runner.state == States.HAILO_MODEL:
            calib_feed_callback = prepare_calibration_data(
                runner, network_info, args.calib_path, logger, args.input_conversion, args.resize
            )
            integrated_postprocessing = get_integrated_postprocessing(network_info)
            if integrated_postprocessing and integrated_postprocessing.enabled:
                runner.optimize_full_precision(calib_data=calib_feed_callback)
                return configure_hef_tf1(args.hef_path, target) if args.hef_path else None
            # We intentionally use base model script and assume its modifications
            # compatible to the performance model script
            model_script = _extract_model_script_path(
                network_info.paths.alls_script, args.model_script_path, args.hw_arch, performance=False
            )

            optimize_full_precision_model(
                runner, calib_feed_callback, logger, model_script, args.resize, args.input_conversion, args.classes
            )

        return configure_hef_tf1(args.hef_path, target) if args.hef_path else None

    if args.hef_path:
        return configure_hef_tf1(args.hef_path, target)

    _ensure_optimized(runner, logger, args, network_info)

    if isinstance(target, SdkPartialNumeric):
        return

    assert isinstance(target, Device)
    _ensure_compiled(runner, logger, args, network_info)
    return None


def _ensure_runnable_state_tf2(args, logger, network_info, runner, target):
    _ensure_parsed(runner, logger, network_info, args)
    if target == InferenceContext.SDK_FP_OPTIMIZED or (
        target == InferenceContext.SDK_HAILO_HW and args.hef_path is not None
    ):
        if runner.state != States.HAILO_MODEL:
            configure_hef_tf2(runner, args.hef_path)
            return

        # We intentionally use base model script and assume its modifications
        # compatible to the performance model script
        model_script = _extract_model_script_path(
            network_info.paths.alls_script, args.model_script_path, args.hw_arch, False
        )
        calib_feed_callback = prepare_calibration_data(
            runner, network_info, args.calib_path, logger, args.input_conversion, args.resize
        )
        optimize_full_precision_model(
            runner, calib_feed_callback, logger, model_script, args.resize, args.input_conversion, args.classes
        )
        configure_hef_tf2(runner, args.hef_path)

    else:
        configure_hef_tf2(runner, args.hef_path)

        _ensure_optimized(runner, logger, args, network_info)

        if target != InferenceContext.SDK_QUANTIZED:
            _ensure_compiled(runner, logger, args, network_info)

    return


def parse(args):
    logger = get_logger()
    nodes = [args.start_node_names, args.end_node_names]
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path, nodes=nodes)
    model_name = network_info.network.network_name
    logger.info(f"Start run for network {model_name} ...")

    logger.info("Initializing the runner...")
    runner = ClientRunner(hw_arch=args.hw_arch)
    parse_model(runner, network_info, ckpt_path=args.ckpt_path, results_dir=args.results_dir, logger=logger)


def optimize(args):
    logger = get_logger()
    nodes = [args.start_node_names, args.end_node_names]
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path, nodes=nodes)
    model_name = network_info.network.network_name
    logger.info(f"Start run for network {model_name} ...")

    if args.calib_path is None and network_info.quantization.calib_set is None:
        raise ValueError("Cannot run optimization without dataset. use --calib-path to provide external dataset.")

    logger.info(f"Initializing the {args.hw_arch} runner...")
    runner = ClientRunner(hw_arch=args.hw_arch, har=args.har_path)
    _ensure_parsed(runner, logger, network_info, args)

    model_script = _extract_model_script_path(
        network_info.paths.alls_script, args.model_script_path, args.hw_arch, args.performance
    )
    _ensure_performance(model_name, model_script, args.performance, args.hw_arch, logger)
    calib_feed_callback = prepare_calibration_data(
        runner, network_info, args.calib_path, logger, args.input_conversion, args.resize
    )
    optimize_model(
        runner,
        calib_feed_callback,
        logger,
        network_info,
        args.results_dir,
        model_script,
        args.resize,
        args.input_conversion,
        args.classes,
    )


def compile(args):
    logger = get_logger()
    nodes = [args.start_node_names, args.end_node_names]
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path, nodes=nodes)
    model_name = network_info.network.network_name
    logger.info(f"Start run for network {model_name} ...")

    logger.info(f"Initializing the {args.hw_arch} runner...")
    runner = ClientRunner(hw_arch=args.hw_arch, har=args.har_path)

    _ensure_optimized(runner, logger, args, network_info)

    model_script = _extract_model_script_path(
        network_info.paths.alls_script, args.model_script_path, args.hw_arch, args.performance
    )
    _ensure_performance(model_name, model_script, args.performance, args.hw_arch, logger)
    compile_model(runner, network_info, args.results_dir, model_script, performance=args.performance)

    logger.info(f"HEF file written to {get_hef_path(args.results_dir, network_info.network.network_name)}")


def profile(args):
    logger = get_logger()
    logger.deprecation_warning(
        (
            "'profile' command is deprecated and will be removed in future release."
            " Please use 'hailo profiler' tool instead."
        ),
        DeprecationVersion.FUTURE,
    )
    nodes = [args.start_node_names, args.end_node_names]
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path, nodes=nodes)
    model_name = network_info.network.network_name
    logger.info(f"Start run for network {model_name} ...")

    logger.info(f"Initializing the {args.hw_arch} runner...")
    runner = ClientRunner(hw_arch=args.hw_arch, har=args.har_path)

    _ensure_parsed(runner, logger, network_info, args)

    if args.hef_path and runner.state == States.HAILO_MODEL:
        model_script = _extract_model_script_path(
            network_info.paths.alls_script, args.model_script_path, args.hw_arch, args.performance
        )
        _ensure_performance(model_name, model_script, args.performance, args.hw_arch, logger)
        calib_feed_callback = prepare_calibration_data(
            runner, network_info, args.calib_path, logger, args.input_conversion, args.resize
        )
        optimize_full_precision_model(
            runner, calib_feed_callback, logger, model_script, args.resize, args.input_conversion, args.classes
        )

    export = runner.profile(should_use_logical_layers=True, hef_filename=args.hef_path)
    outpath = args.results_dir / f"{model_name}.html"
    report_generator = ReactReportGenerator(export, outpath)
    csv_data = report_generator.create_report(should_open_web_browser=False)
    logger.info(f"Profiler report generated in {outpath}")

    return export["stats"], csv_data, export["latency_data"]


def evaluate(args):
    logger = get_logger()

    if args.target == "hardware" and not HEF_EXISTS:
        raise ModuleNotFoundError(
            f"HailoRT is not available, in case you want to run on {args.target} you should install HailoRT first"
        )

    if (args.hw_arch == ["hailo15h", "hailo15m"] and args.target == "hardware") and not args.hailort_server_ip:
        raise ValueError("Evaluation of hw_arch hailo15h is currently not supported in the Hailo Model Zoo")

    if args.hef_path and not HEF_EXISTS:
        raise ModuleNotFoundError(
            "HailoRT is not available, in case you want to evaluate with hef you should install HailoRT first"
        )

    hardware_targets = set(DEVICE_NAMES)
    hardware_targets.update(["hardware"])
    if args.hef_path and args.target not in hardware_targets:
        raise ValueError(
            f"hef is not used when evaluating with {args.target}. use --target hardware for evaluating with a hef."
        )

    if args.video_outpath and not args.visualize_results:
        raise ValueError("The --video-output argument requires --visualize argument")

    nodes = [args.start_node_names, args.end_node_names]
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path, nodes=nodes)

    if args.data_path is None and network_info.evaluation.data_set is None:
        raise ValueError("Cannot run evaluation without dataset. use --data-path to provide external dataset.")

    if path := args.custom_infer_config:
        custom_infer_config = Path(path)
        if not custom_infer_config.is_file:
            raise ValueError(
                "The given path for '--custom-infer-file' is not a file, please provide a valid file for the argument"
            )
        if not args.target == "emulator":
            raise ValueError("custom_infer_config only works on target: emulator")

    model_name = network_info.network.network_name
    logger.info(f"Start run for network {model_name} ...")

    logger.info("Initializing the runner...")
    if args.hef_path and not args.har_path:
        logger.info(
            "Using HEF without specifying har_path requires a calibration set "
            "to build the network. Please refer to the DATA.rst file for detailed "
            "instructions on creating a calibration set"
        )
    runner = ClientRunner(hw_arch=args.hw_arch, har=args.har_path)

    #  Enabling service for hailo15h
    if args.hailort_server_ip:
        # This property will print a warning when set.
        runner.hailort_server_ip = args.hailort_server_ip

    logger.info(f"Chosen target is {args.target}")
    batch_size = args.batch_size or __get_batch_size(network_info, args.target)

    target = INFERENCE_TARGETS[args.target]
    _ensure_runnable_state_tf2(args, logger, network_info, runner, target)

    device_info = DEVICES.get(args.target)
    # overrides nms score threshold if postprocess on-host
    nms_score_threshold = (
        network_info["postprocessing"].get("score_threshold", None)
        if network_info["postprocessing"]["hpp"] and not network_info["postprocessing"]["bbox_decoding_only"]
        else None
    )
    context = runner.infer_context(
        target,
        device_ids=device_info,
        nms_score_threshold=nms_score_threshold,
        custom_infer_config=args.custom_infer_config,
    )
    return infer_model_tf2(
        runner,
        network_info,
        context,
        logger,
        args.eval_num_examples,
        args.data_path,
        batch_size,
        args.print_num_examples,
        args.visualize_results,
        args.video_outpath,
        args.use_lite_inference,
        dump_results=False,
        input_conversion_args=args.input_conversion,
        resize_args=args.resize,
        show_results_per_class=args.show_results_per_class,
    )


def __get_batch_size(network_info, target):
    if target == "full_precision":
        return network_info.inference.full_precision_batch_size
    return network_info.inference.emulator_batch_size
