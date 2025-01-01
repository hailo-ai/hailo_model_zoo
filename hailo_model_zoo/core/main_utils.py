import argparse
import json
from pathlib import Path

from omegaconf.listconfig import ListConfig

from hailo_sdk_client.exposed_definitions import States
from hailo_sdk_client.tools.parser_cli import NetParser

from hailo_model_zoo.core.augmentations import make_model_callback
from hailo_model_zoo.core.eval import eval_factory
from hailo_model_zoo.core.hn_editor.activations_changer import change_activations
from hailo_model_zoo.core.hn_editor.channel_remove import channel_remove
from hailo_model_zoo.core.hn_editor.channel_transpose import bgr2rgb
from hailo_model_zoo.core.hn_editor.layer_splitter import LayerSplitter
from hailo_model_zoo.core.hn_editor.network_chainer import integrate_postprocessing
from hailo_model_zoo.core.hn_editor.normalization_folding import fold_normalization
from hailo_model_zoo.core.infer import infer_factory
from hailo_model_zoo.core.info_utils import get_network_info  # noqa (F401) - exports this function for backwards compat
from hailo_model_zoo.core.postprocessing import postprocessing_factory
from hailo_model_zoo.core.preprocessing import preprocessing_factory
from hailo_model_zoo.utils import data, downloader, path_resolver
from hailo_model_zoo.utils.parse_utils import get_normalization_params

unsupported_data_folder = {"stereonet"}


def _get_input_shape(runner):
    return runner.get_native_hn_model().get_input_shapes(ignore_conversion=True)[0][1:4]


def _get_output_shapes(runner):
    return [
        output_layer.output_shape
        for output_layer in runner.get_hn_model().get_output_layers(remove_non_neural_core_layers=False)
    ]


def resolve_alls_path(path, hw_arch="hailo8", performance=False):
    if not path:
        return None
    return path_resolver.resolve_alls_path(Path(hw_arch) / Path("base" if not performance else "performance") / path)


def is_network_performance(model_name, hw_arch):
    return path_resolver.get_alls_path(Path(hw_arch) / Path("performance") / Path(f"{model_name}.alls")).is_file()


def _apply_output_scheme(runner, network_info):
    output_scheme = network_info.hn_editor.output_scheme

    if not output_scheme:
        return

    split_output = output_scheme.split_output
    if split_output:
        split_fc = any("fc" in x for x in output_scheme.outputs_to_split)
        layer_splitter = LayerSplitter(runner, network_info, split_fc)
        runner = layer_splitter.modify_network()

    activation_changes = output_scheme.change_activations
    if activation_changes and activation_changes.enabled:
        change_activations(runner, activation_changes)


def get_integrated_postprocessing(network_info):
    output_scheme = network_info.hn_editor.output_scheme
    return output_scheme.integrated_postprocessing


def _add_postprocess(runner, network_info):
    integrated_postprocessing = get_integrated_postprocessing(network_info)
    if integrated_postprocessing and integrated_postprocessing.enabled:
        integrate_postprocessing(runner, integrated_postprocessing, network_info)


def download_model(network_info, logger):
    network_path = network_info.paths.network_path
    ckpt_path = path_resolver.resolve_model_path(network_path)
    # we don't use resolve_model_path, as it strips the extension from .ckpt files
    paths = [path_resolver.resolve_data_path(path) for path in network_path]
    # Check that all files exist. FUTURE: verify checksum
    if all(path.is_file() for path in paths):
        return ckpt_path

    url = network_info.paths.url
    downloader.download(url, ckpt_path.parent, logger)
    return ckpt_path


def parse_model(runner, network_info, *, ckpt_path=None, results_dir=Path("."), logger=None):
    """Parses TF or ONNX model and saves as <results_dir>/<model_name>.(hn|npz)"""
    start_node_shapes = network_info.parser.start_node_shapes
    if isinstance(start_node_shapes, ListConfig):
        start_node_shapes = list(start_node_shapes)

    # we don't try to download the file in case the ckpt_path is overridden by the user
    if ckpt_path is None:
        ckpt_path = download_model(network_info, logger)

    model_name = network_info.network.network_name
    start_node_names, end_node_names = network_info.parser.nodes[0:2]

    parser_args = argparse.Namespace(
        net_name=model_name,
        input_framework=str(ckpt_path).split(".")[-1],
        input_format=None,
        model_path=str(ckpt_path),
        tensor_shapes=start_node_shapes,
        start_node_names=start_node_names,
        end_node_names=end_node_names,
        y=True,
        hw_arch=runner.hw_arch,
        har_path=None,
        augmented_path=None,
        disable_rt_metadata_extraction=False,
        parsing_report_path=None,
        compare=False,
    )
    parser = NetParser(argparse.ArgumentParser(description="HailoMZ parser"))
    try:
        runner = parser.run(parser_args, save_model=False)
    except Exception as err:
        raise Exception(f"Encountered error during parsing: {err}") from None

    _apply_output_scheme(runner, network_info)

    hn_editor = network_info.hn_editor
    channels_remove = hn_editor.channels_remove
    bgr_to_rgb = hn_editor.bgr2rgb

    if channels_remove.enabled and any(x is not None for x in channels_remove.values()):
        channel_remove(runner, channels_remove)
    if bgr_to_rgb:
        bgr2rgb(runner)

    # hn editing
    if network_info.parser.normalization_params.fold_normalization:
        normalize_in_net, mean_list, std_list = get_normalization_params(network_info)
        assert normalize_in_net, f"fold_normalization implies normalize_in_net==true, but got {normalize_in_net}"
        fold_normalization(runner, mean_list, std_list)

    _add_postprocess(runner, network_info)

    # save model
    runner.save_har(results_dir / f"{network_info.network.network_name}.har")

    return runner


def load_model(runner, har_path, logger):
    logger.info(f"Loading {har_path}")
    runner.load_har(har_path)


def get_input_modifications(runner, network_info, input_conversion_args=None, resize_args=None):
    def _is_yuv2rgb(conversion_type):
        return conversion_type in ["yuv_to_rgb", "yuy2_to_rgb", "nv12_to_rgb"]

    def _is_yuv2(conversion_type):
        return conversion_type in ["yuy2_to_hailo_yuv", "yuy2_to_rgb"]

    def _is_nv12(conversion_type):
        return conversion_type in ["nv12_to_hailo_yuv", "nv12_to_rgb"]

    def _is_rgbx(conversion_type):
        return conversion_type == "tf_rgbx_to_hailo_rgb"

    hn_editor = network_info.hn_editor
    yuv2rgb = hn_editor.yuv2rgb if not input_conversion_args else _is_yuv2rgb(input_conversion_args)
    yuy2 = hn_editor.yuy2 if not input_conversion_args else _is_yuv2(input_conversion_args)
    nv12 = hn_editor.nv12 if not input_conversion_args else _is_nv12(input_conversion_args)
    rgbx = hn_editor.rgbx if not input_conversion_args else _is_rgbx(input_conversion_args)
    if resize_args:
        hn_editor.input_resize.enabled = True
        hn_editor.input_resize.input_shape = [*resize_args]
    input_resize = hn_editor.input_resize

    for configs in runner.modifications_meta_data.inputs.values():
        for config in configs:
            if config.cmd_type == "input_conversion":
                if config.emulate_conversion:
                    yuv2rgb = yuv2rgb or _is_yuv2rgb(config.conversion_type.value)
                    yuy2 = yuy2 or _is_yuv2(config.conversion_type.value)
                    nv12 = nv12 or _is_nv12(config.conversion_type.value)
                    rgbx = rgbx or _is_rgbx(config.conversion_type.value)
            elif config.cmd_type == "resize":
                input_resize["enabled"] = True
                input_resize["input_shape"] = [config.output_shape[1], config.output_shape[2]]

    return yuv2rgb, yuy2, nv12, rgbx, input_resize


def make_preprocessing(runner, network_info, input_conversion_args=None, resize_args=None):
    preprocessing_args = network_info.preprocessing
    meta_arch = preprocessing_args.get("meta_arch")
    yuv2rgb, yuy2, nv12, rgbx, input_resize = get_input_modifications(
        runner, network_info, input_conversion_args, resize_args
    )
    normalize_in_net, mean_list, std_list = get_normalization_params(network_info)
    normalization_params = [mean_list, std_list] if not normalize_in_net else None
    height, width, channels = _get_input_shape(runner)
    flip = runner.get_hn_model().is_transposed()
    preproc_callback = preprocessing_factory.get_preprocessing(
        meta_arch,
        height=height,
        width=width,
        flip=flip,
        yuv2rgb=yuv2rgb,
        yuy2=yuy2,
        nv12=nv12,
        rgbx=rgbx,
        input_resize=input_resize,
        normalization_params=normalization_params,
        output_shapes=_get_output_shapes(runner),
        network_name=network_info.network.network_name,
        channels=channels,
        **preprocessing_args,
    )

    return preproc_callback


def _make_data_feed_callback(data_path, dataset_name, two_stage_arch, preproc_callback, network_info, batch_size=None):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Couldn't find dataset in {data_path}. Please refer to docs/DATA.rst.")
    data_tf = data_path.is_file() and data_path.suffix in [".tfrecord", ".000"]
    args = (preproc_callback, batch_size, data_path)
    if two_stage_arch:
        func = data.RegionProposalFeed
    elif data_tf:
        func = data.TFRecordFeed
        args += (dataset_name,)
    elif data_path.is_dir():
        if network_info.network.network_name in unsupported_data_folder:
            raise ValueError(f"Folder data path is currently not supported for {network_info.network.network_name}")
        func = data.ImageFeed
    else:
        if network_info.network.network_name in unsupported_data_folder:
            raise ValueError(f"Video feed is currently not supported for {network_info.network.network_name}")
        func = data.VideoFeed
    return lambda: func(*args)


def _make_dataset_callback(network_info, preproc_callback, absolute_path, dataset_name, batch_size=None):
    two_stage_arch = network_info.evaluation.two_stage_arch
    return _make_data_feed_callback(
        absolute_path, dataset_name, two_stage_arch, preproc_callback, network_info, batch_size=batch_size
    )


def make_evalset_callback(network_info, preproc_callback, override_path, return_iterator_cb=False, batch_size=None):
    dataset_name = network_info.evaluation.dataset_name
    resolved_data_path = override_path or path_resolver.resolve_data_path(network_info.evaluation.data_set)
    data_feed_cb = _make_dataset_callback(
        network_info, preproc_callback, resolved_data_path, dataset_name, batch_size=batch_size
    )
    if return_iterator_cb:
        if batch_size is None:
            raise ValueError("Batch size is required for iterator")
        return lambda: data_feed_cb().iterator
    return data_feed_cb().dataset


def make_calibset_callback(network_info, preproc_callback, override_path):
    dataset_name = network_info.quantization.calib_set_name
    if override_path is None and network_info.quantization.calib_set is None:
        raise ValueError("Optimization requires calibration data, please modify YAML or use --calib-path")
    calib_path = override_path or path_resolver.resolve_data_path(network_info.quantization.calib_set[0])
    data_feed_cb = _make_dataset_callback(network_info, preproc_callback, calib_path, dataset_name)
    return lambda: data_feed_cb().dataset


def _handle_classes_argument(runner, logger, classes):
    script_commands = runner.model_script.split("\n")
    nms_idx = ["nms_postprocess" in x for x in script_commands]
    if not any(nms_idx):
        logger.warning("Ignoring classes parameter since the model has no NMS post-process.")
        return

    nms_idx = nms_idx.index(True)
    nms_command = script_commands[nms_idx]
    nms_args = nms_command[:-1].split("(", 1)[-1].split(", ")
    arg_to_append = f"classes={classes}"
    if "classes" in nms_command:
        classes_idx = ["classes" in x for x in nms_args].index(True)
        nms_args.pop(classes_idx)
    elif ".json" in nms_command:
        # Duplicate the config file, edit the classes and update the path in the command.
        path_idx = [".json" in x for x in nms_args].index(True)
        orig_path = nms_args[path_idx].split("=")[-1].replace('"', "").replace("'", "")
        with open(orig_path, "r") as f:
            nms_cfg = json.load(f)
        nms_cfg["classes"] = classes
        tmp_path = f"{orig_path.split('.json')[0]}_tmp.json"
        with open(tmp_path, "w") as f:
            json.dump(nms_cfg, f, indent=4)
        nms_args.pop(path_idx)
        arg_to_append = f'config_path="{tmp_path}"'

    nms_args.append(arg_to_append)
    script_commands[nms_idx] = f'nms_postprocess({", ".join(nms_args)})'
    runner.load_model_script("\n".join(script_commands))


def prepare_calibration_data(runner, network_info, calib_path, logger, input_conversion_args=None, resize_args=None):
    logger.info("Preparing calibration data...")
    preproc_callback = make_preprocessing(runner, network_info, input_conversion_args, resize_args)
    calib_feed_callback = make_calibset_callback(network_info, preproc_callback, calib_path)
    return calib_feed_callback


def optimize_full_precision_model(runner, calib_feed_callback, logger, model_script, resize, input_conversion, classes):
    runner.load_model_script(model_script)
    if runner.state != States.HAILO_MODEL:
        return
    if classes is not None:
        _handle_classes_argument(runner, logger, classes)
    input_layers = runner.get_hn_model().get_input_layers()
    scope_name = input_layers[0].scope
    if resize is not None:
        height, width = resize
        resize_layer_names = ", ".join(
            f"{input_layer.scope}/resize_input{i}" for i, input_layer in enumerate(input_layers, start=1)
        )
        runner.load_model_script(f"{resize_layer_names} = resize(resize_shapes=[{height},{width}])", append=True)
    if input_conversion is not None:
        hailo_conversion_type = {"rgbx_to_rgb": "tf_rgbx_to_hailo_rgb"}.get(input_conversion, input_conversion)
        conversion_layers = [f"{scope_name}/input_conversion1"]
        if hailo_conversion_type in ["yuy2_to_rgb", "nv12_to_rgb"]:
            conversion_layers.append(f"{scope_name}/yuv_to_rgb1")
        runner.load_model_script(
            f'{", ".join(conversion_layers)} = input_conversion({hailo_conversion_type}, emulator_support=True)',
            append=True,
        )
    runner.optimize_full_precision(calib_data=calib_feed_callback)


def optimize_model(
    runner,
    calib_feed_callback,
    logger,
    network_info,
    results_dir,
    model_script,
    resize=None,
    input_conversion=None,
    classes=None,
):
    optimize_full_precision_model(runner, calib_feed_callback, logger, model_script, resize, input_conversion, classes)

    runner.optimize(calib_feed_callback)

    model_name = network_info.network.network_name
    runner.save_har(results_dir / f"{model_name}.har")


def make_visualize_callback(network_info):
    network_type = network_info.preprocessing.network_type
    meta_arch = network_info.postprocessing.meta_arch
    dataset_name = network_info.evaluation.dataset_name
    channels_remove = network_info.hn_editor.channels_remove
    labels_offset = network_info.evaluation.labels_offset
    classes = network_info.evaluation.classes
    visualize_function = postprocessing_factory.get_visualization(network_type)

    # TODO NET-4282: Move this logic to the relevant place once pipeline flow is in place
    if isinstance(visualize_function, type):
        vis_args = {
            "network_type": network_type,
            "meta_arch": meta_arch,
            "dataset_name": dataset_name,
            "channels_remove": channels_remove,
            "labels_offset": labels_offset,
            "classes": classes,
        }
        visualize_function = visualize_function(**vis_args)

    def visualize_callback(logits, image, **kwargs):
        return visualize_function(
            logits,
            image,
            dataset_name=dataset_name,
            channels_remove=channels_remove,
            labels_offset=labels_offset,
            meta_arch=meta_arch,
            classes=classes,
            **kwargs,
        )

    return visualize_callback


def _gather_postprocessing_dictionary(runner, network_info):
    height, width, _ = _get_input_shape(runner)
    postproc_info = {"img_dims": (height, width)}
    postproc_info.update(network_info.hn_editor)
    postproc_info.update(network_info.evaluation)
    postproc_info.update(network_info.preprocessing)
    postproc_info.update(network_info.postprocessing)
    return postproc_info


def get_postprocessing_callback(runner, network_info):
    postproc_info = _gather_postprocessing_dictionary(runner, network_info)
    network_type = postproc_info.pop("network_type")
    device_pre_post_layers = postproc_info.pop("device_pre_post_layers")
    flip = postproc_info.pop("flip")

    postproc_callback = postprocessing_factory.get_postprocessing(network_type, flip=flip)

    def postprocessing_callback(endnodes, gt_images=None, image_info=None, **kwargs):
        probs = postproc_callback(
            endnodes=endnodes,
            device_pre_post_layers=device_pre_post_layers,
            gt_images=gt_images,
            image_info=image_info,
            **postproc_info,
            **kwargs,
        )
        return probs

    return postprocessing_callback


def make_eval_callback(network_info, runner, show_results_per_class, logger):
    network_type = network_info.evaluation.network_type
    if show_results_per_class and network_type not in ["detection", "instance_segmentation"]:
        logger.info(
            "print-AP-per-class flag is available only for object detection or "
            "instance segmentation tasks, ignoring flag"
        )
        show_results_per_class = False
    net_name = network_info.network.network_name
    gt_json_path = network_info.evaluation.gt_json_path
    meta_arch = network_info.evaluation.meta_arch
    gt_json_path = path_resolver.resolve_data_path(gt_json_path) if gt_json_path else None
    input_shape = _get_input_shape(runner)[:2]
    eval_args = {
        "net_name": net_name,
        "network_type": network_type,
        "labels_offset": network_info.evaluation.labels_offset,
        "labels_map": network_info.evaluation.labels_map,
        "classes": network_info.evaluation.classes,
        "gt_labels_path": path_resolver.resolve_data_path(network_info.quantization.calib_set[0]).parent,
        "ckpt_path": path_resolver.resolve_model_path(network_info.paths.network_path),
        "channels_remove": network_info.hn_editor.channels_remove,
        "dataset_name": network_info.evaluation.dataset_name,
        "gt_json_path": gt_json_path,
        "centered": network_info.preprocessing.centered,
        "nms_iou_thresh": network_info.postprocessing.nms_iou_thresh,
        "score_threshold": network_info.postprocessing.score_threshold,
        "input_shape": input_shape,
        "meta_arch": meta_arch,
        "mask_thresh": network_info.postprocessing.mask_threshold,
        "show_results_per_class": show_results_per_class,
    }

    evaluation_constructor = eval_factory.get_evaluation(network_type)

    def eval_callback():
        return evaluation_constructor(**eval_args)

    return eval_callback


def get_infer_type(network_info, use_lite_inference):
    infer_type = network_info.evaluation.infer_type
    if not use_lite_inference:
        return infer_type

    # infer type is already lite
    if infer_type.endswith("_lite"):
        return infer_type

    # make sure infer type actually applies
    if infer_type not in ["np_infer", "model_infer"]:
        raise ValueError(f"lite inference can only be used with np_infer or model_infer but used with {infer_type}")

    final_infer_type = infer_type + "_lite"
    return final_infer_type


def make_infer_callback(network_info, use_lite_inference):
    infer_type = get_infer_type(network_info, use_lite_inference)
    infer_callback = infer_factory.get_infer(infer_type)

    return infer_callback


def infer_model_tf2(
    runner,
    network_info,
    target,
    logger,
    eval_num_examples,
    data_path,
    batch_size,
    print_num_examples=256,
    visualize_results=False,
    video_outpath=None,
    use_lite_inference=False,
    dump_results=False,
    input_conversion_args=None,
    resize_args=None,
    show_results_per_class=False,
):
    logger.info("Initializing the dataset ...")
    if eval_num_examples:
        eval_num_examples = eval_num_examples + network_info.evaluation.data_count_offset
    preproc_callback = make_preprocessing(runner, network_info, input_conversion_args, resize_args)
    # we do not pass batch_size, batching is now done in infer_callback
    dataset = make_evalset_callback(network_info, preproc_callback, data_path)
    # TODO refactor
    postprocessing_callback = get_postprocessing_callback(runner, network_info)
    eval_callback = make_eval_callback(network_info, runner, show_results_per_class, logger)
    visualize_callback = make_visualize_callback(network_info) if visualize_results else None

    model_wrapper_callback = make_model_callback(network_info)
    infer_callback = make_infer_callback(network_info, use_lite_inference)
    return infer_callback(
        runner,
        target,
        logger,
        eval_num_examples,
        print_num_examples,
        batch_size,
        dataset,
        postprocessing_callback,
        eval_callback,
        visualize_callback,
        model_wrapper_callback,
        video_outpath,
        dump_results,
        results_path=None,
    )


def infer_model_tf1(
    runner,
    network_info,
    target,
    logger,
    eval_num_examples,
    data_path,
    batch_size,
    print_num_examples=256,
    visualize_results=False,
    video_outpath=None,
    dump_results=False,
    network_groups=None,
    show_results_per_class=False,
):
    logger.info("Initializing the dataset ...")
    if eval_num_examples:
        eval_num_examples = eval_num_examples + network_info.evaluation.data_count_offset
    preproc_callback = make_preprocessing(runner, network_info)
    data_feed_callback = make_evalset_callback(
        network_info, preproc_callback, data_path, return_iterator_cb=True, batch_size=batch_size
    )

    def tf_graph_callback(preprocessed_data, rescale_output=None):
        sdk_export = runner.get_tf_graph(
            target,
            preprocessed_data,
            use_preloaded_compilation=True,
            network_groups=network_groups,
            rescale_output=rescale_output,
        )

        return sdk_export

    postprocessing_callback = get_postprocessing_callback(runner, network_info)
    eval_callback = make_eval_callback(network_info, runner, show_results_per_class, logger)
    visualize_callback = make_visualize_callback(network_info) if visualize_results else None

    infer_type = network_info.evaluation.infer_type
    infer_callback = infer_factory.get_infer(infer_type)

    return infer_callback(
        runner,
        target,
        logger,
        eval_num_examples,
        print_num_examples,
        batch_size,
        data_feed_callback,
        tf_graph_callback,
        postprocessing_callback,
        eval_callback,
        visualize_callback,
        video_outpath,
        dump_results,
        results_path=None,
    )


def get_hef_path(results_dir, model_name):
    return results_dir.joinpath(f"{model_name}.hef")


def compile_model(runner, network_info, results_dir, allocator_script_filename, performance=False):
    model_name = network_info.network.network_name
    model_script_parent = None
    if allocator_script_filename is not None:
        allocator_script_filename = Path(allocator_script_filename)
        runner.load_model_script(allocator_script_filename)
        model_script_parent = allocator_script_filename.parent.name
    if performance:
        if model_script_parent == "generic" or allocator_script_filename is None:
            runner.load_model_script("performance_param(compiler_optimization_level=max)", append=True)
    hef = runner.compile()

    with open(get_hef_path(results_dir, model_name), "wb") as hef_out_file:
        hef_out_file.write(hef)

    runner.save_har(results_dir / f"{model_name}.har")
