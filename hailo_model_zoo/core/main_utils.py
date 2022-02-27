from omegaconf import OmegaConf
from pathlib import Path
from functools import lru_cache

from hailo_sdk_common.targets.inference_targets import ParamsKinds

from hailo_sdk_client.tools.core_postprocess.core_postprocess_api import MetaArchitectures, add_nms_postprocess
from hailo_sdk_client.tools.hn_modifications import transpose_hn_height_width
from hailo_sdk_client.tools.hn_modifications import add_yuv_to_rgb_layers, add_resize_input_layers

from hailo_model_zoo.core.infer import infer_factory
from hailo_model_zoo.core.eval import eval_factory
from hailo_model_zoo.core.postprocessing import postprocessing_factory
from hailo_model_zoo.core.preprocessing import preprocessing_factory

from hailo_model_zoo.core.hn_editor.activations_changer import change_activations
from hailo_model_zoo.core.hn_editor.channel_remove import channel_remove
from hailo_model_zoo.core.hn_editor.channel_transpose import bgr2rgb
from hailo_model_zoo.core.hn_editor.layer_splitter import LayerSplitter
from hailo_model_zoo.core.hn_editor.network_chainer import integrate_postprocessing
from hailo_model_zoo.core.hn_editor.normalization_folding import fold_normalization
from hailo_model_zoo.core.hn_editor.remove_skip_connection import skip_connection

from hailo_model_zoo.utils import data, downloader, path_resolver
from hailo_model_zoo.utils.parse_utils import translate_model, get_normalization_params


def _get_input_shape(runner, network_info):
    return (network_info.preprocessing.input_shape or runner.get_hn_model().get_input_layers()[0].output_shape[1:])


def resolve_alls_path(path):
    if not path:
        return None

    return path_resolver.resolve_alls_path(path)


def _apply_output_scheme(runner, network_info):
    output_scheme = network_info.hn_editor.output_scheme

    if not output_scheme:
        return

    split_output = output_scheme.split_output
    if split_output:
        if any(["fc" in x for x in output_scheme.outputs_to_split]):
            split_fc = True
        else:
            split_fc = False
        layer_splitter = LayerSplitter(runner, network_info, split_fc)
        runner = layer_splitter.modify_network()

    activation_changes = output_scheme.change_activations
    if activation_changes and activation_changes.enabled:
        change_activations(runner, activation_changes)

    integrated_postprocessing = output_scheme.integrated_postprocessing
    if integrated_postprocessing and integrated_postprocessing.enabled:
        integrate_postprocessing(runner, integrated_postprocessing)


def get_network_info(model_name, read_only=False, yaml_path=None):
    '''
    Args:
        model_name: The network name to load.
        read_only: If set return read-only object.
                   The read_only mode save run-time and memroy.
        yaml_path: Path to external YAML file for network configuration
    Return:
        OmegaConf object that represent network configuration.
    '''
    if model_name is None and yaml_path is None:
        raise ValueError("Either model_name or yaml_path must be given")
    net = f'networks/{model_name}.yaml'
    cfg_path = Path(yaml_path) if yaml_path is not None else path_resolver.resolve_cfg_path(net)
    if not cfg_path.is_file():
        raise ValueError('cfg file is missing in {}'.format(cfg_path))
    cfg = _load_cfg(cfg_path)
    if read_only:
        OmegaConf.set_readonly(cfg, True)
        return cfg
    else:
        cfg_copy = cfg.copy()
        OmegaConf.set_readonly(cfg_copy, False)
        return cfg_copy


@lru_cache()
def _load_cfg(cfg_file):
    cfg = OmegaConf.load(cfg_file)
    base = cfg.get('base')
    if not base:
        return cfg
    del cfg['base']
    # if extension exists make a recursive call for each extension
    config = OmegaConf.create()
    for f in base:
        cfg_path = path_resolver.resolve_cfg_path(f)
        config = OmegaConf.merge(config, _load_cfg(cfg_path))
    # override with cfg fields
    config = OmegaConf.merge(config, cfg)
    return config


def download_model(network_info, logger):
    network_path = network_info.paths.network_path
    ckpt_path = path_resolver.resolve_model_path(network_path)
    # we don't use resolve_model_path, as it strips the extension from .ckpt files
    paths = [path_resolver.resolve_data_path(path) for path in network_path]
    # Check that all files exist. FUTURE: verify checksum
    if all([path.is_file() for path in paths]):
        return ckpt_path
    url = network_info.paths.url
    downloader.download(url, ckpt_path.parent, logger)
    return ckpt_path


def parse_model(runner, network_info, *, ckpt_path=None, results_dir=Path("."), logger=None):
    """Parses TF or ONNX model and saves as <results_dir>/<model_name>.(hn|npz)"""
    start_node_shape = network_info.parser.start_node_shape

    # we don't try to download the file in case the ckpt_path is overridden by the user
    if ckpt_path is None:
        ckpt_path = download_model(network_info, logger)

    model_name = translate_model(runner, network_info, ckpt_path, tensor_shapes=start_node_shape)

    # hn editing
    if network_info.parser.normalization_params.fold_normalization:
        normalize_in_net, mean_list, std_list = get_normalization_params(network_info)
        assert normalize_in_net, f"fold_normalization implies normalize_in_net==true, but got {normalize_in_net}"
        fold_normalization(runner, mean_list, std_list)

    hn_editor = network_info.hn_editor
    flip = hn_editor.flip
    channels_remove = hn_editor.channels_remove
    input_resize = hn_editor.input_resize
    bgr_to_rgb = hn_editor.bgr2rgb
    yuv_to_rgb = hn_editor.yuv2rgb
    skip_connection_remove = hn_editor.skip_connection_remove
    postprocess_config_json = network_info.postprocessing.postprocess_config_json
    if postprocess_config_json and postprocess_config_json.endswith('.json'):
        config_json = path_resolver.resolve_data_path(postprocess_config_json)
        add_nms_postprocess(
            runner,
            config_json_path=str(config_json),
            meta_arch=MetaArchitectures[network_info.postprocessing.meta_arch.upper()],
            params_kind=ParamsKinds.NATIVE,
        )
    if flip:
        transpose_hn_height_width(runner)
    if channels_remove.enabled and any(x is not None for x in channels_remove.values()):
        channel_remove(runner, channels_remove)
    if skip_connection_remove:
        skip_connection(runner, skip_connection_remove)
    if bgr_to_rgb:
        bgr2rgb(runner)
    if yuv_to_rgb:
        add_yuv_to_rgb_layers(runner)
    if input_resize.enabled:
        add_resize_input_layers(runner, input_resize.input_shape)

    _apply_output_scheme(runner, network_info)

    # save model
    runner.save_har(results_dir / f'{model_name}.har')


def load_model(runner, har_path, logger):
    logger.info(f"Loading {har_path}")
    runner.load_har(har_path)


def make_preprocessing(runner, network_info):
    preprocessing_args = network_info.preprocessing
    meta_arch = preprocessing_args.get('meta_arch')
    hn_editor = network_info.hn_editor
    flip = hn_editor.flip
    yuv2rgb = hn_editor.yuv2rgb
    input_resize = hn_editor.input_resize
    normalize_in_net, mean_list, std_list = get_normalization_params(network_info)
    normalization_params = [mean_list, std_list] if not normalize_in_net else None
    height, width, _ = _get_input_shape(runner, network_info)

    preproc_callback = preprocessing_factory.get_preprocessing(
        meta_arch, height=height, width=width, flip=flip, yuv2rgb=yuv2rgb,
        input_resize=input_resize, normalization_params=normalization_params,
        **preprocessing_args)

    return preproc_callback


def _make_data_feed_callback(batch_size, data_path, dataset_name, two_stage_arch, preproc_callback):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Couldn't find dataset in {data_path}. Please refer to docs/DATA.md.")
    data_tf = True if data_path.is_file() and data_path.suffix in [".tfrecord", ".000"] else False
    args = (preproc_callback, batch_size, data_path)

    if two_stage_arch:
        func = data.RegionProposalFeed
    elif data_tf:
        func = data.TFRecordFeed
        args += (dataset_name,)
    elif data_path.is_dir():
        func = data.ImageFeed
    else:
        func = data.VideoFeed
    return lambda: func(*args)


def _make_dataset_callback(network_info, batch_size, preproc_callback, absolute_path, dataset_name):
    two_stage_arch = network_info.evaluation.two_stage_arch
    return _make_data_feed_callback(batch_size, absolute_path, dataset_name, two_stage_arch, preproc_callback)


def make_evalset_callback(network_info, batch_size, preproc_callback, override_path):
    dataset_name = network_info.evaluation.dataset_name
    resolved_data_path = override_path or path_resolver.resolve_data_path(network_info.evaluation.data_set)
    data_feed_cb = _make_dataset_callback(network_info, batch_size, preproc_callback, resolved_data_path, dataset_name)
    return lambda: data_feed_cb().iterator


def make_calibset_callback(network_info, batch_size, preproc_callback, override_path):
    dataset_name = network_info.quantization.calib_set_name or network_info.evaluation.dataset_name
    calib_path = override_path or path_resolver.resolve_data_path(network_info.quantization.calib_set[0])
    data_feed_cb = _make_dataset_callback(network_info, batch_size, preproc_callback, calib_path, dataset_name)
    return data_feed_cb()._dataset.unbatch()


def quantize_model(runner, logger, network_info, calib_path, results_dir):
    quant_batch_size = network_info.quantization.quantization_batch_size

    logger.info('Using batch size of {} for quantization'.format(quant_batch_size))
    preproc_callback = make_preprocessing(runner, network_info)
    calib_feed_callback = make_calibset_callback(network_info, quant_batch_size, preproc_callback, calib_path)
    batch_size = network_info.quantization.quantization_batch_size
    calib_set_size = network_info.quantization.calib_set_size

    calib_num_batch = calib_set_size // batch_size
    if calib_set_size % batch_size != 0:
        raise ValueError(
            "Using only {} images for calibration instead of {}, because {} is not an integer product "
            "of the batch size {}".format(batch_size * calib_num_batch, calib_set_size, calib_set_size, batch_size)
        )

    max_elementwise_feed_repeat = network_info.allocation.max_elementwise_feed_repeat
    quantization_script_filename = resolve_alls_path(network_info.paths.alls_script)

    runner.quantize(calib_feed_callback, calib_num_batch=calib_num_batch,
                    batch_size=batch_size,
                    max_elementwise_feed_repeat=max_elementwise_feed_repeat,
                    model_script=quantization_script_filename,
                    work_dir=None)

    model_name = network_info.network.network_name
    runner.save_har(results_dir / f'{model_name}.har')


def make_visualize_callback(network_info):
    network_type = network_info.preprocessing.network_type
    dataset_name = network_info.evaluation.dataset_name
    channels_remove = network_info.hn_editor.channels_remove
    visualize_function = postprocessing_factory.get_visualization(network_type)

    def visualize_callback(logits, image, **kwargs):
        return visualize_function(logits, image, dataset_name=dataset_name, channels_remove=channels_remove, **kwargs)
    return visualize_callback


def _gather_postprocessing_dictionary(runner, network_info):
    height, width, _ = _get_input_shape(runner, network_info)
    postproc_info = dict(img_dims=(height, width))
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

    def postprocessing_callback(endnodes, gt_images=None, image_info=None):
        probs = postproc_callback(endnodes=endnodes, device_pre_post_layers=device_pre_post_layers,
                                  gt_images=gt_images, image_info=image_info, **postproc_info)
        return probs

    return postprocessing_callback


def make_eval_callback(network_info):
    network_type = network_info.evaluation.network_type or network_info.preprocessing.network_type
    gt_json_path = network_info.evaluation.gt_json_path
    gt_json_path = path_resolver.resolve_data_path(gt_json_path) if gt_json_path else None

    eval_args = dict(
        network_type=network_type,
        labels_offset=network_info.evaluation.labels_offset,
        labels_map=network_info.evaluation.labels_map,
        classes=network_info.evaluation.classes,
        gt_labels_path=path_resolver.resolve_data_path(network_info.quantization.calib_set[0]).parent,
        ckpt_path=path_resolver.resolve_model_path(network_info.paths.network_path),
        channels_remove=network_info.hn_editor.channels_remove,
        dataset_name=network_info.evaluation.dataset_name,
        gt_json_path=gt_json_path,
        centered=network_info.preprocessing.centered,
        nms_iou_thresh=network_info.postprocessing.nms_iou_thresh,
        score_threshold=network_info.postprocessing.score_threshold,
        input_shape=network_info.preprocessing.input_shape,
    )

    evaluation_constructor = eval_factory.get_evaluation(network_type)

    def eval_callback():
        return evaluation_constructor(**eval_args)

    return eval_callback


def infer_model(runner, network_info, target, logger, eval_num_examples,
                data_path, batch_size, print_num_examples=256, visualize_results=False,
                video_outpath=None, dump_results=False, network_groups=None):
    logger.info('Initializing the dataset ...')
    preproc_callback = make_preprocessing(runner, network_info)
    data_feed_callback = make_evalset_callback(network_info, batch_size, preproc_callback, data_path)

    def tf_graph_callback(preprocessed_data):
        sdk_export = runner.get_tf_graph(
            target, preprocessed_data,
            use_preloaded_compilation=True,
            network_groups=network_groups)

        return sdk_export

    postprocessing_callback = get_postprocessing_callback(runner, network_info)
    eval_callback = make_eval_callback(network_info)
    visualize_callback = make_visualize_callback(network_info) if visualize_results else None

    infer_type = network_info.evaluation.infer_type
    infer_callback = infer_factory.get_infer(infer_type)

    return infer_callback(runner, target, logger, eval_num_examples, print_num_examples, batch_size,
                          data_feed_callback, tf_graph_callback, postprocessing_callback, eval_callback,
                          visualize_callback, video_outpath, dump_results, results_path=None)


def get_hef_path(results_dir, model_name):
    return results_dir.joinpath(f"{model_name}.hef")


def compile_model(runner, network_info, results_dir):
    fps = network_info.allocation.required_fps
    model_name = network_info.network.network_name
    allocator_script_filename = resolve_alls_path(network_info.paths.alls_script)
    mapping_timeout = network_info.allocation.allocation_timeout
    hef = runner.get_hw_representation(
        fps=fps,
        mapping_timeout=mapping_timeout,
        allocator_script_filename=allocator_script_filename
    )

    with open(get_hef_path(results_dir, model_name), "wb") as hef_out_file:
        hef_out_file.write(hef)

    runner.save_har(results_dir / f'{model_name}.har')


def info_model(model_name, network_info, logger):

    def build_dict(info):
        keys_list = ['task', 'input_shape', 'output_shape', 'operations', 'parameters',
                     'framework', 'training_data', 'validation_data', 'eval_metric',
                     'full_precision_result', 'source', 'license_url']
        info_vals = [info[key_curr] for key_curr in keys_list]
        info_dict = dict(zip(keys_list, info_vals))
        return keys_list, info_dict

    keys_list, info_dict = build_dict(network_info['info'])
    msgs_list = list()
    for key_curr in keys_list:
        msg = "\t{0:<25}{1}".format(key_curr + ':', info_dict[key_curr])
        msgs_list.append(msg)
    msg_w_line = '\033[0m\n' + "\n".join(msgs_list)
    logger.info(msg_w_line)
