from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.utils.platform_discovery import PLATFORM_AVAILABLE

if PLATFORM_AVAILABLE:
    from hailo_platform import FormatType, VDevice

from hailo_sdk_client import InferenceContext

from hailo_model_zoo.core.datasets.parse_nuscenes import parse_petrv2_cascade_record
from hailo_model_zoo.core.eval.detection_3d_evaluation import PETRv2EvalCascade
from hailo_model_zoo.core.factory import CASCADE_FACTORY
from hailo_model_zoo.core.infer.infer_utils import log_accuracy, to_numpy
from hailo_model_zoo.core.postprocessing.detection_3d_postprocessing import petrv2_postprocess
from hailo_model_zoo.core.preprocessing.detection_3d_preprocessing import petrv2_repvggB0_backbone_pp_800x320_cascade
from hailo_model_zoo.utils import path_resolver


def make_dataset(tfrecord_path, input_shape):
    tfrecord_path = path_resolver.resolve_data_path(tfrecord_path)
    dataset = tf.data.TFRecordDataset([str(tfrecord_path)])
    dataset = dataset.map(parse_petrv2_cascade_record)

    height, width = input_shape

    def preprocess(*args, **kwargs):
        return petrv2_repvggB0_backbone_pp_800x320_cascade(*args, **kwargs, height=height, width=width)

    dataset = dataset.map(preprocess)
    return dataset


def verify_config(cfg):
    if cfg.batch_size != 12:
        raise ValueError(
            (
                "Only batch size 12 is supported (input frames) "
                "which are required to produce a single BEV prediction in PETRv2"
            )
        )

    if cfg.models.backbone.har is None:
        raise ValueError("har for backbone model cannot be None")

    if cfg.models.transformer.har is None:
        raise ValueError("har for transformer model cannot be None")


@CASCADE_FACTORY.register
def petrv2(
    models,
    logger,
    cfg,
):
    verify_config(cfg)

    ref_points_path = path_resolver.resolve_data_path(cfg.ref_points_path)
    if not Path(ref_points_path).is_file():
        raise FileNotFoundError(f"Could not find {ref_points_path}")

    has_hw_target = (
        models["backbone"].target is InferenceContext.SDK_HAILO_HW
        or models["transformer"].target is InferenceContext.SDK_HAILO_HW
    )
    if not PLATFORM_AVAILABLE and has_hw_target:
        raise ValueError("hardware target selected but hailo_platform is not available")

    eval_metric = PETRv2EvalCascade(
        dataset_name="nuscenes", gt_json_path=path_resolver.resolve_data_path(cfg.gt_json_path)
    )

    input_shape = models["backbone"].get_input_shapes()[0][1:3]
    dataset = make_dataset(cfg.data_path, input_shape)
    if cfg.data_count:
        dataset = dataset.take(cfg.data_count)
    batched_dataset = dataset

    ref_points = np.load(ref_points_path)

    network_name = models["transformer"].info.network.network_name
    transformer_input_shape = models["transformer"].get_input_shapes()[0]

    def predict_batch(preprocessed_data, coords3d, timestamp, classes):
        output_tensors = backbone(preprocessed_data)

        logits = transformer(
            {
                f"{network_name}/input_layer1": tf.reshape(output_tensors, transformer_input_shape),
                f"{network_name}/input_layer2": tf.expand_dims(coords3d, axis=0),
            }
        )
        output = petrv2_postprocess(
            logits,
            timestamp,
            ref_points,
            classes,
        )
        return output

    if not has_hw_target:
        predict_batch = tf.function(predict_batch, jit_compile=True, reduce_retracing=True)
    logger.info("Running inference...")
    with ExitStack() as stack:
        device = stack.enter_context(VDevice()) if has_hw_target else None
        hef_kwargs = (
            None
            if not PLATFORM_AVAILABLE
            else {
                "batch_size": cfg.batch_size,
                "input_format_type": FormatType.UINT8,
            }
        )
        backbone = models["backbone"].get_keras_model(stack, device, hef_kwargs)
        transformer = models["transformer"].get_keras_model(stack, device)

        num_images = 0

        for preprocessed_data, img_info in tqdm(batched_dataset, desc="Processed", unit="images"):
            coords3d = img_info["coords3d"]
            output = predict_batch(
                preprocessed_data,
                coords3d,
                img_info["timestamp"],
                models["transformer"].info.evaluation.classes,
            )
            output["predictions"] = output["predictions"][0]
            output = to_numpy(output)
            img_info = to_numpy(img_info)
            eval_metric.update_op(output, img_info)
            num_images += 1

        eval_metric.evaluate()
        accuracy = eval_metric.get_accuracy()
        log_accuracy(logger, num_images, accuracy)
