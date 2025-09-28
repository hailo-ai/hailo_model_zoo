import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from hailo_model_zoo.core.factory import INFER_FACTORY
from hailo_model_zoo.core.infer.infer_utils import get_logits_per_image, log_accuracy, save_image, write_results


def _visualize(logits_batch, img_info, num_of_images, visualize_callback, video_outpath, video_writer):
    logits_per_image = get_logits_per_image(logits_batch)
    batch_size = len(logits_per_image)
    image_names = img_info.get(
        "image_name", ["image{}".format(num_of_images - batch_size + i).encode("utf8") for i in range(batch_size)]
    )
    info_per_image = [{k: v[i] for k, v in img_info.items()} for i in range(batch_size)]
    for image_logits, original_image, image_name, image_info in zip(
        logits_per_image, img_info["img_orig"], image_names, info_per_image
    ):
        # Decode image if needed
        if isinstance(original_image, bytes):
            original_image = cv2.imdecode(np.fromstring(original_image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        original_image = np.expand_dims(original_image, axis=0)

        image = visualize_callback(image_logits, original_image, img_info=image_info, image_name=image_name)
        if not video_outpath:
            save_image(Image.fromarray(image), image_name)
        else:
            width, height = (original_image.shape[2], original_image.shape[1])
            video_writer = video_writer or cv2.VideoWriter(
                video_outpath, cv2.VideoWriter_fourcc(*"mp4v"), 24, (width, height)
            )
            video_writer.write(image)

    return video_writer


@INFER_FACTORY.register
def tf_infer(
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
    results_path,
):
    with tf.Graph().as_default():
        logger.info("Building preprocess...")
        iterator = data_feed_callback()
        [preprocessed_data, image_info] = iterator.get_next()

        logger.info("Compiling and integrating with Tensorflow graph...")
        sdk_export = tf_graph_callback(preprocessed_data)
        if len(sdk_export.output_tensors) == 1:
            probs = postprocessing_callback(sdk_export.output_tensors[0], gt_images=image_info)
        else:
            probs = postprocessing_callback(sdk_export.output_tensors, gt_images=image_info)
        eval_metric = eval_callback()
        logger.info("Running inference...")
        # Can't initialize video_writer here because we don't know the image width/height until the first session.run
        video_writer = None
        with sdk_export.session.as_default(), runner.hef_infer_context(sdk_export):
            sdk_export.session.run([iterator.initializer])
            num_of_images = 0
            try:
                with tqdm(
                    total=eval_num_examples,
                    desc="Processed",
                    unit="images",
                    disable=None if not print_num_examples < 1e9 else True,
                ) as pbar:
                    while True:
                        if eval_num_examples is not None and num_of_images >= eval_num_examples:
                            break
                        logits_batch, img_info = sdk_export.session.run([probs, image_info])
                        # Try to get the actual batch size from img_info (since last batch could be smaller)
                        current_batch_size = len(img_info["img_orig"]) if "img_orig" in img_info else batch_size
                        num_of_images += current_batch_size
                        if not visualize_callback and not dump_results:
                            eval_metric.update_op(logits_batch, img_info)
                            if num_of_images % print_num_examples == 0:
                                eval_metric.evaluate()
                                log_accuracy(logger, num_of_images, eval_metric.get_accuracy())
                        else:
                            if visualize_callback:
                                video_writer = _visualize(
                                    logits_batch,
                                    img_info,
                                    num_of_images,
                                    visualize_callback,
                                    video_outpath,
                                    video_writer,
                                )
                            if dump_results:
                                write_results(logits_batch, img_info, results_path)
                        pbar.update(current_batch_size)
            except tf.errors.OutOfRangeError:
                pass
            finally:
                if video_writer:
                    video_writer.release()

        # Write message and exit if we finished to iterate over the data
        if not visualize_callback and not dump_results and num_of_images % print_num_examples != 0:
            eval_metric.evaluate()
            log_accuracy(logger, num_of_images, eval_metric.get_accuracy())
    return eval_metric.get_accuracy()
