import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from hailo_model_zoo.core.infer.infer_utils import log_accuracy, save_image, write_results


def np_infer(
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
        eval_metric = eval_callback()

        logger.info("Running inference...")
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
                        logits_batch, img_info = sdk_export.session.run([sdk_export.output_tensors, image_info])
                        num_of_images += batch_size
                        if len(logits_batch) == 1:
                            logits_batch = logits_batch[0]
                        probs = postprocessing_callback(logits_batch, gt_images=img_info)
                        if not visualize_callback and not dump_results:
                            eval_metric.update_op(probs, img_info)
                            if num_of_images % print_num_examples == 0:
                                eval_metric.evaluate()
                                log_accuracy(logger, num_of_images, eval_metric.get_accuracy())
                        else:
                            if visualize_callback:
                                save_image(
                                    Image.fromarray(visualize_callback(probs, img_info["img_orig"])),
                                    img_info["image_name"][0],
                                )
                            if dump_results:
                                write_results(probs, img_info, results_path)
                        pbar.update(batch_size)
            except tf.errors.OutOfRangeError:
                pass

        # Write message and exit if we finished to iterate over the data
        if not visualize_callback and not dump_results and num_of_images % print_num_examples != 0:
            eval_metric.evaluate()
            log_accuracy(logger, num_of_images, eval_metric.get_accuracy())
    return eval_metric.get_accuracy()
