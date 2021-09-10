import tensorflow as tf
from tqdm import tqdm
from PIL import Image

from hailo_model_zoo.core.infer.infer_utils import log_accuracy, write_results, save_image


def facenet_infer(runner, target, logger, eval_num_examples, print_num_examples,
                  batch_size, data_feed_callback, tf_graph_callback, postprocessing_callback,
                  eval_callback, visualize_callback, video_outpath, dump_results, results_path):
    with tf.Graph().as_default():
        logger.info('Building preprocess...')
        iterator = data_feed_callback()
        [preprocessed_data, image_info] = iterator.get_next()
        preprocessed_data = tf.reshape(
            tf.concat([preprocessed_data, tf.image.flip_left_right(preprocessed_data)], axis=1),
            [-1, int(preprocessed_data.shape[1]), int(preprocessed_data.shape[2]), 3])
        image_info = image_info['image_name'], image_info['is_same']

        logger.info('Compiling and integrating with Tensorflow graph...')
        sdk_export = tf_graph_callback(preprocessed_data)
        probs = postprocessing_callback(tf.reshape(sdk_export.output_tensors[0],
                                                   [-1, 2 * int(sdk_export.output_tensors[0].shape[1])]))
        eval_metric = eval_callback()

        logger.info('Running inference...')
        with sdk_export.session.as_default(), runner.hef_infer_context(sdk_export):
            sdk_export.session.run([iterator.initializer, tf.compat.v1.local_variables_initializer()])
            num_of_images = 0
            try:
                with tqdm(total=None, desc="Processed", unit="images",
                          disable=None if not print_num_examples < 1e9 else True) as pbar:
                    while num_of_images < eval_num_examples:
                        logits_batch, img_info = sdk_export.session.run([probs, image_info])
                        logits_batch = logits_batch['predictions']
                        num_of_images += int(logits_batch.shape[0])
                        if not visualize_callback and not dump_results:
                            eval_metric.update_op({'predictions': logits_batch}, img_info)
                            if num_of_images % print_num_examples == 0:
                                eval_metric.evaluate()
                                log_accuracy(logger, num_of_images, eval_metric.get_accuracy())
                        else:
                            if visualize_callback:
                                save_image(Image.fromarray(visualize_callback(logits_batch, img_info['img_orig'])),
                                           img_info['image_name'][0])
                            if dump_results:
                                write_results(logits_batch, img_info, results_path)
                        pbar.update(batch_size)
            except tf.errors.OutOfRangeError:
                pass

        # Write message and exit if we finished to iterate over the data
        if not visualize_callback and not dump_results and num_of_images % print_num_examples != 0:
            eval_metric.evaluate()
            log_accuracy(logger, num_of_images, eval_metric.get_accuracy())
    return eval_metric.get_accuracy()
