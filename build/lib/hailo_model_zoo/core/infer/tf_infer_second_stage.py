import tensorflow as tf

from hailo_model_zoo.core.infer.infer_utils import log_accuracy


def tf_infer_second_stage(runner, target, logger, eval_num_examples, print_num_examples,
                          batch_size, data_feed_callback, tf_graph_callback, postprocessing_callback,
                          eval_callback, visualize_callback, video_outpath, dump_results, results_path):
    with tf.Graph().as_default():
        logger.info('Building preprocess...')
        iterator = data_feed_callback()
        [preprocessed_data, image_info] = iterator.get_next()

        logger.info('Compiling and integrating with Tensorflow graph...')
        sdk_export = tf_graph_callback(preprocessed_data)
        if len(sdk_export.output_tensors) == 1:
            probs = postprocessing_callback(sdk_export.output_tensors[0])
        else:
            probs = postprocessing_callback(sdk_export.output_tensors, image_info=image_info)
        eval_metric = eval_callback()
        logger.info('Running inference...')
        # Can't initialize video_writer here because we don't know the image width/height until the first session.run
        video_writer = None
        with sdk_export.session.as_default(), runner.hef_infer_context(sdk_export):
            sdk_export.session.run([iterator.initializer])
            number_proposals = 0
            try:
                print_num_examples = 100
                overall_processed_images = 0
                while True:
                    if eval_num_examples is not None and eval_metric.num_evaluated_images >= eval_num_examples:
                        break
                    logits_batch, img_info = sdk_export.session.run([probs, image_info])
                    # Try to get the actual batch size from img_info (since last batch could be smaller)
                    number_proposals += batch_size
                    if not visualize_callback and not dump_results:
                        eval_metric.update_op(logits_batch, img_info)
                        if eval_metric.num_evaluated_images > overall_processed_images \
                                and eval_metric.num_evaluated_images % print_num_examples == 0:
                            eval_metric.evaluate()
                            log_accuracy(logger, eval_metric.num_evaluated_images, eval_metric.get_accuracy())
                            overall_processed_images = eval_metric.num_evaluated_images
            except tf.errors.OutOfRangeError:
                pass
            finally:
                if video_writer:
                    video_writer.release()

        # # Write message and exit if we finished to iterate over the data
        if not visualize_callback and not dump_results and eval_metric.num_evaluated_images % print_num_examples != 0:
            eval_metric.evaluate(force_last_img=True)
            log_accuracy(logger, eval_metric.num_evaluated_images, eval_metric.get_accuracy())
    return eval_metric.get_accuracy()
