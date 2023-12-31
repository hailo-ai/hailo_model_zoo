import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.core.infer.infer_utils import log_accuracy, aggregate, visualize, to_numpy


def model_infer_lite(runner, context, logger, eval_num_examples, print_num_examples,
                     batch_size, dataset, postprocessing_callback,
                     eval_callback, visualize_callback, model_augmentation_callback,
                     video_outpath, dump_results, results_path, *, np_infer=False):
    eval_metric = eval_callback()
    if not np_infer:
        postprocessing_callback = tf.function(postprocessing_callback, reduce_retracing=True)
    if eval_num_examples:
        dataset = dataset.take(eval_num_examples)
    batched_dataset = dataset.batch(batch_size)
    logger.info('Running inference...')
    with context as ctx, tqdm(total=None, desc="Processed", unit="images",
                              disable=None if not print_num_examples < 1 else True) as pbar:
        model = runner.get_keras_model(ctx)
        model = model_augmentation_callback(model)

        @tf.function()
        def predict_function(data):
            return model(data, training=False)

        num_of_images = 0
        logits = []
        gt = []
        for preprocessed_data, img_info in batched_dataset:
            output_tensors = predict_function(preprocessed_data)
            if np_infer:
                output_tensors = to_numpy(output_tensors)
                img_info = to_numpy(img_info)
            logits_batch = postprocessing_callback(output_tensors, gt_images=img_info)
            current_batch_size = (output_tensors[0].shape[0] if isinstance(output_tensors, list)
                                  else output_tensors.shape[0])
            num_of_images += current_batch_size
            if not visualize_callback:
                if "img_orig" in img_info:
                    del img_info["img_orig"]
                if "img_resized" in img_info:
                    del img_info["img_resized"]
            image_info = to_numpy(img_info)
            if not visualize_callback and not dump_results:
                logits_batch = to_numpy(logits_batch)
                eval_metric.update_op(logits_batch, image_info)
            if visualize_callback:
                logits.append(logits_batch)
                gt.append(image_info)
            pbar.update(current_batch_size)

    accuracy = None
    if not visualize_callback and not dump_results:
        eval_metric.evaluate()
        accuracy = eval_metric.get_accuracy()
        log_accuracy(logger, num_of_images, accuracy)

    if visualize_callback:
        probs = {k: aggregate([p[k] for p in logits]) for k in logits[0].keys()}
        img_info_per_image = [x[1] for x in dataset]
        visualize(probs, img_info_per_image, visualize_callback, video_outpath)
    return accuracy
