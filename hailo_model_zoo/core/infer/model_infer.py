import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.core.infer.infer_utils import log_accuracy, write_results, aggregate, visualize, to_numpy
from hailo_model_zoo.core.factory import INFER_FACTORY


@INFER_FACTORY.register
def model_infer(runner, context, logger, eval_num_examples, print_num_examples,
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
            pbar.update(current_batch_size)
            logits.append(logits_batch)
            if not visualize_callback:
                if "img_orig" in img_info:
                    del img_info["img_orig"]
                if "img_resized" in img_info:
                    del img_info["img_resized"]
            gt.append(to_numpy(img_info))
    labels_keys = list(gt[0].keys())
    labels = {k: aggregate([p[k] for p in gt]) for k in labels_keys}
    probs = {k: aggregate([p[k] for p in logits]) for k in logits[0].keys()}
    accuracy = None
    if not visualize_callback and not dump_results:
        eval_metric.update_op(probs, labels)
        eval_metric.evaluate()
        accuracy = eval_metric.get_accuracy()
        log_accuracy(logger, num_of_images, accuracy)
    if dump_results:
        write_results(probs, labels, results_path)

    if visualize_callback:
        img_info_per_image = [x[1] for x in dataset]
        visualize(probs, img_info_per_image, visualize_callback, video_outpath)
    return accuracy


@INFER_FACTORY.register
def np_infer(*args, **kwargs):
    return model_infer(*args, **kwargs, np_infer=True),


INFER_FACTORY.register(model_infer, name="facenet_infer")
