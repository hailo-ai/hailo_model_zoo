from hailo_model_zoo.core.factory import INFER_FACTORY
from hailo_model_zoo.core.infer.infer_utils import aggregate, log_accuracy, visualize, write_results


@INFER_FACTORY.register
def runner_infer(
    runner,
    context,
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
    results_path,
):
    eval_metric = eval_callback()
    logger.info("Running inference...")
    with context as ctx:
        logits = runner.infer(context=ctx, dataset=dataset, batch_size=batch_size, data_count=eval_num_examples)
    num_of_images = logits.shape[0]
    img_info_per_image = [x[1] for x in dataset.take(num_of_images)]
    img_info = {k: aggregate([p[k] for p in img_info_per_image]) for k in img_info_per_image[0].keys()}
    probs = postprocessing_callback(logits)
    accuracy = None
    if not visualize_callback and not dump_results:
        eval_metric.update_op(probs, img_info)
        eval_metric.evaluate()
        accuracy = eval_metric.get_accuracy()
        log_accuracy(logger, num_of_images, accuracy)
    if dump_results:
        write_results(logits, img_info, results_path)

    if visualize_callback:
        visualize(probs, img_info_per_image, visualize_callback, video_outpath)

    return accuracy
