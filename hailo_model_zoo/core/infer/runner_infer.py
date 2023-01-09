import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from hailo_model_zoo.core.infer.infer_utils import log_accuracy, write_results, save_image, get_logits_per_image


class ImageSaver:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def write(self, image, image_name):
        save_image(Image.fromarray(image), image_name)


class VideoWriter:
    def __init__(self, width, height, video_outpath):
        self.video_writer = cv2.VideoWriter(video_outpath,
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            24,
                                            (width, height))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.video_writer.release()

    def write(self, image, image_name):
        self.video_writer.write(image)


def _make_writer(info_per_image, video_outpath):
    if not video_outpath:
        writer = ImageSaver()
    else:
        ref_image = info_per_image[0]['img_orig']
        width, height = ref_image.shape[2], ref_image.shape[1]
        writer = VideoWriter(width, height, video_outpath)
    return writer


def _visualize(logits_batch, info_per_image, visualize_callback, video_outpath):
    with _make_writer(info_per_image, video_outpath) as writer:
        logits_per_image = get_logits_per_image(logits_batch)
        for image_index, (image_logits, image_info) in enumerate(zip(logits_per_image, info_per_image)):
            original_image = image_info['img_orig']
            image_name = image_info.get('image_name', f'image{image_index}')
            if hasattr(image_name, 'numpy'):
                image_name = image_name.numpy().decode('utf8')
            # Decode image if needed
            if type(original_image) is bytes:
                original_image = cv2.imdecode(np.fromstring(original_image, dtype=np.uint8),
                                              cv2.IMREAD_UNCHANGED)
            original_image = np.expand_dims(original_image, axis=0)

            image = visualize_callback(image_logits, original_image, img_info=image_info, image_name=image_name)
            writer.write(image, image_name)


def aggregate(elements):
    if not elements:
        return elements
    e = elements[0]
    assert tf.is_tensor(e)
    if e.dtype == tf.string:
        return elements
    if len(e.shape) == 0:
        return np.array(elements)
    return np.concatenate(elements, axis=0)


def runner_infer(runner, target, logger, eval_num_examples, print_num_examples,
                 batch_size, dataset, postprocessing_callback,
                 eval_callback, visualize_callback, video_outpath, dump_results, results_path):
    eval_metric = eval_callback()
    logger.info('Running inference...')

    logits = runner.infer(target, dataset, batch_size=batch_size, data_count=eval_num_examples)
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
        _visualize(probs, img_info_per_image, visualize_callback, video_outpath)

    return accuracy
