import os
from collections.abc import Mapping

import cv2
import numpy as np
from PIL import Image

from hailo_model_zoo.utils.numpy_utils import to_numpy


def save_image(img, image_name):
    if isinstance(image_name, bytes):
        image_name = image_name.decode("utf-8")
    img_name = os.path.splitext(image_name)[0]
    img_name = img_name.replace("/", "_")
    img.save("./{}_out.png".format(img_name))


def _get_logits(logits, idx, img_info):
    if isinstance(logits, list):
        ret = {}
        ret["logits"] = np.squeeze(logits[0])
        ret["image_info"] = {}
        ret["image_info"]["rpn_proposals"] = np.squeeze(logits[1])
        ret["image_info"]["num_rpn_proposals"] = logits[1].shape[0]
        img_info.pop("img_orig")
        for k in img_info:
            ret["image_info"].setdefault(k, {})
            ret["image_info"][k] = img_info[k][idx]
    elif type(logits) is np.ndarray:
        ret = logits[idx]
    elif isinstance(logits, dict):
        ret = {}
        for key in logits.keys():
            ret[key] = logits[key][idx]
    else:
        raise Exception("Logits structure is not recognize")
    return ret


def write_results(logits, img_info, results_directory):
    os.makedirs(results_directory, exist_ok=True)
    for idx, img_name in enumerate(img_info["image_name"]):
        data = {}
        base_image_name = os.path.basename(img_name.decode()).replace(".jpg", "")
        data[base_image_name] = _get_logits(logits, idx, img_info)
        filename = "{}.npz".format(base_image_name)
        np.savez(os.path.join(results_directory, filename), **data)


def log_degradation(logger, accuracies_output, accuracies_output_native):
    log = "Overall Degradation:"
    for result_native, result_quantized in zip(accuracies_output_native, accuracies_output):
        norm_coeff = 100.0 if result_native.is_percentage else 1.0
        diff_coeff = 1 if result_native.is_bigger_better else -1.0
        diff = result_native.value - result_quantized.value
        deg = diff * diff_coeff
        log += " {}={:.3f}".format(result_native.name, norm_coeff * deg)
    logger.info(log)
    return log


def log_accuracy(logger, num_of_images, accuracies_output):
    log = "Done {} images".format(num_of_images)
    for result in accuracies_output:
        norm_coeff = 100.0 if result.is_percentage else 1.0
        log += " {}={:.3f}".format(result.name, norm_coeff * result.value)
    logger.info(log)
    return log


def get_logits_per_image(logits):
    assert isinstance(logits, dict), f"Expected logits to be dict but got {type(logits)}"
    types = {type(v) for v in logits.values()}
    assert len(types) == 1, f"Assumed dict of lists or dict of arrays but got {types}"
    t = list(types)[0]

    if t is np.ndarray:
        # (BATCH, someshape) -> (BATCH, 1, someshape)
        expanded_vals = [np.expand_dims(v, axis=1) for v in logits.values()]
        return [dict(zip(logits, v)) for v in zip(*expanded_vals)]

    if t is list:
        return [dict(zip(logits, v)) for v in zip(*logits.values())]

    raise ValueError("Unsupported type {} for logits".format(type(logits)))


class ImageSaver:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def write(self, image, image_name):
        save_image(Image.fromarray(image), image_name)


class VideoWriter:
    def __init__(self, width, height, video_outpath):
        self.video_writer = cv2.VideoWriter(video_outpath, cv2.VideoWriter_fourcc(*"mp4v"), 24, (width, height))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.video_writer.release()

    def write(self, image, image_name):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.video_writer.write(image)


class RawWriter:
    def __init__(self, raw_suffix):
        self.raw_suffix = raw_suffix.numpy().decode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def write(self, image, image_name):
        with open(".".join([image_name, self.raw_suffix]), "wb") as outfile:
            outfile.write(image)


def _make_writer(info_per_image, video_outpath):
    if info_per_image[0].get("raw_suffix"):
        if video_outpath:
            raise ValueError("Impossible to write both raw and video in parallel")
        writer = RawWriter(info_per_image[0]["raw_suffix"])
    elif not video_outpath:
        writer = ImageSaver()
    else:
        ref_image = info_per_image[0]["img_orig"]
        width, height = ref_image.shape[-2], ref_image.shape[-3]
        writer = VideoWriter(width, height, video_outpath)
    return writer


class WriterHook:
    def __init__(self, visualize_callback, video_outpath) -> None:
        self.video_outpath = video_outpath
        self.visualize_callback = visualize_callback

        self.writer = None
        self.image_index = 0

    def __enter__(self):
        if self.writer:
            self.writer.__enter__()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.writer:
            self.writer.__exit__(exc_type, exc_value, exc_traceback)

    def visualize(self, image_logits, image_info):
        logits_per_image = get_logits_per_image(image_logits)
        batch_size = len(logits_per_image)
        # image_info could either be per_image list or dictionary with info of entire batch
        if isinstance(image_info, Mapping):
            image_info = [{k: v[i] for k, v in image_info.items()} for i in range(batch_size)]
        if not self.writer:
            self.writer = _make_writer(image_info, self.video_outpath)

        for image_index_in_batch, (image_logits, img_info) in enumerate(zip(logits_per_image, image_info)):
            image_index = image_index_in_batch + self.image_index
            original_image = img_info["img_orig"]
            original_image = to_numpy(original_image)
            image_name = img_info.get("image_name", f"image{image_index}")
            image_name = to_numpy(image_name, decode=True)
            # Decode image if needed
            if isinstance(original_image, bytes):
                original_image = cv2.imdecode(np.fromstring(original_image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            original_image = np.expand_dims(original_image, axis=0)

            image = self.visualize_callback(image_logits, original_image, img_info=img_info, image_name=image_name)
            self.writer.write(image, image_name)
        self.image_index += image_index_in_batch


def visualize(logits_batch, info_per_image, visualize_callback, video_outpath):
    with WriterHook(visualize_callback, video_outpath) as writer:
        writer.visualize(logits_batch, info_per_image)


def aggregate(elements):
    if not elements:
        return elements
    e = elements[0]
    # we got a list instead of tensor - flatten the list of lists
    if isinstance(e, list):
        return [item for sublist in elements for item in sublist]

    # we got primitives - collect them to an array
    if len(e.shape) == 0 or min(e.shape) == 0:  # added to handle SparseTensor
        return np.array(elements)

    # we got multiple numpy arrays, concatenate them
    return np.concatenate(elements, axis=0)
