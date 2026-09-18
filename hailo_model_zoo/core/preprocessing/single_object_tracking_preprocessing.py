import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY

TEMPLATE_SIZE = 128


@PREPROCESS_FACTORY.register(name="single_object_tracking")
def single_object_tracking_preprocessing(image, image_info=None, height=None, width=None, **kwargs):
    """Preprocessing for the STARK tracker.

    Two paths, because tracking calibration and tracking evaluation need different data:

    * **Calibration** -- the record already holds a cropped template/search pair, so
      emit the two-input dict directly.
    * **Evaluation** -- the record is a whole frame. The template and search crops both
      depend on the tracker's own previous prediction, which the tf.data pipeline cannot
      know, so the frame is passed through untouched and ``tracker_infer`` crops it
      inside the closed loop.

    Normalization is folded into the network (``normalize_in_net`` in the yaml), so both
    paths hand over raw uint8 RGB.
    """
    if image_info is not None and "template" in image_info:
        network_name = kwargs.get("network_name")
        template = tf.cast(image_info.pop("template"), tf.float32)
        search = tf.cast(image, tf.float32)
        return {
            f"{network_name}/input_layer1": template,
            f"{network_name}/input_layer2": search,
        }, image_info

    image = tf.cast(image, tf.uint8)
    if image_info is not None:
        image_info["img_orig"] = image
    return image, image_info
