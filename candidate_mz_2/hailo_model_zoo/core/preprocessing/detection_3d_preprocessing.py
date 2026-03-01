import numpy as np
import tensorflow as tf
from PIL import Image

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY


@PREPROCESS_FACTORY.register(name="petrv2_transformer")
def petrv2_repvggB0_transformer_pp_800x320(images, image_info=None, height=None, width=None, **kwargs):
    mlvl_feats = images["mlvl_feats"]
    coords_3d = images["coords_3d"]
    net_name = kwargs.get("network_name", None)
    image = {
        f"{net_name}/input_layer1": mlvl_feats,
        f"{net_name}/input_layer2": coords_3d,
    }
    return image, image_info


def _resize_and_crop(image, bot_pct_lim, orig_height=None, orig_width=None, height=None, width=None):
    resize = max(height / orig_height, width / orig_width)
    resize_dims = (int(orig_width * resize), int(orig_height * resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.mean(bot_pct_lim)) * newH) - height
    crop_w = int(max(0, newW - width) / 2)
    crop = (crop_w, crop_h, crop_w + width, crop_h + height)

    if len(image.shape) == 3:
        img = Image.fromarray(np.uint8(image))
        img = img.resize(resize_dims)
        img = img.crop(crop)
        return img

    images = []
    for img in image:
        img = Image.fromarray(np.uint8(img))
        img = img.resize(resize_dims)
        img = img.crop(crop)
        images.append(np.array(img))
    return np.array(images)


@PREPROCESS_FACTORY.register(name="petrv2_backbone")
def petrv2_repvggB0_backbone_pp_800x320(images, image_info=None, height=None, width=None, **kwargs):
    bot_pct_lim = tf.constant([0.0, 0.0])
    H, W = image_info["orig_height"], image_info["orig_width"]
    img = tf.numpy_function(_resize_and_crop, [images, bot_pct_lim, H, W, height, width], [tf.uint8])[0]
    img.set_shape((height, width, 3))

    return img, image_info


def petrv2_repvggB0_backbone_pp_800x320_cascade(images, image_info=None, height=None, width=None, **kwargs):
    bot_pct_lim = tf.constant([0.0, 0.0])
    H, W = image_info["orig_height"], image_info["orig_width"]
    img = tf.numpy_function(_resize_and_crop, [images, bot_pct_lim, H, W, height, width], [tf.uint8])[0]
    img.set_shape((None, height, width, 3))

    return img, image_info
