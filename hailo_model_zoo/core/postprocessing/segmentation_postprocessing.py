import tensorflow as tf
import numpy as np
from PIL import ImageFilter
from PIL import Image


class PostProcessingException(Exception):
    pass


def get_dataset_colors(dataset):
    colors = {'cityscapes': np.array([[128., 64., 128., 115.],  # orange - road 1
                                      [244., 35., 232., 115.],  # orange - sidewalk 2
                                      [70., 70., 70., 115.],  # yellow - building 3
                                      [102., 102., 156., 115.],  # yellow - wall 4
                                      [190., 153., 153., 115.],  # yellow - fence 5
                                      [153., 153., 153., 115.],  # purple - pole 6
                                      [250., 170., 30., 115.],  # purple - trafficliight 7
                                      [220., 220., 0., 115.],  # purple - trafficsign 8
                                      [107., 142., 35., 115.],  # orange - vegetation 9
                                      [152., 251., 152., 115.],  # orange - terrain 10
                                      [70., 130., 180., 115.],  # blue - sky 11
                                      [220., 20., 60., 115.],  # red - person 12
                                      [255., 0., 0., 115.],  # green - rider 13
                                      [0., 0., 142., 115.],  # green - car 14
                                      [0., 0., 70., 115.],  # green - truck 15
                                      [0., 60., 100., 115.],  # green - bus 16
                                      [0., 80., 100., 115.],  # green - train 17
                                      [0., 0., 230., 115.],  # green - motorcycle 18
                                      [119., 11., 32., 115.]])}  # green - bicycle 19
    return colors[dataset]


def color_segment_img(orig_img, logits, dataset):
    """Colors a given image and return it

    Args:
      orig_img: the input image to the net. The size of this image must be the same as data
      logits: the logits to color

    Returns:
      Colored image
    """
    colors = get_dataset_colors(dataset)
    seg_mask = np.array(np.take(colors, np.array(logits, np.uint8), axis=0), np.uint8)
    image_in_walpha = np.concatenate((orig_img, (255 - colors[1, 3]) * np.ones(orig_img.shape[:2] + (1,))), axis=2)
    composite = Image.alpha_composite(Image.fromarray(np.uint8(image_in_walpha)),
                                      Image.fromarray(np.uint8(np.squeeze(seg_mask))))
    composite_backscaled = composite.resize(orig_img.shape[1::-1], Image.LANCZOS)
    return np.array(composite_backscaled, np.uint8)[:, :, :3]


def segmentation_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    device_pre_post_layers = device_pre_post_layers if device_pre_post_layers is not None else {
        'bilinear': False, 'argmax': False}
    if device_pre_post_layers['argmax']:
        predictions = endnodes
    else:
        if device_pre_post_layers['bilinear']:
            logits = endnodes
        else:
            size = np.array(endnodes.shape[1:3]) * kwargs['ext_upsample']
            logits = tf.compat.v1.image.resize_bilinear(endnodes, size=size.tolist(), align_corners=True)
        predictions = tf.cast(tf.argmax(logits, axis=-1), tf.float32)
    return {'predictions': predictions}


def visualize_segmentation_result(logits, image, **kwargs):
    logits = logits['predictions']
    dataset = kwargs['dataset_name']
    if dataset == 'coco_segmentation':
        blur_img = Image.fromarray(np.array(image[0], np.uint8)).filter(ImageFilter.GaussianBlur(radius=4))
        mask_3d = logits[0, :, :, None] * np.ones(3, dtype=int)[None, None, :]
        blur_img = np.array(blur_img) * mask_3d
        img_out = np.array(image[0] * (1 - mask_3d) + blur_img, np.uint8)
    elif dataset == 'cityscapes':
        img_out = color_segment_img(image[0], logits[0], dataset)
    else:
        raise PostProcessingException("Visualization is not implemented for dataset {}".format(dataset))
    return img_out
