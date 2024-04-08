import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY


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
                                      [119., 11., 32., 115.]]),  # green - bicycle 19
              'pascal': np.array([[200., 200., 200., 115.],   # grey - background 0
                                  [244., 35., 232., 115.],    # orange - aeroplane 1
                                  [70., 70., 70., 115.],      # yellow - bicycle 2
                                  [102., 102., 156., 115.],   # yellow - bird 3
                                  [190., 153., 153., 115.],   # yellow - boat 4
                                  [153., 153., 153., 115.],   # purple - bottle 5
                                  [250., 170., 30., 115.],    # purple - bus 6
                                  [220., 220., 0., 115.],     # purple - car 7
                                  [107., 142., 35., 115.],    # orange - cat 8
                                  [244., 97., 66., 115.],     # orange - chair 9
                                  [70., 130., 180., 115.],    # blue   - cow 10
                                  [220., 20., 60., 115.],     # red    - diningtable 11
                                  [255., 0., 0., 115.],       # green  - dog 12
                                  [0., 0., 142., 115.],       # green  - horse 13
                                  [0., 0., 70., 115.],        # green  - motorbike 14
                                  [64., 248., 162., 115.],    # green  - person 15
                                  [174., 239., 162., 115.],   # green  - pottedplant 16
                                  [0., 230., 0., 115.],       # green  - sheep 17
                                  [128., 146., 85., 115.],    # green  - sofa 18
                                  [64., 187., 144., 115.],    # green  - train 19
                                  [89., 197., 132., 115.]])}  # green  - tvmonitor 20
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


@POSTPROCESS_FACTORY.register(name="segmentation")
def segmentation_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    device_pre_post_layers = device_pre_post_layers if device_pre_post_layers is not None else {
        'bilinear': False, 'argmax': False}
    if device_pre_post_layers['argmax']:
        if len(endnodes.shape) == 4:
            endnodes = tf.squeeze(endnodes, axis=-1)
        predictions = endnodes
    else:
        if device_pre_post_layers['bilinear']:
            logits = endnodes
        else:
            size = np.array(endnodes.shape[1:3]) * kwargs['ext_upsample']
            logits = tf.image.resize(endnodes, size=size.tolist(), method='bilinear')
        predictions = tf.cast(tf.argmax(logits, axis=-1), tf.float32)
    return {'predictions': predictions}


@VISUALIZATION_FACTORY.register(name="segmentation")
def visualize_segmentation_result(logits, image, **kwargs):
    logits = logits['predictions']
    dataset = kwargs['dataset_name']
    height, width = logits.squeeze().shape
    image = Image.fromarray(image.astype(np.uint8).squeeze()).resize((width, height))
    image = np.array(image)
    if dataset == 'coco_segmentation':
        blur_img = Image.fromarray(np.array(image, np.uint8)).filter(ImageFilter.GaussianBlur(radius=4))
        mask_3d = logits[0, :, :, None] * np.ones(3, dtype=int)[None, None, :]
        blur_img = np.array(blur_img) * mask_3d
        img_out = np.array(image * (1 - mask_3d) + blur_img, np.uint8)
    elif dataset == 'cityscapes':
        img_out = color_segment_img(image, logits[0], dataset)
    elif dataset == 'pascal':
        img_out = color_segment_img(image, logits[0], dataset)
    else:
        raise PostProcessingException("Visualization is not implemented for dataset {}".format(dataset))
    return img_out
