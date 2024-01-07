import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageDraw


def stable_diffusion_v2_decoder_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    output_image = tf.clip_by_value(endnodes / 2 + 0.5, 0.0, 1.0)
    return {'predictions': output_image}


def visualize_stable_diffusion_v2_decoder(logits, img_gt, **kwargs):
    max_chars_per_line = 80
    image_gen = logits['predictions']
    image_gen = (image_gen * 255).round().astype("uint8")
    image_gen_pil = Image.fromarray(image_gen.squeeze())
    draw = ImageDraw.Draw(image_gen_pil)
    text = kwargs['img_info']['prompt'].numpy().decode('utf-8')
    text_color = "black"
    chunks = [text[i:i + max_chars_per_line] for i in range(0, len(text), max_chars_per_line)]
    for i, c in enumerate(chunks):
        text_position = (0, i * 10)
        c = c.encode('utf-8').decode('latin-1')
        draw.text(text_position, c, fill=text_color)
    return np.array(image_gen_pil, np.uint8)


def stable_diffusion_v2_unet_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    endnodes = tf.transpose(endnodes, (0, 2, 3, 1))
    return {'predictions': endnodes}


def visualize_stable_diffusion_v2_unet(logits, img_gt, **kwargs):
    max_chars_per_line = 80
    image_gen = logits['predictions']
    image_gen = (image_gen * 255).round().astype("uint8")
    image_gen_pil = Image.fromarray(image_gen.squeeze())
    draw = ImageDraw.Draw(image_gen_pil)
    text = kwargs['img_info']['prompt'].numpy().decode('utf-8')
    text_color = "black"
    chunks = [text[i:i + max_chars_per_line] for i in range(0, len(text), max_chars_per_line)]
    for i, c in enumerate(chunks):
        text_position = (0, i * 10)
        c = c.encode('utf-8').decode('latin-1')
        draw.text(text_position, c, fill=text_color)
    return np.array(image_gen_pil, np.uint8)
