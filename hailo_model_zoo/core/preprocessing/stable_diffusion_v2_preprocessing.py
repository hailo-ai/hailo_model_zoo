import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY

VAE_CONFIG_SCALING_FACTOR = 0.18215


@PREPROCESS_FACTORY.register(name="stable_diffusion_v2_decoder")
def vae_decoder(image, iamge_info, height, width, **kwargs):
    image = tf.cast(image, tf.float32)
    image = image / VAE_CONFIG_SCALING_FACTOR
    image = tf.squeeze(image, axis=0)

    return image, iamge_info


@PREPROCESS_FACTORY.register(name="stable_diffusion_v2_unet")
def unet(image, image_info, height, width, **kwargs):
    return image, image_info
