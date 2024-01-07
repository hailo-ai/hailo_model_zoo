import tensorflow as tf

VAE_CONFIG_SCALING_FACTOR = 0.18215


def vae_decoder(image, iamge_info, height, width, **kwargs):
    image = tf.cast(image, tf.float32)
    image = image / VAE_CONFIG_SCALING_FACTOR
    image = tf.squeeze(image, axis=0)

    return image, iamge_info


def unet(image, image_info, height, width, **kwargs):
    return image, image_info
