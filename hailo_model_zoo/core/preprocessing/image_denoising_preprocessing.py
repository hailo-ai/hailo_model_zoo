import tensorflow as tf

NOISE_MEAN = 0
NOISE_STD = 15


def dncnn3(image, image_info, height, width, **kwargs):
    transpose = False
    if tf.shape(image)[0] > tf.shape(image)[1]:
        image = tf.squeeze(tf.transpose(tf.expand_dims(image, axis=0), [0, 2, 1, 3]), axis=0)
        transpose = True
    image_info['img_orig'] = image
    image_info['img'] = image
    image = tf.cast(image, tf.float32)
    image += tf.random.normal((height, width, tf.shape(image)[2]), mean=NOISE_MEAN, stddev=NOISE_STD)
    image_info['img_noised'] = image
    image_info['transpose'] = transpose
    return image, image_info
