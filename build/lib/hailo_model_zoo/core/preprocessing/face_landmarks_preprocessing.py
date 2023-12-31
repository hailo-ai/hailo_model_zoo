import tensorflow as tf


def face_landmark_cnn(image, image_info=None, output_height=None, output_width=None, **kwargs):
    if output_height and output_width:
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize(image, [output_height, output_width], method='bilinear')
        image = tf.squeeze(image)
    if image_info:
        image_info['img_orig'] = tf.cast(image, tf.uint8)
    return image, image_info
