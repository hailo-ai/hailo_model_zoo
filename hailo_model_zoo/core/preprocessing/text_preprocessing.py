import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY


@PREPROCESS_FACTORY.register(name="text_encoder")
def text_encoder_preprocessing(image, image_info, height, width, **kwargs):
    sequence_length = width
    current_length = tf.shape(image)[1]
    padding_length = sequence_length - current_length
    padding_value = image[:, -1:, :]  # use end_of_text token for padding
    padding = tf.tile(padding_value, (1, padding_length, 1))
    image = tf.concat((image, padding), axis=1)
    image = tf.ensure_shape(image, (1, sequence_length, kwargs["channels"]))
    # We find the first end of text token, which is the maximum
    image_info["last_token"] = tf.argmax(image_info["input_ids"])
    return image, image_info
