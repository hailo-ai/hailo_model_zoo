import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY


@PREPROCESS_FACTORY.register
def zero_dce(image, image_info, height, width, output_shapes, **kwargs):
    """
    preprocessing function for zero_dce
        1. resize the images by taking a central crop with padding
    Args:
        image: a tensor with the low-light image (network input)
        height: the required height of the image that will be the low-light enhancement net input
        width: the required width of the image that will be the low-light enhancement net input
    Returns:
       ll_image resized - the low-light input to the network
       lle_image resized - the low-light imhanced image for performance analysis
    """
    if width and height:
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_with_crop_or_pad(image, height, width)
        image = tf.cast(tf.squeeze(image, axis=0), tf.float32)

        if image_info:
            image_info["img_orig"] = image
            enhanced_img = image_info.get("ll_enhanced_img")
            if enhanced_img is not None:
                enhanced_img = tf.divide(tf.cast(enhanced_img, tf.float32), 255.0)
                enhanced_img = tf.expand_dims(enhanced_img, axis=0)
                assert len(output_shapes) == 1, f"expects 1 output shape but got {len(output_shapes)}"
                enhanced_img = tf.image.resize_with_crop_or_pad(enhanced_img, output_shapes[0][1], output_shapes[0][2])
                enhanced_img = tf.squeeze(enhanced_img, axis=0)
                image_info["enhanced_img_processed"] = tf.cast(enhanced_img, tf.float32)
    return image, image_info
