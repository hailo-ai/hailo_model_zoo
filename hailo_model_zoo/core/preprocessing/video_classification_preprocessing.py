import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY

RESIZE_SIZE = (128, 171)


@PREPROCESS_FACTORY.register(name="r3d_18")
def r3d_18_preprocessing(image, image_info=None, height=112, width=112, **kwargs):
    image = tf.cast(image, tf.float32)

    # Reshape image to combine frames and perform resizing/padding once
    image = tf.transpose(image, perm=[3, 0, 1, 2])  # [num_frames, h, w, 3]
    crop_size = (height, width)
    image = tf.image.resize(image, RESIZE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False)
    image = tf.round(image)

    # Crop the image - no padding
    start_height = (RESIZE_SIZE[0] - crop_size[0]) // 2
    start_width = (RESIZE_SIZE[1] - crop_size[1]) // 2
    image = tf.image.crop_to_bounding_box(image, start_height, start_width, crop_size[0], crop_size[1])

    if image_info:
        # save first frame for visualization
        image_info["img_orig"] = tf.cast(image[0], tf.uint8)

    # Normalize each frame's channels
    mean = tf.constant([110.2008, 100.63983, 95.99475], dtype=tf.float32)
    std = tf.constant([58.14765, 56.46975, 55.332195], dtype=tf.float32)
    image = (image - mean) / std  # [num_frames, h, w, 3]

    # Combine frames by merging the channel dimension
    image = tf.transpose(image, perm=[1, 2, 0, 3])  # [h, w, num_frames, channels]
    image = tf.reshape(image, [height, width, -1])  # [h, w, 3*num_frames]

    return image, image_info
