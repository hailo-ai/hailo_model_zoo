import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY

TO_BLUR = True
BLUR_SIZE = 5
BLUR_MEAN = 1.0
BLUR_STD = 0.66

RGB2YUV_mat = [
    [0.25678824, -0.14822353, 0.43921569],
    [0.50412941, -0.29099216, -0.36778824],
    [0.09790588, 0.43921569, -0.07142745],
]
RGB2YUV_offset = [16, 128, 128]


def _gaussian_kernel(size=5, mean=1.0, std=0.66):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum("i,j->ij", vals, vals)
    gauss_kernel = tf.cast(gauss_kernel, tf.float32)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def _blur_image(image, size=5, mean=1.0, std=0.66):
    # Make Gaussian Kernel with desired specs.
    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    gauss_kernel = _gaussian_kernel(size, mean, std)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    gauss_kernel = tf.concat(image.shape[-1] * [gauss_kernel], 2)
    # Convolve.
    blurred = tf.nn.depthwise_conv2d(
        image, filter=gauss_kernel, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC"
    )
    return blurred


@PREPROCESS_FACTORY.register(name="sr_resnet")
def resnet(hr_image, image_info=None, height=136, width=260, **kwargs):
    hr_image = tf.cast(hr_image, tf.float32)
    hr_image = tf.expand_dims(hr_image, 0)
    hr_image.set_shape((1, None, None, 3))
    if TO_BLUR:
        with tf.device("CPU:0"):
            hr_image_blurred = _blur_image(hr_image, size=BLUR_SIZE, mean=BLUR_MEAN, std=BLUR_STD)
    else:
        hr_image_blurred = hr_image
    lr_image = tf.image.resize(hr_image_blurred, [height, width], method="bicubic")

    lr_image = tf.clip_by_value(lr_image, 0, 255)
    lr_image = tf.squeeze(lr_image)
    hr_image = tf.squeeze(hr_image)
    return lr_image, hr_image


@PREPROCESS_FACTORY.register
def srgan(image, image_info, height, width, output_shapes=None, **kwargs):
    """
    preprocessing function for srgan and div2k
        1. resize the images by taking a central crop with padding
        2. normalize the input to [-1,1]
    Args:
        image: a tensor with the low-resolution image (network input)
        height: the required height of the image that will be the super-resolution net input
        width: the required width of the image that will be the super-resolution net input
    Returns:
       lr_image_normalized - the low-resolution input to the net, normalized to range [-1.0,1.0]
       hr_image - the high-resolution image for performance analysis
    """
    if width and height:
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_with_crop_or_pad(image, height, width)
        image = tf.cast(tf.squeeze(image, axis=0), tf.float32)
        if image_info:
            image_info["img_orig"] = image
            hr_img = image_info.get("hr_img")
            if hr_img is not None:
                hr_img = tf.expand_dims(hr_img, axis=0)
                assert len(output_shapes) == 1, f"expects 1 output shape but got {len(output_shapes)}"
                hr_img = tf.image.resize_with_crop_or_pad(hr_img, output_shapes[0][1], output_shapes[0][2])
                hr_img = tf.squeeze(hr_img, axis=0)
                image_info["hr_img"] = tf.cast(hr_img, tf.uint8)
    # return {'input_layer1': image, 'input_layer1_new': image}, image_info # removed due to hn_editor in srgan
    return image, image_info


@PREPROCESS_FACTORY.register
def espcn(image, image_info, height, width, output_shapes, **kwargs):
    if width and height:
        image_orig = image
        # Verify input is landscape
        if tf.shape(image)[0] > tf.shape(image)[1]:
            image = tf.squeeze(tf.transpose(tf.expand_dims(image, axis=0), [0, 2, 1, 3]), axis=0)
        image_orig = tf.image.resize_with_pad(image, height, width)
        # RGB ==> YUV
        image = tf.cast(image, tf.float32)
        image = tf.matmul(image, RGB2YUV_mat)
        image = tf.add(image, RGB2YUV_offset)

        # Taking luminance channel only
        image = tf.expand_dims(tf.expand_dims(image[..., 0], axis=-1), axis=0)
        image = tf.image.resize_with_pad(image, height, width)
        image = tf.cast(tf.squeeze(image, axis=0), tf.float32)

        if image_info:
            image_info["img_orig"] = image_orig
            hr_img = image_info.get("hr_img")
            if hr_img is not None:
                # Verify input is landscape
                if tf.shape(hr_img)[0] > tf.shape(hr_img)[1]:
                    hr_img = tf.squeeze(tf.transpose(tf.expand_dims(hr_img, axis=0), [0, 2, 1, 3]), axis=0)
                # RGB ==> YUV
                hr_img = tf.cast(hr_img, tf.float32)
                hr_img = tf.matmul(hr_img, RGB2YUV_mat)
                hr_img = tf.add(hr_img, RGB2YUV_offset)
                hr_img /= 255.0  # Normalization
                # Taking luminance channel only
                hr_img = tf.expand_dims(tf.expand_dims(hr_img[..., 0], axis=-1), axis=0)
                assert len(output_shapes) == 1, f"expects 1 output shape but got {len(output_shapes)}"
                hr_img = tf.image.resize_with_pad(hr_img, output_shapes[0][1], output_shapes[0][2])
                hr_img = tf.squeeze(hr_img, axis=0)
                image_info["hr_img"] = tf.cast(hr_img, tf.float32)
    return image, image_info
