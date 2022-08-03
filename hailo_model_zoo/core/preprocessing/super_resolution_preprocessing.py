import tensorflow as tf

TO_BLUR = True
BLUR_SIZE = 5
BLUR_MEAN = 1.0
BLUR_STD = 0.66


def _gaussian_kernel(size=5, mean=1.0, std=0.66):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    gauss_kernel = tf.cast(gauss_kernel, tf.float32)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def _blur_image(image, size=5, mean=1.0, std=0.66):
    # Make Gaussian Kernel with desired specs.
    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    gauss_kernel = _gaussian_kernel(size, mean, std)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    gauss_kernel = tf.concat(image.shape[-1] * [gauss_kernel], 2)
    # Convolve.
    blurred = tf.nn.depthwise_conv2d(image, filter=gauss_kernel, strides=[1, 1, 1, 1], padding="SAME",
                                     data_format="NHWC")
    return blurred


def resnet(hr_image, image_info=None, height=136, width=260, **kwargs):
    hr_image = tf.cast(hr_image, tf.float32)
    hr_image = tf.expand_dims(hr_image, 0)
    hr_image.set_shape((1, None, None, 3))
    if TO_BLUR:
        with tf.device('CPU:0'):
            hr_image_blurred = _blur_image(hr_image, size=BLUR_SIZE, mean=BLUR_MEAN, std=BLUR_STD)
    else:
        hr_image_blurred = hr_image
    lr_image = tf.image.resize(hr_image_blurred, [height, width], method='bicubic')

    lr_image = tf.clip_by_value(lr_image, 0, 255)
    lr_image = tf.squeeze(lr_image)
    hr_image = tf.squeeze(hr_image)
    return lr_image, hr_image


def srgan(image, image_info, height, width, **kwargs):
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
            image_info['img_orig'] = image
            hr_img = image_info.get('hr_img')
            if hr_img is not None:
                hr_img = tf.expand_dims(hr_img, axis=0)
                hr_img = tf.image.resize_with_crop_or_pad(hr_img, 4 * height, 4 * width)
                hr_img = tf.squeeze(hr_img, axis=0)
                image_info['hr_img'] = tf.cast(hr_img, tf.uint8)
    # return {'input_layer1': image, 'input_layer1_new': image}, image_info # removed due to hn_editor in srgan
    return image, image_info
