from tensorflow import ensure_shape

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY


@PREPROCESS_FACTORY.register(name="amat_preprocessing")
def amat_preprocessing(image, image_info, height, width, **kwargs):
    # Images from AMAT are 3 channels, each channel is a gray-scale image.
    #  The model uses each channel as a separate input.
    image = {
        f"{kwargs['network_name']}/input_layer1": ensure_shape(image[..., :1], (height, width, 1)),
        f"{kwargs['network_name']}/input_layer2": ensure_shape(image[..., 1:2], (height, width, 1)),
        f"{kwargs['network_name']}/input_layer3": ensure_shape(image[..., 2:], (height, width, 1)),
    }
    return image, image_info
