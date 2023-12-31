import tensorflow as tf
import numpy as np
import cv2


def image_denoising_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    endnodes = tf.cast(tf.math.round(tf.clip_by_value(endnodes, 0, 1) * 255.0), tf.uint8)
    return {'predictions': endnodes}


def visualize_image_denoising_result(predicted_img, orig_img, **kwargs):
    noised_img = kwargs['img_info']['img_noised'].numpy()
    if kwargs['img_info']['transpose'].numpy():
        predicted_img['predictions'] = np.transpose(predicted_img['predictions'], axes=[0, 2, 1, 3]).copy()
        orig_img = np.transpose(orig_img, axes=[0, 2, 1, 3]).copy()
        noised_img = np.transpose(noised_img, axes=[1, 0, 2]).copy()
    noised_img = noised_img.squeeze() * 255
    predicted_img = predicted_img['predictions'].squeeze()
    orig_img = orig_img.squeeze()
    predicted_img = predicted_img.astype(np.uint8)
    predicted_img = cv2.putText(predicted_img, 'Denoised Image', (5, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    orig_img = cv2.putText(orig_img, 'Original Image', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                           (255, 255, 255), 2)
    noised_img = cv2.putText(noised_img, 'Noised Image', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                             (255, 255, 255), 2)
    combined_image = np.concatenate([orig_img, noised_img, predicted_img], axis=1)
    combined_image = combined_image.astype(np.uint8)
    return combined_image
