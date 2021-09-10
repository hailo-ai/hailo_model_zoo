import numpy as np
import tensorflow as tf
import os


def create_calib_set(calib_feed_callback, eval_num_examples, calib_filename):
    calibation_set = []
    with tf.Graph().as_default():
        iterator = calib_feed_callback()
        [preprocessed_data, _] = iterator.get_next()
        with tf.compat.v1.Session() as sess:
            sess.run([iterator.initializer])
            num_of_images = 0
            try:
                while num_of_images < eval_num_examples:
                    calib_data = sess.run(preprocessed_data)
                    calibation_set.append(calib_data)
                    num_of_images += len(calib_data)
            except tf.errors.OutOfRangeError:
                pass
    calibation_set = np.concatenate(calibation_set, axis=0)
    np.savez(calib_filename, calibation_set)
    return num_of_images


def save_image(img, image_name):
    img_name = os.path.splitext(image_name.decode("utf-8"))[0]
    img_name = img_name.replace('/', '_')
    img.save('./{}_out.png'.format(img_name))


def _get_logits(logits, idx, img_info):
    if type(logits) is list:
        ret = {}
        ret['logits'] = np.squeeze(logits[0])
        ret['image_info'] = {}
        ret['image_info']['rpn_proposals'] = np.squeeze(logits[1])
        ret['image_info']['num_rpn_proposals'] = logits[1].shape[0]
        img_info.pop('img_orig')
        for k in img_info:
            ret['image_info'].setdefault(k, {})
            ret['image_info'][k] = img_info[k][idx]
    elif type(logits) is np.ndarray:
        ret = logits[idx]
    elif type(logits) is dict:
        ret = dict()
        for key in logits.keys():
            ret[key] = logits[key][idx]
    else:
        raise Exception("Logits structure is not recognize")
    return ret


def write_results(logits, img_info, results_directory):
    os.makedirs(results_directory, exist_ok=True)
    for idx, img_name in enumerate(img_info['image_name']):
        data = dict()
        base_image_name = os.path.basename(img_name.decode()).replace(".jpg", "")
        data[base_image_name] = _get_logits(logits, idx, img_info)
        filename = '{}.npz'.format(base_image_name)
        np.savez(os.path.join(results_directory, filename), **data)


def log_degradation(logger, accuracies_output, accuracies_output_native):
    log = 'Overall Degradation:'
    for result_native, result_quantized in zip(accuracies_output_native, accuracies_output):
        norm_coeff = 100.0 if result_native.is_percentage else 1.0
        diff_coeff = 1 if result_native.is_bigger_better else -1.0
        diff = (result_native.value - result_quantized.value)
        deg = diff * diff_coeff
        log += ' {}={:.2f}'.format(result_native.name, norm_coeff * deg)
    logger.info(log)
    return log


def log_accuracy(logger, num_of_images, accuracies_output):
    log = 'Done {} images'.format(num_of_images)
    for result in accuracies_output:
        norm_coeff = 100.0 if result.is_percentage else 1.0
        log += ' {}={:.2f}'.format(result.name, norm_coeff * result.value)
    logger.info(log)
    return log
