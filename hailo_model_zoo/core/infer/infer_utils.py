import numpy as np
import os


def save_image(img, image_name):
    if isinstance(image_name, bytes):
        image_name = image_name.decode("utf-8")
    img_name = os.path.splitext(image_name)[0]
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
        log += ' {}={:.3f}'.format(result_native.name, norm_coeff * deg)
    logger.info(log)
    return log


def log_accuracy(logger, num_of_images, accuracies_output):
    log = 'Done {} images'.format(num_of_images)
    for result in accuracies_output:
        norm_coeff = 100.0 if result.is_percentage else 1.0
        log += ' {}={:.3f}'.format(result.name, norm_coeff * result.value)
    logger.info(log)
    return log


def get_logits_per_image(logits):
    if type(logits) is np.ndarray:
        # (BATCH, someshape) -> (BATCH, 1, someshape)
        return np.expand_dims(logits, axis=1)

    if type(logits) is list:
        return zip(*[get_logits_per_image(logit) for logit in logits])

    if type(logits) is dict:
        return [dict(zip(logits, t)) for t in get_logits_per_image(list(logits.values()))]

    raise ValueError("Unsupported type {} for logits".format(type(logits)))
