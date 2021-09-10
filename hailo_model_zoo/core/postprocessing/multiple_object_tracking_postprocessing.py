import tensorflow as tf
import numpy as np

from hailo_model_zoo.core.postprocessing.detection.centernet import CenternetPostProc
from detection_tools.utils.visualization_utils import visualize_boxes_and_labels_on_image_array


def multiple_object_tracking_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    kwargs['meta_arch'] = kwargs.get('meta_arch', {})
    if kwargs['meta_arch'] != 'fair_mot':
        raise NotImplementedError('Tracking post-processing currently supports only FairMOT based architectures')

    centernet_postprocessing = CenternetPostProc(device_pre_post_layers=device_pre_post_layers,
                                                 score_threshold=kwargs['score_threshold'])
    detection_dict = centernet_postprocessing.postprocessing([endnodes[2], endnodes[1], endnodes[3]],
                                                             device_pre_post_layers=device_pre_post_layers,
                                                             **kwargs)

    re_id_values = tf.nn.l2_normalize(endnodes[0], axis=-1)
    top_indices = tf.compat.v1.py_func(_get_top_indices, [re_id_values, detection_dict['top_k_indices']], [tf.int64])
    detection_dict['re_id_values'] = tf.gather_nd(re_id_values, top_indices)
    return dict(**detection_dict)


def _get_top_indices(re_id_values, top_k):
    top_indices_including_all_features = []
    for ind in top_k:
        top_indices_including_all_features.append(np.stack([ind + [0, 0, 0, j] for j in range(128)]))
    top_indices_including_all_features = np.stack(top_indices_including_all_features)
    return top_indices_including_all_features.reshape((re_id_values.shape[0], -1, 128, top_k.shape[-1]))


def visualize_tracking_result(logits, image, threshold=0.4, image_info=None, use_normalized_coordinates=True,
                              max_boxes_to_draw=20, dataset_name=None, **kwargs):
    boxes = logits['detection_boxes'][0]
    labels = {1: {'name': 'person', 'id': 1}}
    keypoints = None
    return visualize_boxes_and_labels_on_image_array(image[0],
                                                     boxes,
                                                     logits['detection_classes'][0],
                                                     logits['detection_scores'][0],
                                                     labels,
                                                     instance_masks=logits.get('detection_masks'),
                                                     use_normalized_coordinates=use_normalized_coordinates,
                                                     max_boxes_to_draw=max_boxes_to_draw,
                                                     line_thickness=4,
                                                     min_score_thresh=threshold,
                                                     keypoints=keypoints)
