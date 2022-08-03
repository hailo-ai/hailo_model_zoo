from collections import OrderedDict

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.eval.kitti_eval import kitti_evaluation


class Detection3DEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ['car_bev_AP_e', 'car_bev_AP_m', 'car_bev_AP_h',
                              'car_3d_AP_e', 'car_3d_AP_m', 'car_3d_AP_h']
        self._metrics_vals = len(self._metric_names) * [0.]
        self._channels_remove = kwargs["channels_remove"] if kwargs["channels_remove"]["enabled"] else None
        if self._channels_remove:
            self.cls_mapping, self.filtered_classes = self._create_class_mapping()
        self.reset()

    def reset(self):
        self.results_dict = {}
        self.old_results_dict_length = 0

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, image_detections, gt_labels):
        image_detections = self._parse_net_output(image_detections)
        img_name = gt_labels['image_name'][0].decode('utf-8').split('.')[0]
        self.results_dict[img_name] = image_detections[0]
        return 0

    def evaluate(self):
        """This evaluation is designed for batch size = 1"""
        new_result_weight = (len(self.results_dict) - self.old_results_dict_length) / len(self.results_dict)
        old_result_weight = self.old_results_dict_length / len(self.results_dict)

        car_bev_AP_e_m_h, car_3d_AP_e_m_h = kitti_evaluation('detection', 'kitti_3d', self.results_dict,
                                                             output_folder='./')
        car_bev_AP_e_m_h = [k / 100 for k in car_bev_AP_e_m_h]
        car_3d_AP_e_m_h = [k / 100 for k in car_3d_AP_e_m_h]

        self._metrics_vals[self._metric_names.index('car_bev_AP_e')] = car_bev_AP_e_m_h[0] * new_result_weight +\
            self._metrics_vals[self._metric_names.index('car_bev_AP_e')] * old_result_weight
        self._metrics_vals[self._metric_names.index('car_bev_AP_m')] = car_bev_AP_e_m_h[1] * new_result_weight +\
            self._metrics_vals[self._metric_names.index('car_bev_AP_m')] * old_result_weight
        self._metrics_vals[self._metric_names.index('car_bev_AP_h')] = car_bev_AP_e_m_h[2] * new_result_weight +\
            self._metrics_vals[self._metric_names.index('car_bev_AP_h')] * old_result_weight

        self._metrics_vals[self._metric_names.index('car_3d_AP_e')] = car_3d_AP_e_m_h[0] * new_result_weight +\
            self._metrics_vals[self._metric_names.index('car_3d_AP_e')] * old_result_weight
        self._metrics_vals[self._metric_names.index('car_3d_AP_m')] = car_3d_AP_e_m_h[1] * new_result_weight +\
            self._metrics_vals[self._metric_names.index('car_3d_AP_m')] * old_result_weight
        self._metrics_vals[self._metric_names.index('car_3d_AP_h')] = car_3d_AP_e_m_h[2] * new_result_weight +\
            self._metrics_vals[self._metric_names.index('car_3d_AP_h')] * old_result_weight

        self.old_results_dict_length = len(self.results_dict)

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0]),
                            (self._metric_names[1], self._metrics_vals[1]),
                            (self._metric_names[2], self._metrics_vals[2]),
                            (self._metric_names[3], self._metrics_vals[3]),
                            (self._metric_names[4], self._metrics_vals[4]),
                            (self._metric_names[5], self._metrics_vals[5])])
