import json
import os
from collections import OrderedDict
from glob import glob
from pathlib import Path

import numpy as np
import pyquaternion
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.data_classes import Box as NuScenesBox
from tqdm import tqdm

from hailo_model_zoo.core.datasets.datasets_info import get_dataset_info
from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.eval.kitti_eval import kitti_evaluation
from hailo_model_zoo.core.factory import EVAL_FACTORY
from hailo_model_zoo.utils.logger import get_logger


@EVAL_FACTORY.register(name="3d_detection")
class Detection3DEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = [
            "car_bev_AP_e",
            "car_bev_AP_m",
            "car_bev_AP_h",
            "car_3d_AP_e",
            "car_3d_AP_m",
            "car_3d_AP_h",
        ]
        self._metrics_vals = len(self._metric_names) * [0.0]
        self._channels_remove = kwargs["channels_remove"] if kwargs["channels_remove"]["enabled"] else None
        if self._channels_remove:
            self.cls_mapping, self.filtered_classes = self._create_class_mapping()
        self.reset()

    def reset(self):
        self.results_dict = {}
        self.old_results_dict_length = 0

    def _parse_net_output(self, net_output):
        return net_output["predictions"]

    def update_op(self, image_detections, gt_labels):
        image_detections = self._parse_net_output(image_detections)
        img_name = gt_labels["image_name"][0].decode("utf-8").split(".")[0]
        self.results_dict[img_name] = image_detections[0]
        return 0

    def evaluate(self):
        """This evaluation is designed for batch size = 1"""
        new_result_weight = (len(self.results_dict) - self.old_results_dict_length) / len(self.results_dict)
        old_result_weight = self.old_results_dict_length / len(self.results_dict)

        car_bev_AP_e_m_h, car_3d_AP_e_m_h = kitti_evaluation(
            "detection", "kitti_3d", self.results_dict, output_folder="./"
        )
        car_bev_AP_e_m_h = [k / 100 for k in car_bev_AP_e_m_h]
        car_3d_AP_e_m_h = [k / 100 for k in car_3d_AP_e_m_h]

        self._metrics_vals[self._metric_names.index("car_bev_AP_e")] = (
            car_bev_AP_e_m_h[0] * new_result_weight
            + self._metrics_vals[self._metric_names.index("car_bev_AP_e")] * old_result_weight
        )
        self._metrics_vals[self._metric_names.index("car_bev_AP_m")] = (
            car_bev_AP_e_m_h[1] * new_result_weight
            + self._metrics_vals[self._metric_names.index("car_bev_AP_m")] * old_result_weight
        )
        self._metrics_vals[self._metric_names.index("car_bev_AP_h")] = (
            car_bev_AP_e_m_h[2] * new_result_weight
            + self._metrics_vals[self._metric_names.index("car_bev_AP_h")] * old_result_weight
        )

        self._metrics_vals[self._metric_names.index("car_3d_AP_e")] = (
            car_3d_AP_e_m_h[0] * new_result_weight
            + self._metrics_vals[self._metric_names.index("car_3d_AP_e")] * old_result_weight
        )
        self._metrics_vals[self._metric_names.index("car_3d_AP_m")] = (
            car_3d_AP_e_m_h[1] * new_result_weight
            + self._metrics_vals[self._metric_names.index("car_3d_AP_m")] * old_result_weight
        )
        self._metrics_vals[self._metric_names.index("car_3d_AP_h")] = (
            car_3d_AP_e_m_h[2] * new_result_weight
            + self._metrics_vals[self._metric_names.index("car_3d_AP_h")] * old_result_weight
        )

        self.old_results_dict_length = len(self.results_dict)

    def _get_accuracy(self):
        return OrderedDict(
            [
                (self._metric_names[0], self._metrics_vals[0]),
                (self._metric_names[1], self._metrics_vals[1]),
                (self._metric_names[2], self._metrics_vals[2]),
                (self._metric_names[3], self._metrics_vals[3]),
                (self._metric_names[4], self._metrics_vals[4]),
                (self._metric_names[5], self._metrics_vals[5]),
            ]
        )


class Box3D:
    def __init__(self, box3d):
        self.tensor = box3d

    @property
    def dims(self):
        """Corners of each box with size (N, 8, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """A vector with yaw of each box."""
        return self.tensor[:, 6]

    @property
    def bottom_center(self):
        """A tensor with center of each box."""
        return self.tensor[:, :3]

    @property
    def gravity_center(self):
        """A tensor with center of each box."""
        bottom_center = self.bottom_center
        gravity_center = np.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    def __len__(self):
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d: Detection bbox.
            - scores_3d: Detection scores.
            - labels_3d: Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = Box3D(detection["boxes_3d"])
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(box_gravity_center[i], box_dims[i], quat, label=labels[i], score=scores[i], velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info, boxes, classes, eval_configs, eval_version="detection_cvpr_2019"):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list


DefaultAttribute = {
    "car": "vehicle.parked",
    "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bus": "vehicle.moving",
    "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "barrier": "",
    "traffic_cone": "",
}


@EVAL_FACTORY.register(name="object_detection_3d")
class PETRv2Eval(Eval):
    def __init__(self, **kwargs):
        self.version = "v1.0-trainval"
        self.eval_version = "detection_cvpr_2019"
        self.eval_detection_configs = config_factory(self.eval_version)
        self.DefaultAttribute = DefaultAttribute
        self.dataroot = str(kwargs.get("gt_json_path", None))

        self.modality = {
            "use_lidar": False,
            "use_camera": True,
            "use_radar": False,
            "use_map": False,
            "use_external": True,
        }
        dataset_name = kwargs.get("dataset_name", None)
        dataset_info = get_dataset_info(dataset_name=dataset_name)
        self.classes = dataset_info.class_names
        self.reset()

    def _parse_net_output(self, net_output):
        return net_output["predictions"]

    def update_op(self, net_output, img_info):
        net_output = self._parse_net_output(net_output)
        nusc_dets = {}
        for idx, det in enumerate(net_output):
            dets = []
            boxes = output_to_nusc_box(det)

            data_info = {
                "lidar2ego_rotation": img_info["lidar2ego_rotation"][idx],
                "lidar2ego_translation": img_info["lidar2ego_translation"][idx],
                "ego2global_rotation": img_info["ego2global_rotation"][idx],
                "ego2global_translation": img_info["ego2global_translation"][idx],
            }
            sample_token = img_info["token"][idx].decode("utf-8")
            boxes = lidar_nusc_box_to_global(
                data_info, boxes, self.classes, self.eval_detection_configs, self.eval_version
            )

            for _, box in enumerate(boxes):
                name = self.classes[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = self.DefaultAttribute[name]

                nusc_det = {
                    "sample_token": sample_token,
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr,
                }
                dets.append(nusc_det)
            nusc_dets[sample_token] = dets
        self.nusc_dets = nusc_dets

    def evaluate(self):
        nusc_submissions = {
            "meta": self.modality,
            "results": self.nusc_dets,
        }

        self.res_path = os.path.join(os.getcwd(), "results_nusc.json")
        print("Writing detection results to ", self.res_path)
        with open(self.res_path, "w") as f:
            json.dump(nusc_submissions, f)

        output_dir = os.path.join(*os.path.split(self.res_path)[:-1])
        nusc = NuScenes(
            version=self.version,
            dataroot=self.dataroot,
            verbose=False,
        )
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=self.res_path,
            eval_set="val",
            output_dir=output_dir,
            verbose=False,
        )
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = json.load(open(os.path.join(output_dir, "metrics_summary.json"), "rb"))

        self.mAP = metrics["mean_ap"]
        self.NDS = metrics["nd_score"]

    def _get_accuracy(self):
        return OrderedDict([("mAP", self.mAP), ("NDS", self.NDS)])

    def reset(self):
        self.mAP = 0
        self.NDS = 0
        self.count = 0
        self.nusc_dets = None
        self.res_path = None


@EVAL_FACTORY.register(name="object_detection_3d_backbone")
class PETRv2BackboneEval(Eval):
    def __init__(self, **kwargs):
        self.num_cams = 6
        self.num_frames = 2 * self.num_cams

        coords3d_pe_folder = kwargs.get("gt_json_path", None)
        files = sorted(glob(os.path.join(coords3d_pe_folder, "*")), key=os.path.getctime)
        # This assumes coords3d file name follows the format <name>_<img_id>.npy
        fname = "_".join(files[0].split("_")[:-1])
        self.coords3d_pe_path = "_".join([fname] + ["{}.npy"])

        # Folder to save model outputs (which are inputs to transformer)
        output_folder = Path(os.getcwd()) / "petrv2_transformer_inputs"
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        self.output_path = str(output_folder / "petrv2_transformer_input_{}.npz")
        self.output_folder = output_folder

        self.logger = get_logger()

        return

    def _parse_net_output(self, net_output):
        return net_output["predictions"]

    def update_op(self, net_output, img_info):
        net_output = self._parse_net_output(net_output)
        ch = net_output.shape[-1]
        self.logger.info("Saving model outputs...")
        batched_dataset = np.split(net_output, net_output.shape[0] // self.num_frames)
        for idx, batched_output in tqdm(enumerate(batched_dataset), total=len(batched_dataset)):
            mlvl_feats = batched_output.transpose(3, 0, 1, 2).reshape(ch, self.num_frames, -1).transpose(1, 2, 0)
            coords3d_pe_path = self.coords3d_pe_path.format(idx)
            coords3d_pe = np.load(coords3d_pe_path)

            output = {
                "input_layer1": mlvl_feats,
                "input_layer2": coords3d_pe,
            }
            output_path = self.output_path.format(idx)
            np.savez(output_path, **output)

        return

    def evaluate(self):
        self.logger.info(
            f"Evaluation for this model is degenerated. Model outputs are saved under {self.output_folder}"
        )
        return

    def _get_accuracy(self):
        return OrderedDict([("N/A", -np.inf)])

    def reset(self):
        return
