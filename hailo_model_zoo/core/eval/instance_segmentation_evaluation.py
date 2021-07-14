import numpy as np
from collections import OrderedDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.postprocessing.instance_segmentation_postprocessing import postprocess
from hailo_model_zoo.core.datasets.datasets_info import get_dataset_info


class InstanceSegmentationEval(Eval):
    """COCO evaluation metric class."""

    def __init__(self, **kwargs):
        """Constructs COCO evaluation class.

        The class provides the interface to metrics_fn in TPUEstimator. The
        _update_op() takes detections from each image and push them to
        self._detections. The _evaluate() loads a JSON file in COCO annotation format
        as the groundtruths and runs COCO evaluation.

        """
        dataset_name = kwargs.get('dataset_name', None)
        dataset_info = get_dataset_info(dataset_name=dataset_name)
        self._label_map = dataset_info.label_map
        self._label_inv_map = {v: k for k, v in self._label_map.items()}
        self._metric_names = ['bbox AP', 'bbox AP50', 'bbox AP75', 'bbox APs', 'bbox APm',
                              'bbox APl', 'bbox ARmax1', 'bbox ARmax10', 'bbox ARmax100',
                              'bbox ARs', 'bbox ARm', 'bbox ARl', 'mask AP', 'mask AP50',
                              'mask AP75', 'mask APs', 'mask APm', 'mask APl', 'mask ARmax1',
                              'mask ARmax10', 'mask ARmax100', 'mask ARs', 'mask ARm', 'mask ARl']
        self._metrics_vals = [0, 0]

        self._channels_remove = kwargs["channels_remove"] if kwargs["channels_remove"]["enabled"] else None
        if self._channels_remove:
            self.cls_mapping, self.catIds = self._create_class_mapping()
        else:
            self._channels_remove = None
        self._mask_data = []
        self._gt_ann_file = kwargs['gt_json_path']
        self._top_k = 200
        self.reset()

    def _create_class_mapping(self):
        cats_to_eval = list(np.where(np.array(self._channels_remove['mask'][0]) == 1)[0])
        num_classes = len(self._channels_remove['mask'][0])
        cls_mapping = {}
        idx = 0
        for cl in range(num_classes):
            if cl in cats_to_eval:
                cls_mapping[idx] = cl
                idx += 1
        cats_to_eval.remove(0)
        cls_mapping.pop(0)
        return cls_mapping, cats_to_eval

    def reset(self):
        """Reset COCO API object."""
        self._coco_gt = COCO()
        self._detections = np.empty(shape=(0, 7))
        self._dataset = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        self._images = set()
        self._annotation_id = 1
        self._category_ids = set()

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, img_info):
        net_output = self._parse_net_output(net_output)
        gt_data_height = img_info["height"]
        gt_data_width = img_info["width"]
        gt_data_image_id = img_info["image_id"]
        gt_data_num_boxes = img_info["num_boxes"]
        gt_data_bbox = img_info["bbox"]
        gt_data_area = img_info["area"]
        gt_data_category_id = img_info["category_id"]
        gt_data_is_crowd = img_info["is_crowd"]
        for batch_idx in range(len(net_output)):
            t = postprocess(net_output, gt_data_width[batch_idx], gt_data_height[batch_idx], batch_idx=batch_idx)
            if t[0] is not None:
                detection_classes, detection_scores, boxes, detection_masks = [x[:self._top_k] for x in t[:4]]
                num_detections = min(self._top_k, detection_classes.shape[0])
                # Rearrange detections bbox to be in the coco format - xmin, ymin, width, height
                bbox = np.transpose(np.vstack([boxes[:, 0], boxes[:, 1],
                                               boxes[:, 2] - boxes[:, 0],
                                               boxes[:, 3] - boxes[:, 1]]))
                # Concat image id per detection in the first column
                image_id_vec = np.ones(shape=(bbox.shape[0], 1)) * gt_data_image_id[batch_idx]
                # Finalize the detections np.array as shape=(num_detections, 7)
                # (image_id, xmin, ymin, width, height, score, class)
                for i in range(num_detections):
                    rle = maskUtils.encode(np.asfortranarray(detection_masks[i]).astype(np.uint8))
                    rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings
                    detection_category_id = detection_classes[i] + 1 \
                        if not self._channels_remove else self.cls_mapping[detection_classes[i] + 1]
                    self._mask_data.append({
                        'image_id': int(gt_data_image_id[batch_idx]),
                        'category_id': int(self._label_inv_map[detection_category_id]),
                        'segmentation': rle,
                        'bbox': [bbox[i, 0], bbox[i, 1], bbox[i, 2], bbox[i, 3]],
                        'score': float(detection_scores[i])
                    })
                    if self._channels_remove:
                        detection_classes[i] = self.cls_mapping[detection_classes[i] + 1] - 1

                new_detections = np.hstack([image_id_vec, bbox, detection_scores[:, np.newaxis],
                                            detection_classes[:, np.newaxis]])[:num_detections, :]
                self._detections = np.vstack([self._detections, new_detections])

        for num_boxes, image_id, width, height, bboxes, area, category_id, is_crowd in zip(gt_data_num_boxes,
                                                                                           gt_data_image_id,
                                                                                           gt_data_width,
                                                                                           gt_data_height,
                                                                                           gt_data_bbox,
                                                                                           gt_data_area,
                                                                                           gt_data_category_id,
                                                                                           gt_data_is_crowd):
            def _add_gt_bbox(annotation_id, image_id, category_id, bbox, is_crowd, area):
                self._dataset['annotations'].append({'id': int(annotation_id),
                                                     'image_id': int(image_id),
                                                     'category_id': int(category_id),
                                                     'bbox': bbox,
                                                     'area': area,
                                                     'iscrowd': is_crowd})

            for i in range(num_boxes):
                if category_id[i] in self._label_map.keys():
                    cat = self._label_map[category_id[i]] - 1
                    _add_gt_bbox(annotation_id=self._annotation_id, image_id=image_id,
                                 category_id=cat,
                                 bbox=bboxes[i, :], is_crowd=is_crowd[i], area=area[i])
                    self._category_ids.add(cat)
                    self._annotation_id += 1
            self._images.add(image_id)
            self._dataset['images'] = [{'id': int(img_id)} for img_id in self._images]
            self._dataset['categories'] = [{'id': int(category_id)} for category_id in self._category_ids]

    def _evaluate_mask(self):
        gt_annotations = COCO(self._gt_ann_file)
        mask_dets = gt_annotations.loadRes(self._mask_data)
        seg_eval = COCOeval(gt_annotations, mask_dets, 'segm')
        seg_eval.params.imgIds = list(self._images)
        if self._channels_remove:
            seg_eval.params.catIds = list([self._label_inv_map[i] for i in self.catIds])
        seg_eval.evaluate()
        seg_eval.accumulate()
        seg_eval.summarize()
        self._metrics_vals += list(np.array(seg_eval.stats, dtype=np.float32))

    def evaluate(self):
        """Evaluates with detections from all images in our data set with COCO API.

        Returns:
            coco_metric: float numpy array with shape [12] representing the
            coco-style evaluation metrics.
        """
        self._coco_gt.dataset = self._dataset

        self._coco_gt.createIndex()
        detections = self._detections
        coco_dt = self._coco_gt.loadRes(detections)
        coco_eval = COCOeval(self._coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.imgIds = list(self._images)
        if self._channels_remove:
            coco_eval.params.catIds = list([i - 1 for i in self.catIds])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self._metrics_vals = list(np.array(coco_eval.stats, dtype=np.float32))
        self._evaluate_mask()

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0]),
                            (self._metric_names[1], self._metrics_vals[1])])
