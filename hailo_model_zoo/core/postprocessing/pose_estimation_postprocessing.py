import math
from operator import itemgetter

import cv2
import numpy as np

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY
from hailo_model_zoo.core.postprocessing.centerpose_postprocessing import centerpose_postprocessing
from hailo_model_zoo.core.postprocessing.cython_utils.cython_nms import nms as cnms
from hailo_model_zoo.core.postprocessing.instance_segmentation_postprocessing import xywh2xyxy

BODY_PARTS_KPT_IDS = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
    [2, 16],
    [5, 17],
]
BODY_PARTS_PAF_IDS = (
    [12, 13],
    [20, 21],
    [14, 15],
    [16, 17],
    [22, 23],
    [24, 25],
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [28, 29],
    [30, 31],
    [34, 35],
    [32, 33],
    [36, 37],
    [18, 19],
    [26, 27],
)

STRIDE = 8

JOINT_PAIRS = [
    [0, 1],
    [1, 3],
    [0, 2],
    [2, 4],
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [12, 14],
    [13, 15],
    [14, 16],
]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _softmax(x):
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)


def scale_kpts(kpts, shape, orig_shape):
    kpts = np.reshape(kpts, (-1, shape[0], shape[1], 17, 3))
    gain = min(shape[0] / orig_shape[0], shape[1] / orig_shape[1])
    pad = [(shape[1] - orig_shape[1] * gain) / 2, (shape[0] - orig_shape[0] * gain) / 2, 0]
    kpts = (kpts - pad) / gain
    return kpts


@POSTPROCESS_FACTORY.register(name="pose_estimation")
def pose_estimation_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    if kwargs.get("meta_arch") == "centerpose":
        return centerpose_postprocessing(endnodes, device_pre_post_layers=None, **kwargs)
    if kwargs.get("meta_arch") == "nanodet_v8":
        return yolov8_pose_estimation_postprocess(endnodes, device_pre_post_layers=None, **kwargs)
    coco_result_list = []
    for i in range(len(endnodes)):
        image_info = kwargs["gt_images"]
        heatmaps, pafs, image_id, pad, orig_shape = (
            endnodes[0][i],
            endnodes[1][i],
            int(image_info["image_id"][i]),
            image_info["pad"][i],
            image_info["orig_shape"][i][:2],
        )
        total_keypoints_num = 0
        all_keypoints_by_type = []

        height, width = orig_shape
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=STRIDE, fy=STRIDE, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0] : heatmaps.shape[0] - pad[2], pad[1] : heatmaps.shape[1] - pad[3] :, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)

        pafs = cv2.resize(pafs, (0, 0), fx=STRIDE, fy=STRIDE, interpolation=cv2.INTER_CUBIC)
        pafs = pafs[pad[0] : pafs.shape[0] - pad[2], pad[1] : pafs.shape[1] - pad[3], :]
        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)

        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
            )

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

        coco_result_list += [
            {
                "image_id": image_id,
                "category_id": 1,  # person
                "keypoints": coco_keypoints[idx],
                "score": scores[idx],
            }
            for idx in range(len(coco_keypoints))
        ]

        return {"predictions": coco_result_list}


@VISUALIZATION_FACTORY.register(name="pose_estimation")
def visualize_pose_estimation_result(
    results, img, dataset_name, *, detection_threshold=0.5, joint_threshold=0.5, **kwargs
):
    assert dataset_name == "cocopose"
    if "predictions" in results:
        results = results["predictions"]
        bboxes, scores, keypoints, joint_scores = results
    else:
        bboxes, scores, keypoints, joint_scores = (
            results["bboxes"],
            results["scores"],
            results["keypoints"],
            results["joint_scores"],
        )

    batch_size = bboxes.shape[0]
    assert batch_size == 1

    box, score, keypoint, keypoint_score = bboxes[0], scores[0], keypoints[0], joint_scores[0]
    if "centerpose" in kwargs["meta_arch"]:
        image = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
    else:
        image = img[0]
    for detection_box, detection_score, detection_keypoints, detection_keypoints_score in zip(
        box, score, keypoint, keypoint_score
    ):
        if detection_score < detection_threshold:
            continue
        xmin, ymin, xmax, ymax = [int(x) for x in detection_box]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        cv2.putText(image, str(detection_score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

        joint_visible = detection_keypoints_score > joint_threshold

        detection_keypoints = detection_keypoints.reshape(17, 2)
        for joint, joint_score in zip(detection_keypoints, detection_keypoints_score):
            if joint_score < joint_threshold:
                continue
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, (255, 0, 255), -1)

        for joint0, joint1 in JOINT_PAIRS:
            if joint_visible[joint0] and joint_visible[joint1]:
                pt1 = (int(detection_keypoints[joint0][0]), int(detection_keypoints[joint0][1]))
                pt2 = (int(detection_keypoints[joint1][0]), int(detection_keypoints[joint1][1]))
                cv2.line(image, pt1, pt2, (255, 0, 255), 3)

    return image


def linspace2d(start, stop, n=10):
    points = 1 / (n - 1) * (stop - start)
    return points[:, None] * np.arange(n) + start[:, None]


def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    heatmap[heatmap < 0.1] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode="constant")
    heatmap_center = heatmap_with_borders[1 : heatmap_with_borders.shape[0] - 1, 1 : heatmap_with_borders.shape[1] - 1]
    heatmap_left = heatmap_with_borders[1 : heatmap_with_borders.shape[0] - 1, 2 : heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1 : heatmap_with_borders.shape[0] - 1, 0 : heatmap_with_borders.shape[1] - 2]
    heatmap_up = heatmap_with_borders[2 : heatmap_with_borders.shape[0], 1 : heatmap_with_borders.shape[1] - 1]
    heatmap_down = heatmap_with_borders[0 : heatmap_with_borders.shape[0] - 2, 1 : heatmap_with_borders.shape[1] - 1]

    heatmap_peaks = (
        (heatmap_center > heatmap_left)
        & (heatmap_center > heatmap_right)
        & (heatmap_center > heatmap_up)
        & (heatmap_center > heatmap_down)
    )
    heatmap_peaks = heatmap_peaks[1 : heatmap_center.shape[0] - 1, 1 : heatmap_center.shape[1] - 1]
    keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # (w, h)
    keypoints = sorted(keypoints, key=itemgetter(0))

    suppressed = np.zeros(len(keypoints), np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i + 1, len(keypoints)):
            if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 + (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                suppressed[j] = 1
        keypoint_with_score_and_id = (
            keypoints[i][0],
            keypoints[i][1],
            heatmap[keypoints[i][1], keypoints[i][0]],
            total_keypoint_num + keypoint_num,
        )
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num


def group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05, demo=False):  # noqa: C901
    pose_entries = []
    all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
        num_kpts_a = len(kpts_a)
        num_kpts_b = len(kpts_b)
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

        if num_kpts_a == 0 and num_kpts_b == 0:  # no keypoints for such body part
            continue
        elif num_kpts_a == 0:  # body part has just 'b' keypoints
            for i in range(num_kpts_b):
                num = 0
                for j in range(len(pose_entries)):  # check if already in some pose, was added by another body part
                    if pose_entries[j][kpt_b_id] == kpts_b[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_b_id] = kpts_b[i][3]  # keypoint idx
                    pose_entry[-1] = 1  # num keypoints in pose
                    pose_entry[-2] = kpts_b[i][2]  # pose score
                    pose_entries.append(pose_entry)
            continue
        elif num_kpts_b == 0:  # body part has just 'a' keypoints
            for i in range(num_kpts_a):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == kpts_a[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = kpts_a[i][3]
                    pose_entry[-1] = 1
                    pose_entry[-2] = kpts_a[i][2]
                    pose_entries.append(pose_entry)
            continue

        connections = []
        for i in range(num_kpts_a):
            kpt_a = np.array(kpts_a[i][0:2])
            for j in range(num_kpts_b):
                kpt_b = np.array(kpts_b[j][0:2])
                mid_point = [(), ()]
                mid_point[0] = (int(round((kpt_a[0] + kpt_b[0]) * 0.5)), int(round((kpt_a[1] + kpt_b[1]) * 0.5)))
                mid_point[1] = mid_point[0]

                vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]
                vec_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
                if vec_norm == 0:
                    continue
                vec[0] /= vec_norm
                vec[1] /= vec_norm
                cur_point_score = (
                    vec[0] * part_pafs[mid_point[0][1], mid_point[0][0], 0]
                    + vec[1] * part_pafs[mid_point[1][1], mid_point[1][0], 1]
                )

                height_n = pafs.shape[0] // 2
                success_ratio = 0
                point_num = 10  # number of points to integration over paf
                if cur_point_score > -100:
                    passed_point_score = 0
                    passed_point_num = 0
                    x, y = linspace2d(kpt_a, kpt_b)
                    for point_idx in range(point_num):
                        if not demo:
                            px = int(round(x[point_idx]))
                            py = int(round(y[point_idx]))
                        else:
                            px = int(x[point_idx])
                            py = int(y[point_idx])
                        paf = part_pafs[py, px, 0:2]
                        cur_point_score = vec[0] * paf[0] + vec[1] * paf[1]
                        if cur_point_score > min_paf_score:
                            passed_point_score += cur_point_score
                            passed_point_num += 1
                    success_ratio = passed_point_num / point_num
                    ratio = 0
                    if passed_point_num > 0:
                        ratio = passed_point_score / passed_point_num
                    ratio += min(height_n / vec_norm - 1, 0)
                if ratio > 0 and success_ratio > 0.8:
                    score_all = ratio + kpts_a[i][2] + kpts_b[j][2]
                    connections.append([i, j, ratio, score_all])
        if len(connections) > 0:
            connections = sorted(connections, key=itemgetter(2), reverse=True)

        num_connections = min(num_kpts_a, num_kpts_b)
        has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)
        has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)
        filtered_connections = []
        for row in range(len(connections)):
            if len(filtered_connections) == num_connections:
                break
            i, j, cur_point_score = connections[row][0:3]
            if not has_kpt_a[i] and not has_kpt_b[j]:
                filtered_connections.append([kpts_a[i][3], kpts_b[j][3], cur_point_score])
                has_kpt_a[i] = 1
                has_kpt_b[j] = 1
        connections = filtered_connections
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
            for i in range(len(connections)):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                    elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                        pose_entries[j][kpt_a_id] = connections[i][0]
            continue
        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0]:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries[i])
    pose_entries = np.asarray(filtered_entries)
    return pose_entries, all_keypoints


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.45, max_det=100, n_kpts=17):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Args:
        prediction: numpy.ndarray with shape (batch_size, num_proposals, 56)
        conf_thres: confidence threshold for NMS
        iou_thres: IoU threshold for NMS
        max_det: Maximal number of detections to keep after NMS
        nm: Number of masks
        multi_label: Consider only best class per proposal or all conf_thresh passing proposals
    Returns:
         A list of per image detections, where each is a dictionary with the following structure:
         {
            'detection_boxes':   numpy.ndarray with shape (num_detections, 4),
            'keypoints':              numpy.ndarray with shape (num_detections, 17, 3),
            'detection_scores':  numpy.ndarray with shape (num_detections, 1),
            'num_detections':  int
         }
    """

    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0"

    nc = prediction.shape[2] - n_kpts * 3 - 4  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # max_wh = 7680  # (pixels) maximum box width and height
    ki = 4 + nc  # keypoints start index
    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            output.append(
                {
                    "bboxes": np.zeros((0, 4)),
                    "keypoints": np.zeros((0, n_kpts, 3)),
                    "scores": np.zeros((0)),
                    "num_detections": 0,
                }
            )
            continue

        # (center_x, center_y, width, height) to (x1, y1, x2, y2)
        boxes = xywh2xyxy(x[:, :4])
        kpts = x[:, ki:]

        conf = np.expand_dims(x[:, 4:ki].max(1), 1)
        j = np.expand_dims(x[:, 4:ki].argmax(1), 1).astype(np.float32)

        keep = np.squeeze(conf, 1) > conf_thres
        x = np.concatenate((boxes, conf, j, kpts), 1)[keep]

        # sort by confidence
        x = x[x[:, 4].argsort()[::-1]]

        boxes = x[:, :4]
        conf = x[:, 4:5]
        preds = np.hstack([boxes.astype(np.float32), conf.astype(np.float32)])

        keep = cnms(preds, iou_thres)
        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        out = x[keep]
        scores = out[:, 4]
        boxes = out[:, :4]
        kpts = out[:, 6:]
        kpts = np.reshape(kpts, (-1, n_kpts, 3))

        out = {"bboxes": boxes, "keypoints": kpts, "scores": scores, "num_detections": int(scores.shape[0])}

        output.append(out)
    return output


def _yolov8_decoding(
    raw_boxes,
    raw_kpts,
    strides,
    image_dims,
    reg_max,
):
    boxes = None
    decoded_kpts = None
    for box_distribute, kpts, stride, _j in zip(raw_boxes, raw_kpts, strides, np.arange(3)):
        # create grid
        shape = [int(x / stride) for x in image_dims]
        grid_x = np.arange(shape[1]) + 0.5
        grid_y = np.arange(shape[0]) + 0.5
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        ct_row = grid_y.flatten() * stride
        ct_col = grid_x.flatten() * stride
        center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

        # box distribution to distance
        reg_range = np.arange(reg_max + 1)
        box_distribute = np.reshape(
            box_distribute, (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1)
        )
        box_distance = _softmax(box_distribute)
        box_distance = box_distance * np.reshape(reg_range, (1, 1, 1, -1))
        box_distance = np.sum(box_distance, axis=-1)
        box_distance = box_distance * stride

        # decode box
        box_distance = np.concatenate([box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1)
        decode_box = np.expand_dims(center, axis=0) + box_distance

        xmin = decode_box[:, :, 0]
        ymin = decode_box[:, :, 1]
        xmax = decode_box[:, :, 2]
        ymax = decode_box[:, :, 3]
        decode_box = np.transpose([xmin, ymin, xmax, ymax], [1, 2, 0])

        xywh_box = np.transpose([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])
        boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)

        # kpts decoding
        kpts[..., :2] *= 2
        kpts[..., :2] = stride * (kpts[..., :2] - 0.5) + np.expand_dims(center[..., :2], axis=1)
        decoded_kpts = kpts if decoded_kpts is None else np.concatenate([decoded_kpts, kpts], axis=1)

    return boxes, decoded_kpts


def yolov8_pose_estimation_postprocess(endnodes, device_pre_post_layers=None, **kwargs):
    """
    endnodes is a list of 10 tensors:
        endnodes[0]:  bbox output with shapes (BS, 20, 20, 64)
        endnodes[1]:  scores output with shapes (BS, 20, 20, 80)
        endnodes[2]:  keypoints output with shapes (BS, 20, 20, 51)
        endnodes[3]:  bbox output with shapes (BS, 40, 40, 64)
        endnodes[4]:  scores output with shapes (BS, 40, 40, 80)
        endnodes[5]:  keypoints output with shapes (BS, 40, 40, 51)
        endnodes[6]:  bbox output with shapes (BS, 80, 80, 64)
        endnodes[7]:  scores output with shapes (BS, 80, 80, 80)
        endnodes[8]:  keypoints output with shapes (BS, 80, 80, 51)
    Returns:
        A list of per image detections, where each is a dictionary with the following structure:
        {
            'detection_boxes':   numpy.ndarray with shape (num_detections, 4),
            'keypoints':              numpy.ndarray with shape (num_detections, 3),
            'detection_classes': numpy.ndarray with shape (num_detections, 80),
            'detection_scores':  numpy.ndarray with shape (num_detections, 80)
        }
    """
    batch_size = endnodes[0].shape[0]
    num_classes = kwargs["classes"]  # always 1
    max_detections = kwargs["nms_max_output_per_class"]
    strides = kwargs["anchors"]["strides"][::-1]
    image_dims = tuple(kwargs["img_dims"])
    reg_max = kwargs["anchors"]["regression_length"]
    raw_boxes = endnodes[:7:3]
    scores = [np.reshape(s, (-1, s.shape[1] * s.shape[2], num_classes)) for s in endnodes[1:8:3]]
    scores = np.concatenate(scores, axis=1)
    kpts = [np.reshape(c, (-1, c.shape[1] * c.shape[2], 17, 3)) for c in endnodes[2:9:3]]
    decoded_boxes, decoded_kpts = _yolov8_decoding(raw_boxes, kpts, strides, image_dims, reg_max)
    score_thres = kwargs["score_threshold"]
    iou_thres = kwargs["nms_iou_thresh"]

    # re-arrange predictions for yolov5_nms
    decoded_kpts = np.reshape(decoded_kpts, (batch_size, -1, 51))
    predictions = np.concatenate([decoded_boxes, scores, decoded_kpts], axis=2)
    nms_res = non_max_suppression(predictions, conf_thres=score_thres, iou_thres=iou_thres, max_det=max_detections)
    output = {}
    output["bboxes"] = np.zeros((batch_size, max_detections, 4))
    output["keypoints"] = np.zeros((batch_size, max_detections, 17, 2))
    output["joint_scores"] = np.zeros((batch_size, max_detections, 17, 1))
    output["scores"] = np.zeros((batch_size, max_detections, 1))
    for b in range(batch_size):
        output["bboxes"][b, : nms_res[b]["num_detections"]] = nms_res[b]["bboxes"]
        output["keypoints"][b, : nms_res[b]["num_detections"]] = nms_res[b]["keypoints"][..., :2]
        output["joint_scores"][b, : nms_res[b]["num_detections"], ..., 0] = _sigmoid(nms_res[b]["keypoints"][..., 2])
        output["scores"][b, : nms_res[b]["num_detections"], ..., 0] = nms_res[b]["scores"]
    return output
