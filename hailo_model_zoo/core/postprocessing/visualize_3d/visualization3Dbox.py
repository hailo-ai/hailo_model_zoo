import csv
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from PIL import Image

from hailo_model_zoo.core.postprocessing.visualize_3d.utils.correspondence_constraint import detectionInfo

ID_TYPE_CONVERSION = {0: "Car", 1: "Cyclist", 2: "Pedestrian"}


def compute_birdviewbox(line, shape, scale):
    npline = [np.float64(line[i]) for i in range(1, len(line))]
    w = npline[8] * scale
    ll = npline[9] * scale
    x = npline[10] * scale
    z = npline[12] * scale
    rot_y = npline[13]

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)], [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, ll, ll, 0]  # -ll/2
    z_corners = [w, w, 0, 0]  # -w/2

    x_corners += -w / 2
    z_corners += -ll / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape / 2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0, :]))


def draw_birdeyes(ax2, line_gt, line_p, shape):
    scale = 15
    pred_corners_2d = compute_birdviewbox(line_p, shape, scale)
    if line_gt:
        gt_corners_2d = compute_birdviewbox(line_gt, shape, scale)
        codes = [Path.LINETO] * gt_corners_2d.shape[0]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        pth = Path(gt_corners_2d, codes)
        p = patches.PathPatch(pth, fill=False, color="orange", label="ground truth")
        ax2.add_patch(p)

    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color="green", label="prediction")
    ax2.add_patch(p)


def compute_3Dbox(P2, line):
    obj = detectionInfo(line)

    # Draw 3D Bounding Box
    R = np.array(
        [
            [np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
            [0, 1, 0],
            [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)],
        ]
    )

    x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h]  # -h
    z_corners = [0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

    x_corners = [i - obj.l / 2 for i in x_corners]
    y_corners = [i - obj.h for i in y_corners]
    z_corners = [i - obj.w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([obj.tx, obj.ty, obj.tz]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D


def draw_3Dbox(ax, P2, line, color):
    corners_2D = compute_3Dbox(P2, line)

    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
    # put a mask on the front
    front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
    ax.add_patch(p)
    ax.add_patch(front_fill)


def generate_kitti_3d_detection(prediction, predict_txt):
    with open(predict_txt, "w", newline="") as f:
        w = csv.writer(f, delimiter=" ", lineterminator="\n")
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                p = p.round(4)
                type = ID_TYPE_CONVERSION[int(p[0])]
                row = [type, 0, 0] + p[1:].tolist()
                w.writerow(row)


def visualization(save_path, label_path, calib_matrix, predictions, pred_path, image, threshold, dataset, VEHICLES):
    start_frame = 0
    end_frame = 1

    for index in range(start_frame, end_frame):
        label_file = os.path.join(label_path, dataset[index] + ".txt")
        if os.path.isfile(label_file):
            draw_labels = True
        else:
            draw_labels = False
        prediction_file = os.path.join(pred_path, dataset[index] + ".txt")
        # generating the prediction file:
        generate_kitti_3d_detection(predictions, prediction_file)

        P2 = calib_matrix

        fig = plt.figure(figsize=(20.00, 5.12), dpi=100)

        gs = GridSpec(1, 4)
        gs.update(wspace=0)  # set the spacing between axes.

        ax = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 3:])

        image = Image.fromarray(image[0, :, :, :])
        shape = 900
        birdimage = np.zeros((shape, shape, 3), np.uint8)
        found_detections = True

        if draw_labels:
            with open(label_file) as f1, open(prediction_file) as f2:
                for line_gt, line_p in zip(f1, f2):
                    line_gt = line_gt.strip().split(" ")
                    line_p = line_p.strip().split(" ")
                    try:
                        truncated = np.abs(float(line_p[1]))
                    except Exception:
                        found_detections = False
                    trunc_level = 255

                    # truncated object in dataset is not observable
                    if (
                        line_p[0] in VEHICLES
                        and truncated < trunc_level
                        and found_detections
                        and P2 != "empty"
                        and float(line_p[-1]) > threshold
                    ):
                        color = "green"
                        if line_p[0] == "Cyclist":
                            color = "yellow"
                        elif line_p[0] == "Pedestrian":
                            color = "cyan"
                        draw_3Dbox(ax, P2, line_p, color)
                        draw_birdeyes(ax2, line_gt, line_p, shape)
        else:
            line_gt = None
            with open(prediction_file) as f2:
                for line_p in f2:
                    line_p = line_p.strip().split(" ")
                    try:
                        truncated = np.abs(float(line_p[1]))
                    except Exception:
                        found_detections = False
                    trunc_level = 255

                    # truncated object in dataset is not observable
                    if (
                        line_p[0] in VEHICLES
                        and truncated < trunc_level
                        and found_detections
                        and P2 != "empty"
                        and float(line_p[-1]) > threshold
                    ):
                        color = "green"
                        if line_p[0] == "Cyclist":
                            color = "yellow"
                        elif line_p[0] == "Pedestrian":
                            color = "cyan"
                        draw_3Dbox(ax, P2, line_p, color)
                        draw_birdeyes(ax2, line_gt, line_p, shape)
        # visualize 3D bounding box
        ax.imshow(image)
        ax.set_xticks([])  # remove axis value
        ax.set_yticks([])

        # plot camera view range
        x1 = np.linspace(0, shape / 2)
        x2 = np.linspace(shape / 2, shape)
        ax2.plot(x1, shape / 2 - x1, ls="--", color="grey", linewidth=1, alpha=0.5)
        ax2.plot(x2, x2 - shape / 2, ls="--", color="grey", linewidth=1, alpha=0.5)
        ax2.plot(shape / 2, 0, marker="+", markersize=16, markeredgecolor="red")

        # visualize bird eye view
        ax2.imshow(birdimage, origin="lower")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # we're expected to return a [h,w,3] numpy!
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        drawn_image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close("all")
        return drawn_image


def visualize_hailo(logits, image, image_name, threshold, calib_matrix):
    base_dir_adk = os.path.expanduser("~/hailo_repos/adk/networks/hailo_networks/")
    save_path = os.path.join(base_dir_adk, "3d_vis_results")
    label_path = os.path.join(base_dir_adk, "models_files/kitti_3d/label")
    pred_path = os.path.join(base_dir_adk, "predictions")

    VEHICLES = ["Car", "Cyclist", "Pedestrian"]
    dataset = [image_name.split(".")[0]]
    logits = logits[0]  # removing batch dim
    return visualization(save_path, label_path, calib_matrix, logits, pred_path, image, threshold, dataset, VEHICLES)
