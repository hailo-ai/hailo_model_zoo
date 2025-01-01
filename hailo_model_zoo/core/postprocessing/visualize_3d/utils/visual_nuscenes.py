import math
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix, view_points
from PIL import Image
from pyquaternion import Quaternion

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("nuScenes dev-kit only supports Python version 3.")

detection_mapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}
category_mapping = {}
for k, v in detection_mapping.items():
    category_mapping[v] = k


class NuScenesExplorer:
    """Helper class to list and visualize NuScenes data. These are meant to serve as tutorials and templates for
    working with the data."""

    def __init__(self, nusc):
        self.nusc = nusc

    def get_color(self, category_name: str) -> Tuple[int, int, int]:
        """
        Provides the default colors based on the category names.
        This method works for the general nuScenes categories, as well as the nuScenes detection categories.
        """

        return self.nusc.colormap[category_name]

    def list_sample(self, sample_token: str) -> None:
        """Prints sample_data tokens and sample_annotation tokens related to the sample_token."""

        sample_record = self.nusc.get("sample", sample_token)
        print("Sample: {}\n".format(sample_record["token"]))
        for sd_token in sample_record["data"].values():
            sd_record = self.nusc.get("sample_data", sd_token)
            print(
                "sample_data_token: {}, mod: {}, channel: {}".format(
                    sd_token, sd_record["sensor_modality"], sd_record["channel"]
                )
            )
        print("")
        for ann_token in sample_record["anns"]:
            ann_record = self.nusc.get("sample_annotation", ann_token)
            print("sample_annotation_token: {}, category: {}".format(ann_record["token"], ann_record["category_name"]))

    def render_sample(
        self,
        token: str,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        nsweeps: int = 1,
        out_path: str = None,
        show_lidarseg: bool = False,
        filter_lidarseg_labels: List = None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
    ) -> None:
        """
        Render all LIDAR and camera sample_data in sample along with annotations.
        :param token: Sample token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param show_lidarseg: Whether to show lidar segmentations labels or not.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to show the rendered sample in a window or not.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        record = self.nusc.get("sample", token)
        # Separate RADAR from LIDAR and vision.
        radar_data = {}
        camera_data = {}
        lidar_data = {}
        for channel, token in record["data"].items():
            sd_record = self.nusc.get("sample_data", token)
            sensor_modality = sd_record["sensor_modality"]

            if sensor_modality == "camera":
                camera_data[channel] = token
            # NOTE: Currently lidar / radar data is not visualized, hence commented out
            # elif sensor_modality == 'lidar':
            #     lidar_data[channel] = token
            # else:
            #     radar_data[channel] = token

        # Create plots.
        num_radar_plots = 1 if len(radar_data) > 0 else 0
        num_lidar_plots = 1 if len(lidar_data) > 0 else 0
        n = num_radar_plots + len(camera_data) + num_lidar_plots
        cols = 2
        # cols = 3
        fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(16, 24))

        # Plot radars into a single subplot.
        if len(radar_data) > 0:
            ax = axes[0, 0]
            for i, (_, sd_token) in enumerate(radar_data.items()):
                self.render_sample_data(
                    sd_token, with_anns=i == 0, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps, verbose=False
                )
            ax.set_title("Fused RADARs")

        # Plot lidar into a single subplot.
        if len(lidar_data) > 0:
            for (_, sd_token), ax in zip(lidar_data.items(), axes.flatten()[num_radar_plots:]):
                self.render_sample_data(
                    sd_token,
                    box_vis_level=box_vis_level,
                    ax=ax,
                    nsweeps=nsweeps,
                    show_lidarseg=show_lidarseg,
                    filter_lidarseg_labels=filter_lidarseg_labels,
                    lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                    verbose=False,
                    show_panoptic=show_panoptic,
                )

        # Plot cameras in separate subplots.
        for (_, sd_token), ax in zip(camera_data.items(), axes.flatten()[num_radar_plots + num_lidar_plots :]):
            self.render_sample_data(
                sd_token, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps, show_lidarseg=False, verbose=False
            )

        # Change plot settings and write to disk.
        axes.flatten()[-1].axis("off")
        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

        if out_path is not None:
            plt.savefig(out_path)

        if verbose:
            plt.show()

        return fig

    def render_ego_centric_map(self, sample_data_token: str, axes_limit: float = 40, ax: Axes = None) -> None:
        """
        Render map centered around the associated ego pose.
        :param sample_data_token: Sample_data token.
        :param axes_limit: Axes limit measured in meters.
        :param ax: Axes onto which to render.
        """

        def crop_image(image: np.array, x_px: int, y_px: int, axes_limit_px: int) -> np.array:
            x_min = int(x_px - axes_limit_px)
            x_max = int(x_px + axes_limit_px)
            y_min = int(y_px - axes_limit_px)
            y_max = int(y_px + axes_limit_px)

            cropped_image = image[y_min:y_max, x_min:x_max]

            return cropped_image

        # Get data.
        sd_record = self.nusc.get("sample_data", sample_data_token)
        sample = self.nusc.get("sample", sd_record["sample_token"])
        scene = self.nusc.get("scene", sample["scene_token"])
        log = self.nusc.get("log", scene["log_token"])
        map_ = self.nusc.get("map", log["map_token"])
        map_mask = map_["mask"]
        pose = self.nusc.get("ego_pose", sd_record["ego_pose_token"])

        # Retrieve and crop mask.
        pixel_coords = map_mask.to_pixel_coords(pose["translation"][0], pose["translation"][1])
        scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
        mask_raster = map_mask.mask()
        cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

        # Rotate image.
        ypr_rad = Quaternion(pose["rotation"]).yaw_pitch_roll
        yaw_deg = -math.degrees(ypr_rad[0])
        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

        # Crop image.
        ego_centric_map = crop_image(
            rotated_cropped, int(rotated_cropped.shape[1] / 2), int(rotated_cropped.shape[0] / 2), scaled_limit_px
        )

        # Init axes and show image.
        # Set background to white and foreground (semantic prior) to gray.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))
        ego_centric_map[ego_centric_map == map_mask.foreground] = 125
        ego_centric_map[ego_centric_map == map_mask.background] = 255
        ax.imshow(
            ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit], cmap="gray", vmin=0, vmax=255
        )

    def render_sample_data(
        self,
        sample_data_token: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax: Axes = None,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels: List = None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
    ) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw box annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to display the image after it is rendered.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        # Get sensor modality.
        sd_record = self.nusc.get("sample_data", sample_data_token)
        sensor_modality = sd_record["sensor_modality"]

        if sensor_modality in ["lidar", "radar"]:
            sample_rec = self.nusc.get("sample", sd_record["sample_token"])
            chan = sd_record["channel"]
            ref_chan = "LIDAR_TOP"
            ref_sd_token = sample_rec["data"][ref_chan]
            ref_sd_record = self.nusc.get("sample_data", ref_sd_token)

            if sensor_modality == "lidar":
                # Get aggregated lidar point cloud in lidar frame.
                pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)
                velocities = None
            else:
                # Get aggregated radar point cloud in reference frame.
                # The point cloud is transformed to the reference frame for visualization purposes.
                pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

                # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
                # point cloud.
                radar_cs_record = self.nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
                ref_cs_record = self.nusc.get("calibrated_sensor", ref_sd_record["calibrated_sensor_token"])
                velocities = pc.points[8:10, :]  # Compensated velocity
                velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
                velocities = np.dot(Quaternion(radar_cs_record["rotation"]).rotation_matrix, velocities)
                velocities = np.dot(Quaternion(ref_cs_record["rotation"]).rotation_matrix.T, velocities)
                velocities[2, :] = np.zeros(pc.points.shape[1])

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.nusc.get("calibrated_sensor", ref_sd_record["calibrated_sensor_token"])
                pose_record = self.nusc.get("ego_pose", ref_sd_record["ego_pose_token"])
                ref_to_ego = transform_matrix(
                    translation=cs_record["translation"], rotation=Quaternion(cs_record["rotation"])
                )

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record["rotation"]).inverse.rotation_matrix,
                )
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Render map if requested.
            if underlay_map:
                assert use_flat_vehicle_coordinates, (
                    "Error: underlay_map requires use_flat_vehicle_coordinates, as "
                    "otherwise the location does not correspond to the map!"
                )
                self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))

            point_scale = 0.2 if sensor_modality == "lidar" else 3.0
            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

            # Show velocities.
            if sensor_modality == "radar":
                points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
                deltas_vel = points_vel - points
                deltas_vel = 6 * deltas_vel  # Arbitrary scaling
                max_delta = 20
                deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
                colors_rgba = scatter.to_rgba(colors)
                for i in range(points.shape[1]):
                    ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

            # Show ego vehicle.
            ax.plot(0, 0, "x", color="red")

            # Get boxes in lidar frame.
            _, boxes, _ = self.nusc.get_sample_data(
                ref_sd_token, box_vis_level=box_vis_level, use_flat_vehicle_coordinates=use_flat_vehicle_coordinates
            )

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)
        elif sensor_modality == "camera":
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(
                sample_data_token, box_vis_level=box_vis_level
            )
            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)
        else:
            raise ValueError(f"Error: Unknown sensor modality {sensor_modality}!")

        ax.axis("off")
        ax.set_title(
            "{} {labels_type}".format(
                sd_record["channel"], labels_type="(predictions)" if lidarseg_preds_bin_path else ""
            )
        )
        ax.set_aspect("equal")

        if out_path is not None:
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=200)

        if verbose:
            plt.show()
