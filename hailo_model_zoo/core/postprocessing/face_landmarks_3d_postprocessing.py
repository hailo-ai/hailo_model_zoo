import pickle

import cv2
import numpy as np
import tensorflow as tf

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY
from hailo_model_zoo.core.infer.infer_utils import to_numpy
from hailo_model_zoo.utils.path_resolver import resolve_data_path


def face_landmarks_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    shape = kwargs["img_dims"]
    return {"predictions": endnodes * shape[0]}


TDDFA_RESCALE_PARAMS = {
    "mean": np.array(
        [
            [
                3.4926363e-04,
                2.5279013e-07,
                -6.8751979e-07,
                6.0167957e01,
                -6.2955132e-07,
                5.7572004e-04,
                -5.0853912e-05,
                7.4278198e01,
                5.4009172e-07,
                6.5741384e-05,
                3.4420125e-04,
                -6.6671577e01,
                -3.4660369e05,
                -6.7468234e04,
                4.6822266e04,
                -1.5262047e04,
                4.3505889e03,
                -5.4261453e04,
                -1.8328033e04,
                -1.5843289e03,
                -8.4566344e04,
                3.8359607e03,
                -2.0811361e04,
                3.8094930e04,
                -1.9967855e04,
                -9.2413701e03,
                -1.9600715e04,
                1.3168090e04,
                -5.2591440e03,
                1.8486478e03,
                -1.3030662e04,
                -2.4355562e03,
                -2.2542065e03,
                -1.4396562e04,
                -6.1763291e03,
                -2.5621920e04,
                2.2639447e02,
                -6.3261235e03,
                -1.0867251e04,
                8.6846509e02,
                -5.8311479e03,
                2.7051238e03,
                -3.6294177e03,
                2.0439901e03,
                -2.4466162e03,
                3.6586970e03,
                -7.6459897e03,
                -6.6744526e03,
                1.1638839e02,
                7.1855972e03,
                -1.4294868e03,
                2.6173665e03,
                -1.2070955e00,
                6.6907924e-01,
                -1.7760828e-01,
                5.6725528e-02,
                3.9678156e-02,
                -1.3586316e-01,
                -9.2239931e-02,
                -1.7260718e-01,
                -1.5804484e-02,
                -1.4168486e-01,
            ]
        ],
        dtype=np.float32,
    ),
    "std": np.array(
        [
            [
                1.76321526e-04,
                6.73794348e-05,
                4.47084894e-04,
                2.65502319e01,
                1.23137695e-04,
                4.49302170e-05,
                7.92367064e-05,
                6.98256302e00,
                4.35044407e-04,
                1.23148900e-04,
                1.74000015e-04,
                2.08030396e01,
                5.75421125e05,
                2.77649062e05,
                2.58336844e05,
                2.55163125e05,
                1.50994375e05,
                1.60086109e05,
                1.11277305e05,
                9.73117812e04,
                1.17198453e05,
                8.93173672e04,
                8.84935547e04,
                7.22299297e04,
                7.10802109e04,
                5.00139531e04,
                5.59685820e04,
                4.75255039e04,
                4.95150664e04,
                3.81614805e04,
                4.48720586e04,
                4.62732383e04,
                3.81167695e04,
                2.81911621e04,
                3.21914375e04,
                3.60061719e04,
                3.25598926e04,
                2.55511172e04,
                2.42675098e04,
                2.75213984e04,
                2.31665312e04,
                2.11015762e04,
                1.94123242e04,
                1.94522031e04,
                1.74549844e04,
                2.25376230e04,
                1.61742812e04,
                1.46716406e04,
                1.51156885e04,
                1.38700732e04,
                1.37463125e04,
                1.26631338e04,
                1.58708346e00,
                1.50770092e00,
                5.88135779e-01,
                5.88974476e-01,
                2.13278517e-01,
                2.63020128e-01,
                2.79642940e-01,
                3.80302161e-01,
                1.61628410e-01,
                2.55969286e-01,
            ]
        ],
        dtype=np.float32,
    ),
}


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr


class BFMModel(object):
    def __init__(self, shape_dim=40, exp_dim=10):
        # FUTURE these should be downloaded lazily
        bfm_path = resolve_data_path(
            "models_files/FaceLandmarks3d/tddfa/tddfa_mobilenet_v1/pretrained/2021-11-28/bfm_noneck_v3.pkl"
        )
        tri_path = resolve_data_path(
            "models_files/FaceLandmarks3d/tddfa/tddfa_mobilenet_v1/pretrained/2021-11-28/tri.pkl"
        )
        if not bfm_path.exists() or not tri_path.exists():
            raise FileNotFoundError(
                "Please sign the license and download the face model manually. "
                "For further information, see: https://github.com/cleardusk/3DDFA_V2/tree/master/bfm"
            )
        with open(bfm_path, "rb") as f:
            bfm = pickle.load(f)
        self._u = bfm.get("u").astype(np.float32)  # fix bug
        self._w_shp = bfm.get("w_shp").astype(np.float32)[..., :shape_dim]
        self._w_exp = bfm.get("w_exp").astype(np.float32)[..., :exp_dim]

        with open(tri_path, "rb") as f:
            self._tri = pickle.load(f)  # this tri/face is re-built for bfm_noneck_v3

        self._tri = _to_ctype(self._tri.T).astype(np.int32)
        self._keypoints = bfm.get("keypoints").astype(int)  # fix bug
        w = np.concatenate((self._w_shp, self._w_exp), axis=1)
        self._w_norm = np.linalg.norm(w, axis=0)

        self.u_base = self._u[self._keypoints].reshape(-1, 1)
        self.w_shp_base = self._w_shp[self._keypoints]
        self.w_exp_base = self._w_exp[self._keypoints]


class BFM_s:
    def __init__(self) -> None:
        self._BFM = None

    def get_bfm(self):
        self._BFM = self._BFM or BFMModel()
        return self._BFM


# BFM = BFMModel()
bfm_s = BFM_s()


def _parse_param(param):
    # pre-defined templates for parameter
    n = param.shape[0]
    if n == 62:
        trans_dim, shape_dim, _ = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, _ = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, _ = 12, 100, 29
    else:
        raise Exception("Undefined templated param parsing rule")

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim : trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim :].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp


def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1  # for Python compatibility
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]

    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])
    return np.array(pts3d, dtype=np.float32)


def face_3dmm_to_landmarks_batch(face_3dmm_params, img_dims, roi_box):
    ptds3d_list = []
    for params, box in zip(face_3dmm_params, roi_box):
        pts3d = face_3dmm_to_landmarks_np(params, img_dims, box)
        ptds3d_list.append(pts3d)
    return np.array(ptds3d_list)


def face_3dmm_to_landmarks_np(face_3dmm_params, img_dims, roi_box):
    R, offset, alpha_shp, alpha_exp = _parse_param(face_3dmm_params)
    pts3d = (
        R
        @ (
            bfm_s.get_bfm().u_base + bfm_s.get_bfm().w_shp_base @ alpha_shp + bfm_s.get_bfm().w_exp_base @ alpha_exp
        ).reshape(3, -1, order="F")
        + offset
    )
    pts3d = similar_transform(pts3d, roi_box, img_dims[0])
    pts3d = pts3d.transpose()
    return pts3d


@POSTPROCESS_FACTORY.register(name="face_landmark_detection_3d")
def face_landmarks_3d_postprocessing(endnodes, device_pre_post_layers=None, *, img_dims=None, gt_images=None, **kwargs):
    assert img_dims[0] == img_dims[1], "Assumes square input"
    batch_size = tf.shape(endnodes)[0]
    endnodes = tf.reshape(endnodes, [batch_size, -1])
    face_3dmm_params = endnodes * TDDFA_RESCALE_PARAMS["std"] + TDDFA_RESCALE_PARAMS["mean"]
    roi_box = gt_images.get("roi_box", tf.tile([[0, 0, img_dims[1], img_dims[0]]], (batch_size, 1)))
    ptds3d = tf.numpy_function(face_3dmm_to_landmarks_batch, [face_3dmm_params, img_dims, roi_box], tf.float32)
    return {"predictions": ptds3d}


@VISUALIZATION_FACTORY.register(name="face_landmark_detection_3d")
def visualize_face_landmarks_3d_result(logits, image, **kwargs):
    logits = logits["predictions"]
    img = to_numpy(kwargs.get("img_info", {}).get("uncropped_image", image[0]))
    box = to_numpy(kwargs.get("img_info", {}).get("roi_box"))

    for landmark in logits[0, :, :2]:
        landmark_as_int = tuple(int(x) for x in landmark)
        img = cv2.circle(img, landmark_as_int, 1, (255, 0, 255), -1)

    if box is not None:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=1)
    return img
