import pickle
import cv2
import numpy as np
import tensorflow as tf

from hailo_model_zoo.utils.path_resolver import resolve_data_path


def face_landmarks_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    shape = kwargs['img_dims']
    return {'predictions': endnodes * shape[0]}


TDDFA_RESCALE_PARAMS = {'mean': np.array([[3.4926363e-04, 2.5279013e-07, -6.8751979e-07, 6.0167957e+01,
                                           -6.2955132e-07, 5.7572004e-04, -5.0853912e-05, 7.4278198e+01,
                                           5.4009172e-07, 6.5741384e-05, 3.4420125e-04, -6.6671577e+01,
                                           -3.4660369e+05, -6.7468234e+04, 4.6822266e+04, -1.5262047e+04,
                                           4.3505889e+03, -5.4261453e+04, -1.8328033e+04, -1.5843289e+03,
                                           -8.4566344e+04, 3.8359607e+03, -2.0811361e+04, 3.8094930e+04,
                                           -1.9967855e+04, -9.2413701e+03, -1.9600715e+04, 1.3168090e+04,
                                           -5.2591440e+03, 1.8486478e+03, -1.3030662e+04, -2.4355562e+03,
                                           -2.2542065e+03, -1.4396562e+04, -6.1763291e+03, -2.5621920e+04,
                                           2.2639447e+02, -6.3261235e+03, -1.0867251e+04, 8.6846509e+02,
                                           -5.8311479e+03, 2.7051238e+03, -3.6294177e+03, 2.0439901e+03,
                                           -2.4466162e+03, 3.6586970e+03, -7.6459897e+03, -6.6744526e+03,
                                           1.1638839e+02, 7.1855972e+03, -1.4294868e+03, 2.6173665e+03,
                                           -1.2070955e+00, 6.6907924e-01, -1.7760828e-01, 5.6725528e-02,
                                           3.9678156e-02, -1.3586316e-01, -9.2239931e-02, -1.7260718e-01,
                                           -1.5804484e-02, -1.4168486e-01]], dtype=np.float32),
                        'std': np.array([[1.76321526e-04, 6.73794348e-05, 4.47084894e-04, 2.65502319e+01,
                                          1.23137695e-04, 4.49302170e-05, 7.92367064e-05, 6.98256302e+00,
                                          4.35044407e-04, 1.23148900e-04, 1.74000015e-04, 2.08030396e+01,
                                          5.75421125e+05, 2.77649062e+05, 2.58336844e+05, 2.55163125e+05,
                                          1.50994375e+05, 1.60086109e+05, 1.11277305e+05, 9.73117812e+04,
                                          1.17198453e+05, 8.93173672e+04, 8.84935547e+04, 7.22299297e+04,
                                          7.10802109e+04, 5.00139531e+04, 5.59685820e+04, 4.75255039e+04,
                                          4.95150664e+04, 3.81614805e+04, 4.48720586e+04, 4.62732383e+04,
                                          3.81167695e+04, 2.81911621e+04, 3.21914375e+04, 3.60061719e+04,
                                          3.25598926e+04, 2.55511172e+04, 2.42675098e+04, 2.75213984e+04,
                                          2.31665312e+04, 2.11015762e+04, 1.94123242e+04, 1.94522031e+04,
                                          1.74549844e+04, 2.25376230e+04, 1.61742812e+04, 1.46716406e+04,
                                          1.51156885e+04, 1.38700732e+04, 1.37463125e+04, 1.26631338e+04,
                                          1.58708346e+00, 1.50770092e+00, 5.88135779e-01, 5.88974476e-01,
                                          2.13278517e-01, 2.63020128e-01, 2.79642940e-01, 3.80302161e-01,
                                          1.61628410e-01, 2.55969286e-01]], dtype=np.float32)}


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


class BFMModel(object):
    def __init__(self, shape_dim=40, exp_dim=10):
        # FUTURE these should be downloaded lazily
        bfm_path = resolve_data_path(
            "models_files/FaceLandmarks3d/tddfa/tddfa_mobilenet_v1/pretrained/2021-11-28/bfm_noneck_v3.pkl")
        tri_path = resolve_data_path(
            "models_files/FaceLandmarks3d/tddfa/tddfa_mobilenet_v1/pretrained/2021-11-28/tri.pkl")
        with open(bfm_path, 'rb') as f:
            bfm = pickle.load(f)
        self._u = bfm.get('u').astype(np.float32)  # fix bug
        self._w_shp = bfm.get('w_shp').astype(np.float32)[..., :shape_dim]
        self._w_exp = bfm.get('w_exp').astype(np.float32)[..., :exp_dim]

        with open(tri_path, 'rb') as f:
            self._tri = pickle.load(f)  # this tri/face is re-built for bfm_noneck_v3

        self._tri = _to_ctype(self._tri.T).astype(np.int32)
        self._keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
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
        raise Exception('Undefined templated param parsing rule')

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

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
    pts3d = R @ (bfm_s.get_bfm().u_base + bfm_s.get_bfm().w_shp_base @ alpha_shp + bfm_s.get_bfm().w_exp_base @
                 alpha_exp).reshape(3, -1, order='F') + offset
    pts3d = similar_transform(pts3d, roi_box, img_dims[0])
    pts3d = pts3d.transpose()
    return pts3d


def face_landmarks_3d_postprocessing(endnodes, device_pre_post_layers=None, *, img_dims=None, gt_images=None, **kwargs):
    assert img_dims[0] == img_dims[1], "Assumes square input"
    batch_size = tf.shape(endnodes)[0]
    endnodes = tf.reshape(endnodes, [batch_size, -1])
    face_3dmm_params = endnodes * TDDFA_RESCALE_PARAMS['std'] + TDDFA_RESCALE_PARAMS['mean']
    roi_box = gt_images.get('roi_box',
                            tf.tile([[0, 0, img_dims[1], img_dims[0]]], (batch_size, 1)))
    ptds3d = tf.compat.v1.py_func(face_3dmm_to_landmarks_batch,
                                  [face_3dmm_params, img_dims, roi_box],
                                  tf.float32)
    return {'predictions': ptds3d}


def visualize_face_landmarks_3d_result(logits, image, **kwargs):
    logits = logits['predictions']
    img = kwargs.get('img_info', {}).get('uncropped_image', image[0])
    box = kwargs.get('img_info', {}).get('roi_box')

    for landmark in logits[0, :, :2]:
        img = cv2.circle(img, tuple(landmark), 1, (255, 0, 255), -1)

    if box is not None:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=1)
    return img
