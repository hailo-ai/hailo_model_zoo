import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="nuscenes_backbone")
def parse_petrv2_backbone_record(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "img": tf.io.FixedLenFeature([], tf.string),
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    img = tf.io.decode_raw(features["img"], tf.float32)
    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    img = tf.reshape(img, (height, width, 3))
    image_info = {"orig_height": height, "orig_width": width}

    return [img, image_info]


@DATASET_FACTORY.register(name="nuscenes")
def parse_petrv2_transformer_record(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "mlvl_feats": tf.io.FixedLenFeature([], tf.string),
            "coords_3d": tf.io.FixedLenFeature([], tf.string),
            "feats_height": tf.io.FixedLenFeature([], tf.int64),
            "feats_width": tf.io.FixedLenFeature([], tf.int64),
            "coords_height": tf.io.FixedLenFeature([], tf.int64),
            "coords_width": tf.io.FixedLenFeature([], tf.int64),
            "mlvl_feats_ch": tf.io.FixedLenFeature([], tf.int64),
            "coords_3d_ch": tf.io.FixedLenFeature([], tf.int64),
            "ind": tf.io.FixedLenFeature([], tf.int64),
            "timestamp": tf.io.VarLenFeature(tf.float32),
            "token": tf.io.FixedLenFeature([], tf.string),
            "lidar2ego_rotation": tf.io.VarLenFeature(tf.float32),
            "lidar2ego_translation": tf.io.VarLenFeature(tf.float32),
            "ego2global_rotation": tf.io.VarLenFeature(tf.float32),
            "ego2global_translation": tf.io.VarLenFeature(tf.float32),
        },
    )

    ind = tf.cast(features["ind"], tf.int32)
    token = tf.cast(features["token"], tf.string)
    lidar2ego_rotation = tf.sparse.to_dense(features["lidar2ego_rotation"], default_value=0)
    lidar2ego_translation = tf.sparse.to_dense(features["lidar2ego_translation"], default_value=0)
    ego2global_rotation = tf.sparse.to_dense(features["ego2global_rotation"], default_value=0)
    ego2global_translation = tf.sparse.to_dense(features["ego2global_translation"], default_value=0)
    timestamp = tf.sparse.to_dense(features["timestamp"], default_value=0)

    mlvl_feats_ch = tf.cast(features["mlvl_feats_ch"], tf.int32)
    mlvl_feats = tf.io.decode_raw(features["mlvl_feats"], tf.float32)
    feats_height = tf.cast(features["feats_height"], tf.int32)
    feats_width = tf.cast(features["feats_width"], tf.int32)
    mlvl_feats_shape = (feats_height, feats_width, mlvl_feats_ch)
    mlvl_feats = tf.reshape(mlvl_feats, mlvl_feats_shape)

    coords_3d_ch = tf.cast(features["coords_3d_ch"], tf.int32)
    coords_3d = tf.io.decode_raw(features["coords_3d"], tf.float32)
    coords_height = tf.cast(features["coords_height"], tf.int32)
    coords_width = tf.cast(features["coords_width"], tf.int32)
    coords_3d_shape = (coords_height, coords_width, coords_3d_ch)
    coords_3d = tf.reshape(coords_3d, coords_3d_shape)

    images = {"mlvl_feats": mlvl_feats, "coords_3d": coords_3d}
    image_info = {
        "ind": ind,
        "token": token,
        "timestamp": timestamp,
        "lidar2ego_rotation": lidar2ego_rotation,
        "lidar2ego_translation": lidar2ego_translation,
        "ego2global_rotation": ego2global_rotation,
        "ego2global_translation": ego2global_translation,
    }

    return [images, image_info]
