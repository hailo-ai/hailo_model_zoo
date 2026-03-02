import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="icdar15_det")
def parse_icdar15_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "num_boxes": tf.io.FixedLenFeature([], tf.int64),
            "xmin": tf.io.VarLenFeature(tf.float32),
            "xmax": tf.io.VarLenFeature(tf.float32),
            "ymin": tf.io.VarLenFeature(tf.float32),
            "ymax": tf.io.VarLenFeature(tf.float32),
            "tags": tf.io.VarLenFeature(tf.string),
            "area": tf.io.VarLenFeature(tf.float32),
            "image_jpeg": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "category_id": tf.io.VarLenFeature(tf.float32),
            "is_crowd": tf.io.VarLenFeature(tf.float32),
            "image_id": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    num_boxes = tf.cast(features["num_boxes"], tf.int32)

    xmin = tf.sparse.to_dense(features["xmin"], default_value=0)
    xmax = tf.sparse.to_dense(features["xmax"], default_value=0)
    ymin = tf.sparse.to_dense(features["ymin"], default_value=0)
    ymax = tf.sparse.to_dense(features["ymax"], default_value=0)
    areas = tf.sparse.to_dense(features["area"], default_value=0)

    image_name = tf.cast(features["image_name"], tf.string)
    image = tf.image.decode_jpeg(features["image_jpeg"], channels=3)
    image_shape = tf.stack([height, width, 3])
    image = tf.cast(tf.reshape(image, image_shape), tf.uint8)
    image = tf.reverse(image, axis=[-1])  # RGB to BGR conversion

    image_id = tf.cast(features["image_id"], tf.int32)
    cat_id = tf.sparse.to_dense(features["category_id"], default_value=-1)
    is_crowd = tf.sparse.to_dense(features["is_crowd"], default_value=0)

    image_info = {
        "image_name": image_name,
        "num_boxes": num_boxes,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "area": areas,
        "width": width,
        "height": height,
        "image_id": image_id,
        "category_id": cat_id,
        "is_crowd": is_crowd,
    }

    return [image, image_info]
