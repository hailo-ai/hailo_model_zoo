import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="dotav1")
def parse_dotav1_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information for DATA v1 dataset."""
    sparse_float_keys = ["area", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    sparse_int_keys = ["is_crowd", "category_id"]
    feature_spec = {
        # Fixed-length features (scalars)
        **{
            key: tf.io.FixedLenFeature([], dtype)
            for key, dtype in [
                ("image_id", tf.int64),
                ("height", tf.int64),
                ("width", tf.int64),
                ("image_name", tf.string),
                ("image_jpeg", tf.string),
                ("num_boxes", tf.int64),
            ]
        },
        # Variable-length features (arrays)
        **{key: tf.io.VarLenFeature(tf.float32) for key in sparse_float_keys},
        # Variable-length integer features
        **{key: tf.io.VarLenFeature(tf.int64) for key in sparse_int_keys},
    }
    features = tf.io.parse_single_example(serialized_example, features=feature_spec)
    dense_features = {
        **{key: tf.sparse.to_dense(features[key], default_value=0.0) for key in sparse_float_keys},
        **{key: tf.sparse.to_dense(features[key], default_value=0) for key in sparse_int_keys},
    }
    image = tf.image.decode_jpeg(features["image_jpeg"], channels=3)
    height, width = tf.cast(features["height"], tf.int32), tf.cast(features["width"], tf.int32)
    image = tf.cast(tf.reshape(image, [height, width, 3]), tf.uint8)

    image_info = {
        "image_name": features["image_name"],
        "height": height,
        "width": width,
        "num_boxes": tf.cast(features["num_boxes"], tf.int32),
        "image_id": tf.cast(features["image_id"], tf.int32),
        **dense_features,
    }

    return [image, image_info]
