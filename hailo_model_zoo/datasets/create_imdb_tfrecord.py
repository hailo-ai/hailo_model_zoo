#!/usr/bin/env python

import os
import argparse
import io
import tensorflow as tf
import PIL
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def create_tf_example(img_path,
                      image_idx,
                      landmarks):
    """Converts image to a tf.Example proto.

    Args:
      file_basename: str,
      image_dir: directory containing the image files.
    Returns:
      example: The converted tf.Example

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    image_width, image_height = image.size
    filename = os.path.basename(img_path)

    feature_dict = {
        'height': _int64_feature(image_height),
        'width': _int64_feature(image_width),
        'image_id': _int64_feature(image_idx),
        'landmarks': _bytes_feature(landmarks.encode('utf8')),
        'image_name': _bytes_feature(filename.encode('utf8')),
        'image_jpeg': _bytes_feature(encoded_jpg),
        'format': _bytes_feature('jpeg'.encode('utf8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def draw_img(image_path, pts_int):
    from PIL import Image, ImageDraw
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    draw.point(pts_int)
    img.show()


def _create_tf_record(image_dir, output_path, num_images, landmarks):
    count = 0
    writer = tf.io.TFRecordWriter(output_path)
    images = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(image_dir)) for
              f in fn if f[-3:] == 'png']
    with open(landmarks, 'r') as landmark_f:
        pts = landmark_f.read()
        for idx, image_path in enumerate(images):
            try:
                if count >= num_images:
                    break
                if idx % 100 == 0:
                    tf.compat.v1.logging.info('On image %d of %d', idx, len(images))
                start_idx = pts.find(os.path.basename(image_path)) + len(os.path.basename(image_path)) + 1
                if pts.find(os.path.basename(image_path)) < 0:
                    continue
                end_index = start_idx + pts[start_idx:].find('\n')
                pts_string = pts[start_idx:end_index][1:-1]
                pts_int = [int(x) for x in pts_string.split(',')]
                pts_string = ' '.join([str(x) for x in pts_int])
                tf_example = create_tf_example(image_path, count, pts_string)
                writer.write(tf_example.SerializeToString())
                count += 1
            except Exception:
                pass
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-img', help="images directory", type=str,
                        default='/local/data/datasets/face/CASIA/CASIA-faces')
    parser.add_argument('--pts', '-pts', help="points txt file path", type=str,
                        default='/local/data/datasets/face/CASIA/landmark_gt.txt')
    parser.add_argument('--output-dir', help="output directory", type=str, default='')
    parser.add_argument('--num-images', help="limit the number of images", type=int, default=512)
    args = parser.parse_args()
    output_path = os.path.join(args.output_dir, "imdb_val.tfrecord")
    _create_tf_record(args.img, output_path, args.num_images, args.pts)
