import os
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _parse_gt_txt(gt_txt_path):
    all_frames_data = {}
    with open(gt_txt_path) as gt:
        for i, line in enumerate(gt):
            if i % 10000 == 0:
                print('parsing label', i, end='\r')
            frame, person_id, x, y, w, h, mark, label = tuple(map(int, line.split(',')[:-1]))
            visibility_ratio = float(line.split(',')[-1])
            if frame not in all_frames_data:
                all_frames_data[frame] = {'person_id': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [], 'mark': [],
                                          'label': [], 'visibility_ratio': [], 'is_ignore': []}
            if label == 1:
                is_ignore = False
            elif label in {2, 7, 8, 12}:
                is_ignore = True
            else:
                continue

            all_frames_data[frame]['person_id'].append(person_id)
            all_frames_data[frame]['is_ignore'].append(int(is_ignore))
            all_frames_data[frame]['xmin'].append(x)
            all_frames_data[frame]['ymin'].append(y)
            all_frames_data[frame]['xmax'].append(w + x)
            all_frames_data[frame]['ymax'].append(h + y)
            all_frames_data[frame]['mark'].append(mark)  # No idea what this is
            all_frames_data[frame]['label'].append(label)  # Kind of redundant, there's only 1 label
            all_frames_data[frame]['visibility_ratio'].append(visibility_ratio)  # Not sure if used

    return all_frames_data


def _write_labeled_images_for_video(video_dir, writer, max_frames=None):
    print('parsing video {}'.format(video_dir))
    gt_txt = os.path.join(video_dir, 'gt', 'gt.txt')
    images_dir = os.path.join(video_dir, 'img1')
    all_frames_data = _parse_gt_txt(gt_txt)
    print('writing video to tf record')
    count = 0

    video_name = str.encode("{:<20}".format(os.path.basename(os.path.dirname(images_dir))))

    for image_name in sorted(os.listdir(images_dir))[:max_frames]:
        image_path = os.path.join(images_dir, image_name)
        img_jpeg = open(image_path, 'rb').read()
        filename = str.encode("{:<20}".format(os.path.basename(image_path)))
        img = np.array(Image.open(image_path))
        height = img.shape[0]
        width = img.shape[1]
        frame = int(image_name.split('.')[0])
        if frame % 100 == 0:
            print('writing frame ', frame, end='\r')
        frame_data = all_frames_data[frame]
        example = tf.train.Example(features=tf.train.Features(feature={
            'video_name': _bytes_feature(video_name),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_name': _bytes_feature(filename),
            'image_jpeg': _bytes_feature(img_jpeg),
            'person_id': _int64_feature(frame_data['person_id']),
            'xmin': _int64_feature(frame_data['xmin']),
            'ymin': _int64_feature(frame_data['ymin']),
            'xmax': _int64_feature(frame_data['xmax']),
            'ymax': _int64_feature(frame_data['ymax']),
            'mark': _int64_feature(frame_data['mark']),
            'label': _int64_feature(frame_data['label']),
            'visibility_ratio': _float_list_feature(frame_data['visibility_ratio']),
            'is_ignore': _int64_feature(frame_data['is_ignore'])
        }))
        writer.write(example.SerializeToString())
        count += 1
    return count


def run(dataset_dir, is_calibration=False, name='val'):
    assert dataset_dir != '' and os.path.isdir(dataset_dir), 'no dataset directory'
    if 'MOT16' in dataset_dir:
        ds_name = 'mot16'
        sub_dirs = [i for i in os.listdir(dataset_dir) if i.startswith('MOT16')]
        assert (set(sub_dirs) == {'MOT16-02',
                                  'MOT16-04',
                                  'MOT16-05',
                                  'MOT16-09',
                                  'MOT16-10',
                                  'MOT16-11',
                                  'MOT16-13'}), \
            'Dataset dir does not contain expected dirs. is it MOT16/train?'
    elif 'MOT17' in dataset_dir:
        ds_name = 'mot17'
        sub_dirs = [i for i in os.listdir(dataset_dir) if i.startswith('MOT17')]

    if is_calibration and name != 'calibration_set':
        print("Calibration set creation is chosen but file name suffix is {}... "
              "Setting it to 'calibration_set'...".format(name))
        name = 'calibration_set'

    tfrecords_filename = os.path.join(dataset_dir, ds_name + name + '.tfrecord')
    writer = tf.io.TFRecordWriter(tfrecords_filename)

    if is_calibration:
        print('ONLY PARSING MOT16-04, since this is calibration set')
        sub_dirs = ['MOT16-04']

    total = 0
    for video in sub_dirs:
        total += _write_labeled_images_for_video(os.path.join(dataset_dir, video), writer,
                                                 max_frames=128 if is_calibration else None)

    writer.close()
    print('Added {} images to tf record'.format(total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help="dataset directory. should be MOT16/train (test does not contain GT)", type=str,
                        default='/local/data/datasets/MOT/FairMOT-includes_other/MOT16/train')
    parser.add_argument('--calibration_set', '-c',
                        help="Create a calibration set containing only MOT16-04 video",
                        action='store_true', default=False)
    parser.add_argument('--name', '-name', help="file name suffix", type=str, default='val')

    args = parser.parse_args()
    run(args.dir, args.calibration_set, args.name)
"""
----------------------------------------------------------------------------
CMD used to create a mot16val.tfrecord of the MOT16 training dataset:
----------------------------------------------------------------------------
python create_mot_tfrecord.py
--dir /local/data/datasets/MOT/FairMOT-includes_other/MOT16/train
--name val

----------------------------------------------------------------------------
CMD used to create a mot17train.tfrecord of the MOT17 training dataset:
----------------------------------------------------------------------------
python create_mot_tfrecord.py
--dir /local/data/datasets/MOT/FairMOT-includes_other/MOT17/images/train
--name train
"""
