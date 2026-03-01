import argparse
import logging
import os

import cv2
import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver

TF_RECORD_TYPE = {}
TF_RECORD_TYPE["val"] = "val"
TF_RECORD_TYPE["calib"] = "train"
TF_RECORD_LOC = {
    "val": "models_files/k400/2024-09-03/k400_val.tfrecord",
    "calib": "models_files/k400/2024-09-03/k400_calib.tfrecord",
}

# assuming .../k400/videos/val/
#          .../k400/videos/train/
#          .../k400/videos/test/

k400_PATH = "/local/adk_models_files/models_files/k400/"


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def process_video(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning(f"Cannot open video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        logging.warning(f"Not enough frames in video: {video_path}")
        return None

    frame_step = total_frames // num_frames
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = cap.read()
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            logging.error(f"Error converting frame {i} in video {video_path}: {str(e)}")
            continue  # Skip this frame and continue
        frames.append(frame)

    cap.release()

    # Check if we have enough frames before returning
    if len(frames) == num_frames:
        return tf.stack(frames, axis=-1)  # TensorShape([720, 1280, 3, 16])
    else:
        logging.warning(f"Not enough valid frames extracted from video: {video_path}")
        return None


def extract_tf_record(data_path, type, num_videos, num_frames):
    tf_record_path = TF_RECORD_LOC[type]
    tf_record_path = path_resolver.resolve_data_path(tf_record_path)
    progress_bar = tqdm(total=num_videos * 400)
    with tf.io.TFRecordWriter(str(tf_record_path)) as writer:
        data_path = data_path + "videos/" + TF_RECORD_TYPE[type]
        video_folders = os.listdir(data_path)
        video_folders.sort()
        for i, class_folder in enumerate(video_folders):
            folder_path = os.path.join(data_path, class_folder)
            if os.path.isdir(folder_path):
                mp4_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
                selected_files = mp4_files[:num_videos]
            for mp4_file in selected_files:
                video_path = os.path.join(folder_path, mp4_file)
                try:
                    tensor = process_video(video_path, num_frames)
                    if tensor is None:
                        logging.warning(f"Skipping video due to insufficient frames: {video_path}")
                        continue
                    tensor_bytes = tensor.numpy().tobytes()
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "height": _int64_feature(tensor.shape[0]),
                                "width": _int64_feature(tensor.shape[1]),
                                "video": _bytes_feature(tensor_bytes),
                                "tensor_name": _bytes_feature(str.encode(os.path.basename(video_path))),
                                "label": _int64_feature(i),
                            }
                        )
                    )
                    writer.write(example.SerializeToString())
                except Exception as e:
                    logging.error(f"Error processing video {video_path}: {str(e)}")
                    continue  # Continue with the next video if an error occurs
                progress_bar.update(1)
    progress_bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help="which tf-record to create {}".format(TF_RECORD_TYPE))
    parser.add_argument("--data", help="k400 data directory", type=str, default=k400_PATH)
    parser.add_argument("--num-videos", type=int, default=10, help="Number of mp4 to take from each class folder")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames to take from each mp4")
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, "need to provide which kind of tfrecord to create {}".format(TF_RECORD_TYPE)
    extract_tf_record(args.data, args.type, args.num_videos, args.num_frames)
