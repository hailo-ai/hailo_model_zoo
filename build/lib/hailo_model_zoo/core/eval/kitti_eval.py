import os
import csv
import logging
import subprocess
import errno

from hailo_model_zoo.utils import path_resolver


ID_TYPE_CONVERSION = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian'
}


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def kitti_evaluation(
        eval_type,
        dataset,
        predictions,
        output_folder,
):
    logger = logging.getLogger(__name__)
    if "detection" in eval_type:
        logger.info("performing kitti detection evaluation: ")
        metrics = do_kitti_detection_evaluation(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger)
        return metrics


def do_kitti_detection_evaluation(dataset,
                                  predictions,
                                  output_folder,
                                  logger
                                  ):
    if dataset != 'kitti_3d':
        print('Currently only kitti_3d dataset is supported.')
        return
    predict_folder = os.path.join(output_folder, 'data')  # only recognize data
    mkdir(predict_folder)

    for image_id, prediction in predictions.items():
        predict_txt = image_id + '.txt'
        predict_txt = os.path.join(predict_folder, predict_txt)

        generate_kitti_3d_detection(prediction, predict_txt)

    logger.info("Evaluate on KITTI dataset")
    MODELS_FILES_DIR = path_resolver.resolve_data_path('')
    EVAL_PATH = str(MODELS_FILES_DIR / 'models_files/'
                    'smoke/smoke_regnet800/kitti_3d/evaluate_object_3d_offline_adk')
    label_dir = str(MODELS_FILES_DIR / 'models_files/kitti_3d/label/')
    command = "{} {} {}".format(EVAL_PATH, label_dir, output_folder)
    output = subprocess.check_output(command, shell=True, universal_newlines=True).strip()

    car_bev_AP_e_m_h = output.split('car_detection_ground AP: ')[1][:24].split(' ')
    car_3d_AP_e_m_h = output.split('car_detection_3d AP: ')[1][:24].split(' ')
    car_bev_AP_e_m_h = [float(elem) for elem in car_bev_AP_e_m_h]
    car_3d_AP_e_m_h = [float(elem) for elem in car_3d_AP_e_m_h]

    logger.info(output)
    return car_bev_AP_e_m_h, car_3d_AP_e_m_h


def generate_kitti_3d_detection(prediction, predict_txt):
    with open(predict_txt, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                p = p.round(4)
                type = ID_TYPE_CONVERSION[int(p[0])]
                row = [type, 0, 0] + p[1:].tolist()
                w.writerow(row)

    check_last_line_break(predict_txt)


def check_last_line_break(predict_txt):
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except -1:
        print('bad prediction file format')
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()
