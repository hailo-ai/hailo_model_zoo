from hailo_model_zoo.core.datasets import (parse_imagenet, parse_coco, parse_facenet, parse_afw, parse_kitti_depth,
                                           parse_widerface, parse_utkfaces, parse_mot, parse_tusimple, parse_landmarks,
                                           parse_div2k, parse_pascal, parse_kitti_3d, parse_aflw2k3d,
                                           parse_aflw2k3d_tddfa, parse_nyu_depth_v2, parse_300w_lp_tddfa)


def get_dataset_parse_func(ds_name):
    """Get the func to parse dictionary from a .tfrecord.
    Each parse function returns <image, image_info>
        image: image tensor
        image_info: dictionary that contains other information of the image (e.g., the label)
    """
    return {
        'imagenet': parse_imagenet.parse_record,
        'coco_segmentation': parse_coco.parse_segmentation_record,
        'cityscapes': parse_coco.parse_segmentation_record,
        'facenet': parse_facenet.parse_facenet_record,
        'face_landmarks': parse_landmarks.parse_record,
        'kitti_depth': parse_kitti_depth.parse_record,
        'kitti_3d': parse_kitti_3d.parse_record,
        'coco_detection': parse_coco.parse_detection_record,
        'visdrone_detection': parse_coco.parse_detection_record,
        'd2s_detection': parse_coco.parse_detection_record,
        'd2s_fruits_detection': parse_coco.parse_detection_record,
        'coco_2017_detection': parse_coco.parse_detection_record,
        'cocopose': parse_coco.parse_pose_estimation_record,
        'afw': parse_afw.parse_record,
        'widerface': parse_widerface.parse_detection_record,
        'utkfaces': parse_utkfaces.parse_age_gender_record,
        'mot16': parse_mot.parse_mot_record,
        'tusimple': parse_tusimple.parse,
        'div2k': parse_div2k.parse_record,
        'pascal': parse_pascal.parse_record,
        'aflw2k3d': parse_aflw2k3d.parse_record,
        'aflw2k3d_tddfa': parse_aflw2k3d_tddfa.parse_record,
        'nyu_depth_v2': parse_nyu_depth_v2.parse_record,
        'vehicle_detection': parse_coco.parse_detection_record,
        '300w-lp_tddfa': parse_300w_lp_tddfa.parse_record,
        'license_plates': parse_coco.parse_detection_record,
    }[ds_name]
