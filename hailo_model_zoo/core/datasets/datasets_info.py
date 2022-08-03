from enum import Enum

CLASS_NAMES_COCO = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

CLASS_NAMES_D2S = ('adelholzener_alpenquelle_classic_075', 'adelholzener_alpenquelle_naturell_075',
                   'adelholzener_classic_bio_apfelschorle_02', 'adelholzener_classic_naturell_02',
                   'adelholzener_gourmet_mineralwasser_02', 'augustiner_lagerbraeu_hell_05',
                   'augustiner_weissbier_05', 'coca_cola_05', 'coca_cola_light_05',
                   'suntory_gokuri_lemonade', 'tegernseer_hell_03', 'corny_nussvoll',
                   'corny_nussvoll_single', 'corny_schoko_banane', 'corny_schoko_banane_single',
                   'dr_oetker_vitalis_knuspermuesli_klassisch', 'koelln_muesli_fruechte',
                   'koelln_muesli_schoko', 'caona_cocoa', 'cocoba_cocoa', 'cafe_wunderbar_espresso',
                   'douwe_egberts_professional_ground_coffee', 'gepa_bio_caffe_crema',
                   'gepa_italienischer_bio_espresso', 'apple_braeburn_bundle', 'apple_golden_delicious',
                   'apple_granny_smith', 'apple_red_boskoop', 'avocado', 'banana_bundle', 'banana_single',
                   'grapes_green_sugraone_seedless', 'grapes_sweet_celebration_seedless', 'kiwi',
                   'orange_single', 'oranges', 'pear', 'clementine', 'clementine_single',
                   'pasta_reggia_elicoidali', 'pasta_reggia_fusilli', 'pasta_reggia_spaghetti',
                   'franken_tafelreiniger', 'pelikan_tintenpatrone_canon', 'ethiquable_gruener_tee_ceylon',
                   'gepa_bio_und_fair_fencheltee', 'gepa_bio_und_fair_kamillentee',
                   'gepa_bio_und_fair_kraeuterteemischung', 'gepa_bio_und_fair_pfefferminztee',
                   'gepa_bio_und_fair_rooibostee', 'kilimanjaro_tea_earl_grey', 'cucumber', 'carrot',
                   'corn_salad', 'lettuce', 'vine_tomatoes', 'roma_vine_tomatoes', 'rocket',
                   'salad_iceberg', 'zucchini')

D2S_LABEL_MAP = dict([(i + 1, i + 1) for i in range(len(CLASS_NAMES_D2S))])

CLASS_NAMES_D2S_FRUITS = ('apple', 'avocado',
                          'banana_single',
                          'clementine_single',
                          'kiwi',
                          'orange_single', 'pear',
                          'cucumber', 'carrot')

D2S_FRUITS_LABEL_MAP = dict([(i + 1, i + 1) for i in range(len(CLASS_NAMES_D2S_FRUITS))])


class DatasetInfo(object):
    def __init__(self, class_names, label_map):
        self._class_names = class_names
        self._label_map = label_map

    @property
    def class_names(self):
        return self._class_names

    @property
    def label_map(self):
        return self._label_map


class BasicDatasetsEnum(Enum):
    COCO = 'coco_detection'
    D2S = 'd2s_detection'
    D2S_FRUITS = 'd2s_fruits_detection'


DATASETS_INFO = {
    BasicDatasetsEnum.COCO.value: DatasetInfo(class_names=CLASS_NAMES_COCO, label_map=COCO_LABEL_MAP),
    BasicDatasetsEnum.D2S.value: DatasetInfo(class_names=CLASS_NAMES_D2S, label_map=D2S_LABEL_MAP),
    BasicDatasetsEnum.D2S_FRUITS.value: DatasetInfo(class_names=CLASS_NAMES_D2S_FRUITS, label_map=D2S_FRUITS_LABEL_MAP),
}


def get_dataset_info(dataset_name):
    if dataset_name not in DATASETS_INFO:
        raise ValueError('ERROR unknown network_selection {}'.format(dataset_name))
    return DATASETS_INFO[dataset_name]
