from collections import OrderedDict

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY

ACCEPTED_AGE_DELTA = 5
ADIENCE_AGE_LIST = [3.0, 7.0, 13.5, 22.5, 35.0, 45.5, 56.5]


def _get_age_range(age):
    for i, _range_min_age in enumerate(ADIENCE_AGE_LIST):
        if age <= ADIENCE_AGE_LIST[i]:
            return i
    return len(ADIENCE_AGE_LIST)


@EVAL_FACTORY.register(name="age_gender")
class AgeGenderEval(Eval):
    """
    Age/Gender estimation evaluation metrics class.
    """

    def __init__(self, **kwargs):
        super(AgeGenderEval, self).__init__()
        self._gender_correct_num = 0
        self._age_correct_num = 0
        self._age_adience_correct_num = 0
        self._age_mae = 0
        self._gender_accuracy = 0
        self._age_accuracy = 0
        self._age_adience_accuracy = 0
        self._num_images = 0

    def reset(self):
        self._gender_correct_num = 0
        self._age_correct_num = 0
        self._age_adience_correct_num = 0
        self._age_mae = 0

    def update_op(self, net_output, gt_labels):
        for i in range(gt_labels["age"].shape[0]):
            real_age = gt_labels["age"][i]
            is_female = gt_labels["is_female_int"][i] == 1

            predicted_age = net_output["age"][i]
            predicted_is_male = net_output["is_male"][i][0]
            if predicted_is_male != is_female:
                self._gender_correct_num += 1

            real_age_range = _get_age_range(real_age)
            age_range = _get_age_range(predicted_age)
            if real_age_range == age_range:
                self._age_adience_correct_num += 1
            if abs(predicted_age - real_age) <= ACCEPTED_AGE_DELTA:
                self._age_correct_num += 1
            self._age_mae += abs(predicted_age - real_age)

            self._num_images += 1

        return 0

    def evaluate(self):
        """
        Evaluates with detections from all images in our data set with widerface API.
        """
        self._gender_accuracy = self._gender_correct_num / self._num_images
        self._age_accuracy = self._age_correct_num / self._num_images
        self._age_adience_accuracy = self._age_adience_correct_num / self._num_images

    def _get_accuracy(self):
        _num_images = self._num_images if self._num_images else 1
        return OrderedDict(
            (
                ("Gender accuracy", self._gender_accuracy),
                ("Age MAE", self._age_mae / _num_images),
                ("Age accuracy", self._age_accuracy),
                ("Age adience accuracy", self._age_adience_accuracy),
            )
        )

    def is_percentage(self):
        return True
