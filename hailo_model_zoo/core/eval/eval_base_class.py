from abc import ABC, abstractmethod

from hailo_model_zoo.core.eval.eval_result import EvalResult


class Eval(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def update_op(self, net_output, img_info):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def _get_accuracy(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def is_percentage(self):
        return True

    def is_bigger_better(self):
        return True

    def get_accuracy(self):
        result_dict = self._get_accuracy()
        return [
            EvalResult(
                value=value, name=key, is_percentage=self.is_percentage(), is_bigger_better=self.is_bigger_better()
            )
            for key, value in result_dict.items()
        ]
