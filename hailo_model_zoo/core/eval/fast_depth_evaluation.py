import math
from collections import OrderedDict
import numpy as np
from hailo_model_zoo.core.eval.eval_base_class import Eval


class FastDepthEval(Eval):
    def __init__(self, **kwargs):
        self.average_meter = AverageMeter()
        self.avg = None
        self.reset()

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def is_percentage(self):
        return False

    def is_bigger_better(self):
        return False

    def update_op(self, net_output, img_info):
        net_output = self._parse_net_output(net_output)
        ground_truth = img_info['depth']
        for i in range(net_output.shape[0]):  # update per inference in batch
            result = Result()
            result.evaluate(net_output[i], ground_truth[i])
            self.average_meter.update(result, 0, 0, 1)

    def evaluate(self):
        self.avg = self.average_meter.average()

    def _get_accuracy(self):
        return OrderedDict([('rmse', self.avg.rmse),
                            ('absrel', self.avg.absrel),
                            ('mse', self.avg.mse),
                            ('lg10', self.avg.lg10),
                            ('mae', self.avg.mae),
                            ('delta1', self.avg.delta1),
                            ('delta2', self.avg.delta2),
                            ('delta3', self.avg.delta3)
                            ])

    def reset(self):
        self.average_meter.reset()
        self.avg = Result()


class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        valid_mask = ((target > 0) + (output > 0)) > 0
        output = np.array(output[valid_mask], dtype=np.float32)
        target = np.array(target[valid_mask], dtype=np.float32)
        abs_diff = np.abs(output - target)

        self.mse = np.mean((np.power(abs_diff, 2)))
        self.rmse = math.sqrt(self.mse)
        self.mae = np.mean(abs_diff)
        self.lg10 = np.mean(np.abs(np.log10(output) - np.log10(target)))
        self.absrel = np.mean(abs_diff / target)

        maxRatio = np.maximum(output / target, target / output)
        self.delta1 = np.mean(maxRatio < 1.25)
        self.delta2 = np.mean(maxRatio < 1.25 ** 2)
        self.delta3 = np.mean(maxRatio < 1.25 ** 3)
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = np.abs(inv_output - inv_target)
        self.irmse = math.sqrt(np.mean(np.power(abs_inv_diff, 2)))
        self.imae = np.mean(abs_inv_diff)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count)
        return avg
