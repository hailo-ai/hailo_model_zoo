import copy

import motmetrics as mm
import numpy as np

mm.lap.default_solver = "scipy.optimize"


class Evaluator(object):
    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, gt_tlwhs, gt_ids, ignore_tlwhs, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = linear_sum_assignment(iou_distance)
            match_is, match_js = (np.asarray(a, dtype=int) for a in [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, "last_mot_events"):
            events = self.acc.last_mot_events
        else:
            events = None
        return events

    @staticmethod
    def get_summary(accs, names, metrics=("mota", "num_switches", "idp", "idr", "idf1", "precision", "recall")):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(accs, metrics=metrics, names=names, generate_overall=True)

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd

        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()
