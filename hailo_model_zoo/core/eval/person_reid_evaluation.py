from collections import OrderedDict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY


@EVAL_FACTORY.register(name="person_reid")
class PersonReidEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ["rank1", "rank5", "mAP"]
        self._metrics_vals = [0, 0, 0]
        self.max_rank = 20
        self.reset()

    def _parse_net_output(self, net_output):
        return net_output["predictions"]

    def update_op(self, net_output, img_info):
        for emb, label, type, cam_id in zip(
            net_output["predictions"], img_info["label"], img_info["type"], img_info["cam_id"]
        ):
            if type.decode() == "gallery":
                self.gallery.append([emb, label, cam_id])
            else:
                self.query_list.append([emb, label, cam_id])

    def evaluate(self):
        # Compute distance matrix between gallery and queries
        gallery_emb, gallery_label, gallery_cam_id = map(list, zip(*self.gallery))
        query_emb, query_label, query_cam_id = map(list, zip(*self.query_list))
        gallery_emb = np.stack(gallery_emb)  # (15913, 2048)
        query_emb = np.stack(query_emb)  # (3287, 2048)
        gallery_label = np.stack(gallery_label)  # (15913,)
        query_label = np.stack(query_label)  # (3287, )
        gallery_cam_id = np.asarray(gallery_cam_id)  # (15913,)
        dist_mat = 1 - np.matmul(query_emb, gallery_emb.T)  # (3287, 2048)x(2048, 15913)=(3287, 15913)

        indices = np.argsort(dist_mat, axis=1)  # sort dist_mat to get embeddings similarity in order

        # binary (3287, 15913) - Mij=1 -> gallery label j (dist_mat sorted order) == query label i
        matches = (gallery_label[indices] == query_label[:, np.newaxis]).astype(np.int32)

        """
        ignore embeddings which came from same camera
        (L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, Q. Tian, “Scalable person reidentification:
        A benchmark,” in Proceedings of the IEEE International Conference on Computer Vision, 2015, pp. 1116–1124.)
        """

        all_cmc = []
        all_AP = []
        num_valid_q = 0.0  # number of valid query
        for i in range(0, query_label.shape[0]):
            curr_q_cam_id = query_cam_id[i]
            curr_q_label = query_label[i]
            order = indices[i]
            remove = (gallery_label[order] == curr_q_label) & (
                gallery_cam_id[order] == curr_q_cam_id
            )  # Boolean (15913,)
            keep = np.invert(remove)
            cmc_r = matches[i][keep]  # binary, same as matches, but removed the same cam_id

            if not np.any(cmc_r):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = cmc_r.cumsum()
            cmc[cmc > 1] = 1  # sets 1 starting from where we got a match

            all_cmc.append(cmc[: self.max_rank])
            num_valid_q += 1

            # calc average precision
            pr_h = cmc_r.cumsum()
            pr = [(x) / (i + 1) for i, x in enumerate(pr_h)]  # calc precision points
            recall_d = 1 / cmc_r.sum()  # recall_d = 1/Npositive
            AP = (pr * cmc_r).sum() * recall_d
            all_AP.append(AP)

        all_cmc = np.asarray(all_cmc).astype(np.float32)  # (3287, max_rank=20)
        all_cmc = all_cmc.sum(0) / num_valid_q  # calc the average across queries to get rank1, rank2...
        mAP = np.mean(all_AP)
        self._metrics_vals[0] = all_cmc[0]  # rank1
        self._metrics_vals[1] = all_cmc[4]  # rank5
        self._metrics_vals[2] = mAP  # mAP

    def _get_accuracy(self):
        return OrderedDict(
            [
                (self._metric_names[0], self._metrics_vals[0]),
                (self._metric_names[1], self._metrics_vals[1]),
                (self._metric_names[2], self._metrics_vals[2]),
            ]
        )

    def reset(self):
        self.gallery = []
        self.query_list = []
