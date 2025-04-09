from collections import namedtuple

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY

EvalResult = namedtuple("EvalResult", ["value", "name", "is_percentage", "is_bigger_better"])


def topk(array, K, *, axis=-1, sort_output=True):
    if array.shape[axis] <= K:
        assert sort_output
        index_array = np.argsort(-array, axis=axis)
        return np.take_along_axis(array, index_array, axis=axis), index_array
    index_array = np.argpartition(-array, K, axis=axis)
    index_array = np.take(index_array, np.arange(K), axis=axis)
    result = np.take_along_axis(array, index_array, axis=axis)
    if sort_output:
        sorted_index_array = np.argsort(-result, axis=axis)
        result = np.take_along_axis(result, sorted_index_array, axis=axis)
        index_array = np.take_along_axis(index_array, sorted_index_array, axis=axis)
    return index_array


def softmax(x, axis=-1):
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=axis), axis=axis)


@EVAL_FACTORY.register(name="text_encoder")
class TextRetrievalEval(Eval):
    def __init__(self, **kwargs):
        self.reset()

    def _parse_net_output(self, net_output):
        return net_output["predictions"]

    def update_op(self, net_output, gt_labels):
        net_output = self._parse_net_output(net_output)
        self.queries.append(net_output)
        self.corpus.append(gt_labels["image_embeds"])
        self.gt.append(gt_labels["text_embeds"])

    def evaluate(self):
        corpus = np.concatenate(self.corpus)
        queries = np.concatenate(self.queries)
        gt_queries = np.concatenate(self.gt)
        gt_queries = gt_queries.squeeze((1, 2))

        corpus = corpus.squeeze((1, 2))
        queries = queries.squeeze((1,))

        corpus_size = len(corpus)
        assert len(corpus) == len(queries)

        # cosine similarity
        logit_scale = 100
        logits_per_text = queries @ corpus.T
        probabilities = softmax(logits_per_text * logit_scale, axis=1)

        # We assume the correct image is its index in the corpus
        target = np.eye(corpus_size)
        top10 = topk(probabilities, 10, axis=-1)
        retrieval_at_10 = np.take_along_axis(target, top10, axis=-1).sum() / target.sum()
        self.retrieval10 = retrieval_at_10
        self.cosine_similarity = np.mean(np.diag(logits_per_text))

    def _get_accuracy(self):
        pass

    def get_accuracy(self):
        return [
            EvalResult(value=self.retrieval10, name="Retrieval@10", is_percentage=True, is_bigger_better=True),
            EvalResult(
                value=self.cosine_similarity, name="cosine_similarity", is_percentage=False, is_bigger_better=True
            ),
        ]

    def reset(self):
        self.gt = []
        self.retrieval10 = 0
        self.cosine_similarity = 0
        self.corpus = []
        self.queries = []
