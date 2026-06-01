"""
Evaluation for MTEB retrieval (corpus-based Recall@10).

SentenceEmbeddingEval:
  Computes Recall@10 for MTEB retrieval tasks. For each query, cosine similarity is
  computed against all corpus embeddings; the metric checks whether the ground-truth
  document appears in the top 10 results.
"""

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.eval.eval_result import EvalResult
from hailo_model_zoo.core.factory import EVAL_FACTORY
from hailo_model_zoo.utils import path_resolver


def _load_corpus_embeddings(embeddings_path):
    """Load corpus embeddings (num_corpus, dim) and companion corpus IDs."""
    embeddings_path = path_resolver.resolve_data_path(str(embeddings_path))
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Corpus embeddings not found at {embeddings_path}.")
    corpus_embeddings = np.load(str(embeddings_path)).astype(np.float32)

    ids_path = embeddings_path.with_name(embeddings_path.stem + "_ids.npy")
    if not ids_path.exists():
        raise FileNotFoundError(
            f"Corpus IDs not found at {ids_path}. "
            "Generate a companion _ids.npy file with corpus document IDs "
            "in the same order as the corpus embeddings."
        )
    corpus_ids = np.load(str(ids_path), allow_pickle=True)
    return corpus_embeddings, corpus_ids


def _topk(array, K, *, axis=-1, sort_output=True):
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


def _softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


@EVAL_FACTORY.register(name="sentence_embedding_generation")
class SentenceEmbeddingEval(Eval):
    """Evaluate MTEB retrieval by Retrieval@10 and cosine similarity, same as TextRetrievalEval."""

    def __init__(self, **kwargs):
        self._corpus_embeddings_path = kwargs.get("corpus_embeddings_path")
        self._corpus_embeddings = None
        self._doc_id_to_corpus_index = None
        self.reset()

    def _ensure_loaded(self):
        if self._corpus_embeddings is None:
            corpus_embeddings, corpus_ids = _load_corpus_embeddings(self._corpus_embeddings_path)
            # L2-normalize for cosine similarity = dot product
            norm = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
            norm = np.where(norm > 0, norm, 1.0)
            self._corpus_embeddings = corpus_embeddings / norm
            self._doc_id_to_corpus_index = {doc_id: i for i, doc_id in enumerate(corpus_ids)}

    def _parse_net_output(self, net_output):
        return np.asarray(net_output["predictions"], dtype=np.float32)

    def update_op(self, net_output, img_info):
        self._ensure_loaded()
        predictions = self._parse_net_output(net_output)
        if predictions.ndim == 3:
            predictions = np.squeeze(predictions, axis=(1, 2))
        elif predictions.ndim == 1:
            predictions = np.expand_dims(predictions, axis=0)

        doc_ids = img_info["doc_id"]
        if isinstance(doc_ids, bytes):
            doc_ids = [doc_ids]
        elif hasattr(doc_ids, "numpy"):
            doc_ids = doc_ids.numpy()
            if isinstance(doc_ids, bytes):
                doc_ids = [doc_ids]

        gt_indices = []
        for doc_id in doc_ids:
            if isinstance(doc_id, bytes):
                doc_id = doc_id.decode("utf-8")
            doc_id = doc_id.strip()
            gt_indices.append(self._doc_id_to_corpus_index[doc_id])

        self._predictions.append(predictions)
        self._gt_corpus_indices.append(np.array(gt_indices, dtype=np.int32))

    def evaluate(self):
        if not self._predictions:
            self._retrieval_at_10 = 0.0
            self._cosine_similarity = 0.0
            return
        predictions = np.concatenate(self._predictions, axis=0)
        gt_corpus_indices = np.concatenate(self._gt_corpus_indices, axis=0)

        # (N, dim) @ (num_corpus, dim).T -> (N, num_corpus); cosine similarity
        logits_per_query = predictions @ self._corpus_embeddings.T
        logits_per_query = logits_per_query.squeeze()

        # cosine similarity
        logit_scale = 100
        probabilities = _softmax(logits_per_query * logit_scale, axis=1)

        # Build target matrix: each row has a 1 at the ground-truth corpus index
        num_queries = len(predictions)
        num_corpus = self._corpus_embeddings.shape[0]
        target = np.zeros((num_queries, num_corpus), dtype=np.float32)
        target[np.arange(num_queries), gt_corpus_indices] = 1.0

        top10 = _topk(probabilities, 10, axis=-1)
        self._retrieval_at_10 = np.take_along_axis(target, top10.squeeze(), axis=-1).sum() / target.sum()
        self._cosine_similarity = logits_per_query[np.arange(num_queries), gt_corpus_indices].mean()

    def _get_accuracy(self):
        return {"Retrieval@10": getattr(self, "_retrieval_at_10", 0.0)}

    def get_accuracy(self):
        return [
            EvalResult(
                value=getattr(self, "_retrieval_at_10", 0.0),
                name="Retrieval@10",
                is_percentage=True,
                is_bigger_better=True,
            ),
            EvalResult(
                value=getattr(self, "_cosine_similarity", 0.0),
                name="cosine_similarity",
                is_percentage=False,
                is_bigger_better=True,
            ),
        ]

    def reset(self):
        self._predictions = []
        self._gt_corpus_indices = []
        self._retrieval_at_10 = 0.0
        self._cosine_similarity = 0.0
