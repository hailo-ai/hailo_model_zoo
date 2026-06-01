import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from hailo_model_zoo.utils import path_resolver

CORPUS_EMBEDDINGS_NPY = "models_files/all_minilm_l6_v2/mteb_arguana_corpus_embeddings.npy"
TF_RECORD_PATH = "models_files/all_minilm_l6_v2/mteb_arguana_val.tfrecord"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _bytes_feature(values):
    if isinstance(values, type(tf.constant(0))):
        values = values.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _mean_pooling(last_hidden_state, attention_mask):
    """Mean pooling over non-padding tokens. Returns (batch, hidden_size)."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def _load_mteb_arguana():
    """Load the mteb/arguana dataset, returning the default, queries, and corpus subsets separately."""
    default = load_dataset("mteb/arguana", "default")
    queries = load_dataset("mteb/arguana", "queries")
    corpus = load_dataset("mteb/arguana", "corpus")
    return default, queries, corpus


def save_corpus_embeddings(args):
    """Load the corpus from mteb/arguana, encode each entry, and save normalized embeddings as .npy."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    _default, _queries, corpus = _load_mteb_arguana()

    out_path = args.embeddings_output
    if not out_path:
        out_path = path_resolver.resolve_data_path(CORPUS_EMBEDDINGS_NPY)
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = path_resolver.resolve_data_path(str(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings_list = []
    with torch.no_grad():
        for entry in tqdm.tqdm(corpus["corpus"], desc="Encoding corpus entries"):
            text = entry["title"] + " " + entry["text"]
            encoded = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            out = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            pooled = _mean_pooling(out.last_hidden_state, encoded["attention_mask"])
            embeddings_list.append(pooled.squeeze(0).numpy())

    embeddings = np.stack(embeddings_list, axis=0).astype(np.float32)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    embeddings = embeddings / norm

    np.save(str(out_path), embeddings)
    print(f"Saved corpus embeddings to {out_path} with shape {embeddings.shape}")

    # Save corpus IDs in the same order as embeddings for evaluation
    corpus_ids = [entry["_id"] for entry in corpus["corpus"]]
    ids_path = out_path.with_name(out_path.stem + "_ids.npy")
    np.save(str(ids_path), np.array(corpus_ids, dtype=object))
    print(f"Saved corpus IDs to {ids_path} with {len(corpus_ids)} entries")


def create_tf_record(args):
    """Create a TFRecord from the mteb/arguana dataset.

    Each record contains query id, query text, matched document id, and document text.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    default, queries, corpus = _load_mteb_arguana()

    # Build corpus lookup by _id
    corpus_lookup = {entry["_id"]: entry["text"] for entry in corpus["corpus"]}

    # Build relevance mapping from the default test split
    # The default config contains train/test/validation splits with qrels (relevance judgments).
    test_split = default["test"]
    qrel_map = {}  # query_id -> list of (doc_id, score)
    for row in test_split:
        qid = row["query-id"]
        did = row["corpus-id"]
        score = row.get("score", 1)
        if qid not in qrel_map:
            qrel_map[qid] = []
        qrel_map[qid].append((did, score))

    # Build query lookup
    query_lookup = {entry["_id"]: entry["text"] for entry in queries["queries"]}

    output_path = args.output
    if not output_path:
        output_path = TF_RECORD_PATH
    if not Path(output_path).is_absolute():
        output_path = path_resolver.resolve_data_path(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for query_id, doc_list in tqdm.tqdm(qrel_map.items(), desc="Processing query-doc pairs"):
            query_text = query_lookup.get(query_id, "")
            if not query_text.strip():
                skipped += 1
                continue

            for doc_id, _score in doc_list:
                doc_text = corpus_lookup.get(doc_id, "")
                if not doc_text.strip():
                    skipped += 1
                    continue

                if args.min_length is not None and len(query_text) < args.min_length:
                    skipped += 1
                    continue

                # Tokenize query text for model input
                query_truncated = query_text[: args.pad] if len(query_text) > args.pad else query_text
                encoded_data = tokenizer(
                    query_truncated,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    max_length=args.pad,
                    padding="max_length",
                    return_tensors="pt",
                )
                valid_len = sum(encoded_data["attention_mask"][0]).item()
                if valid_len < 2:
                    skipped += 1
                    continue

                word_embeddings = model.embeddings.word_embeddings(encoded_data["input_ids"])
                model_input = word_embeddings.detach().numpy().squeeze().astype(np.float32)
                attention_mask = encoded_data["attention_mask"].detach().numpy().squeeze().astype(np.int32)
                multiplicative_mask = attention_mask.squeeze().copy()

                padded_query_text = query_truncated.ljust(args.pad)

                assert len(attention_mask) == args.pad
                assert model_input.shape == (args.pad, word_embeddings.shape[-1])

                feature = {
                    "query_id": _bytes_feature(query_id.encode("utf-8")),
                    "query_text": _bytes_feature(padded_query_text.encode("utf-8")),
                    "doc_id": _bytes_feature(doc_id.encode("utf-8")),
                    "doc_text": _bytes_feature(doc_text.encode("utf-8")),
                    "multiplicative_mask": _int64_feature(multiplicative_mask),
                    "model_input": _float_feature(model_input.flatten().tolist()),
                    "channels": _int64_feature([word_embeddings.shape[-1]]),
                    "pad": _int64_feature([args.pad]),
                }
                tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())
                count += 1

                if args.limit is not None and count >= args.limit:
                    break
            if args.limit is not None and count >= args.limit:
                break

    if skipped:
        print(f"\n[DEBUG] Skipped {skipped} entries (empty text or below min length)")
    print(f"Created TFRecord at {output_path} with {count} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create TFRecord from mteb/arguana HuggingFace dataset")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output TFRecord path (default: resolved path under models_files/all_minilm_l6_v2)",
    )
    parser.add_argument(
        "--limit", type=int, default=2048, help="Limit the number of examples to process (default: 2048)"
    )
    parser.add_argument("--pad", type=int, default=128, help="Pad the input_ids to this length (default: 128)")
    parser.add_argument(
        "--min-length", type=int, default=2, help="Minimum length of query text to include (default: 2 chars)"
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Generate and save corpus embeddings .npy file",
    )
    parser.add_argument(
        "--embeddings-output",
        type=str,
        default=None,
        help=f"Output path for .npy when using --save-embeddings (default: {CORPUS_EMBEDDINGS_NPY})",
    )
    args = parser.parse_args()

    if args.save_embeddings:
        save_corpus_embeddings(args)
    else:
        create_tf_record(args)
