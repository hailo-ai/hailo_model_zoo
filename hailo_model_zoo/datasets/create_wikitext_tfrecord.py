import argparse

import numpy as np
import tensorflow as tf
import tqdm
from datasets import load_dataset
from transformers import BertModel, BertTokenizer


def _bytes_feature(values):
    if isinstance(values, type(tf.constant(0))):
        values = values.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def create_tf_record(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)
    model = BertModel.from_pretrained("bert-base-uncased")
    count = 0
    with tf.io.TFRecordWriter(args.output) as writer:
        for _, item in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc="Processing examples"):
            text = item["text"]
            if not text.strip():
                continue
            if args.min_length is not None and len(text) < args.min_length:
                continue
            if len(text) > args.pad:
                text = text[: args.pad]
            encoded_data = tokenizer(
                text,
                return_token_type_ids=True,
                return_attention_mask=True,
                max_length=args.pad,
                padding="max_length",
                return_tensors="pt",
            )
            random_idx = tf.random.uniform(
                [], minval=1, maxval=min(sum(encoded_data["attention_mask"][0]) - 1, args.pad - 1), dtype=tf.int32
            ).numpy()
            original_token_id = encoded_data["input_ids"][0][random_idx].item()
            encoded_data["input_ids"][0][random_idx] = tokenizer.mask_token_id
            word_embeddings = model.embeddings.word_embeddings(encoded_data["input_ids"])
            token_type_embeddings = model.embeddings.token_type_embeddings(encoded_data["token_type_ids"])
            model_input = (word_embeddings + token_type_embeddings).detach().numpy().squeeze().astype(np.float32)
            attention_mask = encoded_data["attention_mask"].detach().numpy().squeeze().astype(np.int32)
            text = text.ljust(args.pad)
            assert len(attention_mask) == args.pad
            assert model_input.shape == (args.pad, word_embeddings.shape[-1])
            assert len(text) == args.pad
            feature = {
                "text": _bytes_feature(text.encode("utf-8")),
                "attention_mask": _int64_feature(attention_mask.tolist()),
                "model_input": _float_feature(model_input.flatten().tolist()),
                "channels": _int64_feature([word_embeddings.shape[-1]]),
                "pad": _int64_feature([args.pad]),
                "mask_index": _int64_feature([random_idx]),
                "original_token_id": _int64_feature([original_token_id]),
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            count += 1
            if args.limit is not None and count >= args.limit:
                break
    print(f"Created TFRecord at {args.output} with {count} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="wikitext.tfrecord", help="Output TFRecord file path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (train, test, validation)")
    parser.add_argument(
        "--limit", type=int, default=1024, help="Limit the number of examples to process (default to 1024)"
    )
    parser.add_argument("--pad", type=int, default=512, help="Pad the input_ids to this length (default to 512)")
    parser.add_argument(
        "--min-length", type=int, default=32, help="Minimum length of text to include (default to 100 chars)"
    )
    args = parser.parse_args()
    create_tf_record(args)
