import argparse
import os
import tarfile

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.utils import downloader, path_resolver

TF_RECORD_TYPE = "0_shot", "5_shot"
TF_RECORD_LOC = {
    "0_shot": "models_files/mmlu/2024-04-02/mmlu_test_0_shot.tfrecord",
    "5_shot": "models_files/mmlu/2024-04-02/mmlu_test_5_shot.tfrecord",
}

# Paths
MMLU_PATH = "/local/adk_models_files/models_files/mmlu/data/"

TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "college_medicine",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
        "global_facts",
        "clinical_knowledge",
    ],
    "social": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}
SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
choices = ["A", "B", "C", "D"]


def format_example(line, include_answer=True):
    example = "Question: " + line["question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\nAnswer: " + line["answer"] + "\n\n"
    else:
        example += "\nAnswer:"
    return example


def generate_few_shot_prompt(k, subject, dev_df):
    def format_subject(subject):
        ll = subject.split("_")
        s = ""
        for entry in ll:
            s += " " + entry
        return s.strip()

    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))

    if k == -1:
        k = dev_df.shape[0]
    for i in range(k):
        prompt += format_example(
            dev_df.iloc[i, :],
            include_answer=True,
        )
    return prompt


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _create_tfrecord(dataset_dir, k, max_question_length):
    question_num = 0
    folder_path = path_resolver.resolve_data_path("models_files/mmlu/2024-04-02/")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    tf_file_name = f"models_files/mmlu/2024-04-02/mmlu_test_{k}_shot.tfrecord"
    tf_file_name = path_resolver.resolve_data_path(tf_file_name)
    with tf.io.TFRecordWriter(str(tf_file_name)) as writer:
        # loop subjects
        for subject_name in tqdm(SUBJECTS):
            dev_file_path = os.path.join(dataset_dir, "data", "dev", f"{subject_name}_dev.csv")
            test_file_path = os.path.join(dataset_dir, "data", "test", f"{subject_name}_test.csv")
            dev_df = pd.read_csv(dev_file_path, names=["question", "A", "B", "C", "D", "answer"])
            test_df = pd.read_csv(test_file_path, names=["question", "A", "B", "C", "D", "answer"])

            few_shot_prompt = generate_few_shot_prompt(k, subject_name, dev_df) if type else []

            idx_list = list(range(0, len(test_df)))
            for i in tqdm(idx_list):
                row = test_df.iloc[i].to_dict()
                question = format_example(row, include_answer=False)
                full_prompt = few_shot_prompt + question
                answer_gt = row["answer"]

                if len(full_prompt) > max_question_length:
                    continue

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "prompt": _bytes_feature(str.encode(full_prompt)),
                            "answer_gt": _bytes_feature(str.encode(answer_gt)),
                        }
                    )
                )

                if example is not None:
                    writer.write(example.SerializeToString())
                    question_num += 1

    return question_num


def download_dataset():
    dataset_dir = path_resolver.resolve_data_path("models_files/mmlu")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    filename = downloader.download_file("https://people.eecs.berkeley.edu/~hendrycks/data.tar")
    with tarfile.open(filename) as tar:
        tar.extractall(str(dataset_dir))
    return dataset_dir


def run(k, dataset_dir, max_question_length):
    if dataset_dir == "":
        dataset_dir = download_dataset()
    question_num = _create_tfrecord(dataset_dir, k, max_question_length)

    print("Done converting questions {}".format(question_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-shot", help="which tf-record to create {}".format(TF_RECORD_TYPE), default=0, type=int)
    parser.add_argument("--data-path", help="MMLU data path directory", type=str, default="")
    parser.add_argument("--max-question-length", type=int, default=None, help="Limit question length to fit har model")
    args = parser.parse_args()
    max_question_length = args.max_question_length if args.max_question_length is not None else 512
    run(args.k_shot, args.data_path, max_question_length)
