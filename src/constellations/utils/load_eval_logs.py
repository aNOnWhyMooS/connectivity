import json
import math, re
from multiprocessing.sharedctypes import Value
import os, warnings

from functools import reduce, partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

import datasets
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from multiprocessing import Pool


def read_hans_data():
    dataset = load_dataset("hans", split="validation")
    return dataset


def def_make_dict_key():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def make_dict_key(*args) -> str:
        return tokenizer.decode(tokenizer.encode(*args))

    return make_dict_key


make_dict_key = def_make_dict_key()


def convert_dataset_to_dict(dataset: datasets.Dataset) -> Dict[str, Dict[str, Any]]:
    new_dataset = {}

    for sample in dataset:
        if "premise" in sample:
            key = make_dict_key(sample["premise"], sample["hypothesis"])
            new_dataset[key] = {
                k: sample[k] for k in sample if k not in ["premise", "hypothesis"]
            }
        elif "sentence" in sample:
            key = make_dict_key(sample["sentence"])
            new_dataset[key] = {k: sample[k] for k in sample if k not in ["sentence"]}
        else:
            raise NotImplementedError(
                f"Unknown sample type. Unable to find \
                'sentence' or 'premise'/'hypothesis' keys. {sample}"
            )
    return new_dataset


hans_dataset = read_hans_data()
hans_data_dict = convert_dataset_to_dict(hans_dataset)


def read_file(filename: str) -> Dict:
    with open(filename) as f:
        file_contents = f.read()

    prediction_dict = json.loads(file_contents)

    return prediction_dict


def adjust_labels(
    dataset: datasets.Dataset,
    data_dict: Dict[str, Dict[str, Any]],
    data_dict_labels: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    data_classes = dataset.info.features["label"].names

    original_labels = {elem: i for i, elem in enumerate(data_classes)}

    remap_dict = {original_labels[k]: data_dict_labels[k] for k in data_dict_labels}

    for sent_pair, labels_dict in data_dict.items():
        labels_dict["label"] = remap_dict[labels_dict["label"]]
        data_dict[sent_pair] = labels_dict

    return data_dict


def get_select_heuristic(
    hans_dataset: Dict[str, Dict[str, Any]],
    heuristic: str,
    onlyNonEntailing: bool = False,
    entailment_idx: int = None,
) -> Callable:
    def select_heuristic(sample_dict):
        sample_attrs = hans_dataset[sample_dict["sentence"]]
        label, sample_heuristic = sample_attrs["label"], sample_attrs["heuristic"]

        if label != sample_dict["label"]:
            raise AssertionError(
                f"Found different labels: {label} and {sample_dict['label']}"
            )

        if onlyNonEntailing:
            if label == entailment_idx:
                return False
        else:
            if label != entailment_idx:
                return False

        return heuristic == sample_heuristic

    return select_heuristic


def add_losses(prediction_dict: Dict[str, Any]) -> Dict[str, Any]:
    """prediction_dict corresponds to logits, probabilities, label,\
        premise, hypothesis etc. of a single sample."""
    correct_label = prediction_dict["label"]
    probs = prediction_dict["probabilities"]

    correct_label_prob = probs[correct_label]

    prediction_dict["loss"] = -math.log(correct_label_prob)

    prediction_dict["correct"] = correct_label == probs.index(max(probs))
    return prediction_dict


def convert_pred_list_to_dict(
    pred_list: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """To convert the list of prediction_dicts from logs to \
       a dictionary"""

    new_preds_dict = {}
    for pred_dict in pred_list:
        new_preds_dict[pred_dict["sentence"]] = pred_dict
    return new_preds_dict


def get_avg_metrics(predictions_list, glue_subset="mnli"):
    total_loss = reduce(
        lambda carry, sample: carry + sample["loss"], predictions_list, 0
    )
    avg_loss = total_loss / len(predictions_list)

    metric = load_metric("glue", glue_subset)
    for sample in predictions_list:
        metric.add(
            prediction=sample["probabilities"].index(max(sample["probabilities"])),
            reference=sample["label"],
        )

    metrics_dict = metric.compute()
    metrics_dict["loss"] = avg_loss
    return metrics_dict


def longest_common_prefix(str_lis):
    common_str = str_lis[0]
    for st in str_lis[1:]:
        for k, (i, j) in enumerate(zip(common_str, st)):
            if i != j:
                common_str = common_str[:k]
                break
        else:
            if len(st) < len(common_str):
                common_str = common_str[: len(st)]
    return common_str


def get_nli_model_metrics(
    filename: str, heuristic: Optional[str] = None, entailing: Optional[bool] = None
):
    assert (heuristic is None) == (entailing is None)

    eval_log = read_file(filename)
    entailment_index = eval_log["label_dict"]["entailment"]
    preds = eval_log["samples"]

    if heuristic is not None and entailing is not None:
        preds = list(
            filter(
                lambda pred_dict: (
                    (hans_data_dict[pred_dict["sentence"]]["heuristic"] == heuristic)
                    and (entailing == (pred_dict["label"] == entailment_index))
                ),
                preds,
            )
        )

    return get_avg_metrics(list(map(add_losses, preds)))


def get_cola_model_metrics(
    filename: str,
    split: str,
):
    eval_log = read_file(filename)
    preds = eval_log["samples"]
    if split == "in_domain_dev":
        preds = preds[:527]
    elif split == "out_domain_dev":
        preds = preds[527:]
    else:
        assert split == "validation"

    return get_avg_metrics(list(map(add_losses, preds)), "cola")


def get_qqp_model_metrics(
    filename: str,
):
    eval_log = read_file(filename)
    preds = eval_log["samples"]
    return get_avg_metrics(list(map(add_losses, preds)), "qqp")


def natural_order(lis: List[str]) -> List[str]:
    prefix_len = len(longest_common_prefix(lis))

    def get_key(word):
        suffix = word[prefix_len:]
        digits = re.findall(r"\d+", suffix)
        if len(digits) != 1:
            warnings.warn(
                f"Unable to find one single group of digits in\
                {suffix}. Returning original order: {lis}"
            )
        else:
            return int(digits[0])

    try:
        sorted_lis = sorted(lis, key=get_key)
    except TypeError as e:
        if "'<' not supported between instances of 'NoneType'" in str(e):
            sorted_lis = lis
        else:
            raise e

    assert len(set(sorted_lis)) == len(sorted_lis)

    return sorted_lis


def get_metrics(
    directory: str, heuristic: Optional[str] = None, entailing: Optional[bool] = None
) -> Dict[str, Tuple[float, float]]:
    files = [
        os.path.join(directory, elem)
        for elem in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, elem)) and elem.endswith(".json")
    ]

    prefix_len = len(longest_common_prefix(files))

    models_to_files = {filename[prefix_len:-5]: filename for filename in files}

    assert len(models_to_files) == len(files)

    sorted_model_names = natural_order([filename[prefix_len:-5] for filename in files])

    return {
        model: get_nli_model_metrics(models_to_files[model], heuristic, entailing)
        for model in sorted_model_names
    }


def get_metrics_with_sufs(
    prefix: str,
    suffixes: List[str],
    heuristic: Optional[str] = None,
    entailing: Optional[bool] = None,
    split: Optional[str] = "validation",
    data_type: Optional[str] = "nli",
) -> Dict[str, Tuple[float, float]]:
    files = [
        prefix + suffix + ("" if suffix.endswith(".json") else ".json")
        for suffix in suffixes
    ]

    if data_type == "nli":
        return {
            suffix: get_nli_model_metrics(file, heuristic, entailing)
            for (suffix, file) in zip(suffixes, files)
        }
    elif data_type == "qqp":
        return {
            suffix: get_qqp_model_metrics(file)
            for (suffix, file) in zip(suffixes, files)
        }
    elif data_type == "cola":
        return {
            suffix: get_cola_model_metrics(file, split)
            for (suffix, file) in zip(suffixes, files)
        }
