import os
from typing import Dict, Iterable, List, Optional
import datasets
from datasets import load_dataset

def get_cola_data(split: str, n_parts: int=1, index: int=0) ->datasets.Dataset:
    dataset = load_dataset("glue", "cola")

    if split=="in_domain_dev":
        dataset = dataset["validation"].select(list(range(527)))
    elif split=="out_domain_dev":
        dataset = dataset["validation"]
        dataset = dataset.select(list(range(527, len(dataset))))
    elif split=="validation":
        dataset = dataset["validation"]
    elif split=="train":
        dataset = dataset["train"]
    elif split=="test":
        dataset = dataset["test"]
    else:
        raise ValueError(f"Unknown split: {split}. Use one of [in_domain_dev, out_domain_dev,\
             validation(for in_domain+out_domain dev sets), train, test]")
    
    if n_parts>1:
        dataset = dataset.shard(n_parts, index)

    return dataset

def get_qqp_data(split:str, n_parts: int=1, index:int=0) -> datasets.Dataset:
    dataset = load_dataset("glue", "qqp")
    dataset = dataset[split]
    if n_parts>1:
        dataset = dataset.shard(n_parts, index)

    return dataset

def get_paws_data(data_dir: str, split:str, n_parts: int=1, index:int=0) -> datasets.Dataset:
    split_to_locs = {"train": os.path.join(data_dir, "train.tsv"), 
                     "dev_and_test": os.path.join(data_dir, "dev_and_test.tsv")}
    dataset = load_dataset("csv", data_files=split_to_locs, delimiter="\t")
    dataset = dataset[split]
    
    if n_parts>1:
        dataset = dataset.shard(n_parts, index)

    return dataset

def remap_labels(labels: Iterable[int], remap_dict: Dict[int, int]) -> List[int]:
    """Remap the entries of labels according to the remap_dict provided."""
    return [remap_dict[label] for label in labels]

def get_mnli_data(split: str, labels_dict: Optional[Dict[str, int]]=None, 
                  n_parts: int=1, index: int=0) -> datasets.Dataset:
    """Returns mnli dataset. Possibly with re-numbered labels.
    Args:
        split:       "train" or "test"=="validation_matched". The split whose data is required. 
        labels_dict:  dict providing bijective mapping b/w {contradiction, entailment, neutral}
                      and {0,1,2}
        n_parts:      The parts to split the dataset into. By default, this is 1; i.e., no splitting.
        index:        The index of the part to return. By default, this is 0.
    Returns:
        The mnli dataset of the specified split, unshuffled, but re-labelled according to 
        labels_dict.
    """
    if split=="test":
        split = "validation_matched"
    if split not in ["train", "validation_matched"]:
        raise ValueError(f"Unknown split: {split}. Only 'train' and 'test'(=='validation_matched') \
                         are supported currently.")

    dataset = load_dataset("glue", "mnli", split=split)

    if labels_dict is not None:
        for k in labels_dict.keys():
            if k not in ["contradiction", "entailment", "neutral"]:
                raise ValueError("Invalid label: {}, found in label dict. \
                                  Only 'contradiction', 'entailment' and \
                                  'neutral' are valid".format(k))
    
    
        original_labels = {elem:i for i, elem in enumerate(dataset.info.features["label"].names)}
        remap_dict = {original_labels[k]: labels_dict[k] for k in labels_dict}
    
        remapped_labels = remap_labels(dataset["label"], remap_dict)
        dataset = dataset.remove_columns(["label"])
        dataset = dataset.add_column("label", remapped_labels)
    
    if n_parts>1:
        dataset = dataset.shard(n_parts, index)

    return dataset


def get_hans_data(split: str, 
                  heuristic_wise: Optional[List[str]] = None,
                  onlyNonEntailing: bool = False, 
                  shuffle: bool=True,
                  n_parts: int=1, index: int=0) -> datasets.Dataset:
    """Returns HANS dataset, modified according to the other provided arguments.
    Args:
        split:              "train" or "test"=="validation". The split whose data is required.
        heuristic_wise:     A list of heuristics, if specified, only the samples corresponding to 
                            these heuristics will be present in the returned dataset.
                            (in {'constituent', 'lexical_overlap', 'subsequence'})
        onlyNonEntailing:   If True, only, a dataset with only non-entailment samples is returned.
                            By default, it is False.
        shuffle:            Whether to shuffle data before returning. By default, this is True.
        n_parts:            The parts to split the dataset into. By default, this is 1; i.e., no splitting.
        index:              The index of the part to return. By default, this is 0.
        
    NOTE:   Shuffling is done after splitting. 
    
    Returns:
        HANS dataset, modified according to the other provided arguments.
    """
    if split=="test":
        split = "validation"
    
    if split not in ["train", "validation"]:
        raise ValueError(f"Unknown split: {split}. Only 'train' and 'test'(=='validation') \
                         are supported currently.")

    dataset = load_dataset("hans", split=split)
    
    if heuristic_wise is not None:
        available_heuristics = set(dataset["heuristic"])
        if len(set(heuristic_wise).intersection(set(available_heuristics))) < len(heuristic_wise):
            raise ValueError("Invalid element in heuristic_wise. Can't find one or more \
                              heuristics in the HANS dataset.")
        dataset = dataset.filter(lambda e: e["heuristic"] in heuristic_wise)
    
    if onlyNonEntailing:
        non_entailment_idx = dataset.info.features["label"].names.index("non-entailment")
        if non_entailment_idx==-1:
            raise AssertionError("Can't find non-entailment category in HANS dataset.info, check install.")
        
        dataset = dataset.filter(lambda e: e["label"]==non_entailment_idx)
    
    if n_parts>1:
        dataset = dataset.shard(n_parts, index)
    
    if shuffle:
        dataset = dataset.shuffle(seed=42)
    
    return dataset