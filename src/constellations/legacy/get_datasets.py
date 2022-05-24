import subprocess
import shlex
from typing import Dict, Iterable, List, Optional, Union
import os
import torch
import datasets
import tensorflow_datasets as tfds
from datasets import concatenate_datasets, load_dataset

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

def get_ml_dataset(data_path: Optional[str]=None,
                   cases: Optional[Dict[str, List[str]]]=None,
                   shuffle: bool=True, 
                   n_parts: int=1, index: int=0) -> datasets.Dataset:
    """Returns the Marvin & Linzen dataset loaded from 
    https://github.com/yoavg/bert-syntax/blob/master/marvin_linzen_dataset.tsv
    Args:
        data_path:   Path to the dataset file from the current working directory.
                     If not provided, the dataset will be downloaded in the current 
                     directory, and loaded from there.
        cases:       A dictionary of cases to load. The keys are the case names, and the
                     values are the lists of subcases. If None, all the cases and subcases
                     are loaded. Specify None as a particular case's value, if all the subcases
                     for that case are to be loaded.
        shuffle:     Whether to shuffle the data before returning. By default, this is True.
        n_parts:     The parts to split the dataset into. By default, this is 1; i.e., no splitting.
        index:       The index of the part to return. By default, this is 0.
    Returns:
        The dataset according to the format specified. The dataset consists of the following fields:
            1. case:           The cond_type of the sample.(str).
            2. subcase:        The subcase of the sample. Tells whether main verb is singular, or plural etc.(str).
            3. correct_sent:   The grammatically correct sentence.(str).
            4. incorrect_sent: The grammatically incorrect sentence.(str).
            5. sentence:       A sentence with [MASK] at exactly one position, constructed.
                               from the correct and incorrect sentences(str).
            6. correct_ans:    The correct word for the [MASK] position, as present in the correct_sent(str).
            7. incorrect_ans:  The incorrect word for the [MASK] position, as present in incorrect_sent(str).
    """
    if data_path is None:
        subprocess.call(shlex.split("git clone https://github.com/yoavg/bert-syntax/"))
        data_path = os.path.join(os.getcwd(), "bert-syntax", "marvin_linzen_dataset.tsv")
    dataset = load_dataset("csv", data_files={"test":"bert-syntax/marvin_linzen_dataset.tsv"}, delimiter="\t", 
                           column_names=["case", "subcase", "correct_sent", "incorrect_sent"])["test"]
    
    def add_masks(correct_sent, incorrect_sent):
        g = correct_sent.split()
        ug = incorrect_sent.split()
        assert (len(g)==len(ug)), (g, ug)
        diffs = [i for i,pair in enumerate(zip(g,ug)) if pair[0]!=pair[1]]
        if (len(diffs)!=1):
            return {"sentence": None, "correct_ans": None, "incorrect_ans": None}
        gv=g[diffs[0]]    # good
        ugv=ug[diffs[0]]  # bad
        g[diffs[0]]="[MASK]"
        g.append(".")
        return {"sentence": " ".join(g), "correct_ans": gv, "incorrect_ans": ugv}
    
    dataset = dataset.map(lambda e: 
                                add_masks(e["correct_sent"], 
                                          e["incorrect_sent"])).filter(
                                                lambda e: e["sentence"] is not None)

    if cases is not None:
        datasets = [dataset.filter(lambda e: (e["case"]==case and 
                                              (subcases is None or e["subcase"] in subcases)))
                    for (case, subcases) in cases.items()]
        dataset = concatenate_datasets(datasets)
    
    if n_parts>1:
        dataset = dataset.shard(n_parts, index)
    
    if shuffle:
        dataset = dataset.shuffle(seed=42)

    return dataset

def get_mlm_data(dataset: str,
                 shuffle: bool=True,
                 n_parts: int=1, index: int=0) -> datasets.Dataset:
    if dataset not in ["bookcorpus"]:
        raise ValueError("Unknown dataset: {}.Use 'bookcorpus'.".format(dataset))
    dataset = load_dataset(dataset, split="train")
    
    if n_parts>1:
        dataset = dataset.shard(n_parts, index)
    
    if shuffle:
        dataset = dataset.shuffle(seed=42)
    
    return dataset

def get_cifar100_dataset(split: str="test",
                         shuffle: bool=True,
                         n_parts: int=1, index: int=0) -> datasets.Dataset:
    
    
    dataset = load_dataset("cifar100", split=split)
    
    if n_parts>1:
        dataset = dataset.shard(n_parts, index)
    
    if shuffle:
        dataset = dataset.shuffle(seed=42)
    
    return dataset

def get_imagenet_dataset(adversarial: Optional[bool]=False,
                         manual_data_dir: Optional[str]=None,
                         split: str="validation",
                         shuffle: bool=True,
                         n_parts: int=1, index: int=0) -> datasets.Dataset:
    if adversarial:
        dataset = tfds.load("imagenet_a", split="test",)
    
    dataset = tfds.load("imagenet2012", split=split, 
                        download_and_prepare_kwargs={'download_dir': manual_data_dir})
    
    if n_parts>1:
        dataset = dataset.shard(n_parts, index)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, seed=42)
    
    return dataset

def get_img_dataset(dataset:str,
                    split: str="test",
                    manual_data_dir: str=None,
                    shuffle: bool=True,
                    n_parts: int=1, index: int=0,) -> datasets.Dataset:
    
    if dataset=="cifar100":
        return get_cifar100_dataset(split, shuffle, n_parts, index)
    elif dataset=="imagenet2012":
        return get_imagenet_dataset(manual_data_dir=manual_data_dir, 
                                    split=split, shuffle=shuffle, 
                                    n_parts=n_parts, index=index)
    elif dataset=="imagenet-a":
        if split!="test":
            raise ValueError(f"Only test split is available for imagenet-a, \
                but {split} split was specified. ")
        
        return get_imagenet_dataset(adversarial=True, shuffle=shuffle,
                                    n_parts=n_parts, index=index)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Only cifar100, \
            or imagenet2012 are supported.")
    
class ECE():
    """Class in with similar interface as datasets.Metric(viz., add_batch() and compute()).
    
    This class is used to compute Expected Calibration Error as described in equation [3]
    of the paper: "On Calibration of Modern Neural Networks" by Chuan Guo and Geoff Pleiss 
    and Yu Sun and Kilian Q. Weinberger.
    """
    def __init__(self, n_bins: int):
        self.samples_per_bin = [0] * n_bins
        self.n_bins = n_bins
        self.samples_per_bin = [0] * n_bins
        self.tot_probs = [0] * n_bins
        self.correct_samples = [0] * n_bins
    
    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor) -> None:
        """Adds a batch of predictions and references for calculating the ECE metric.
        Args:
            predictions:    (batch_size, n_classes) probability of each class.
            references:     (batch_size, 1) ground truth label.
        """
        labels = torch.max(predictions, dim=-1).indices
        
        predictions = predictions.detach().cpu().numpy()
        references = references.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        if not (labels.shape[0]==predictions.shape[0]==references.shape[0]):
            raise AssertionError("Number of predictions, references and labels must be equal.")
        
        for label, probs, ref in zip(labels, predictions, references):
            pred_label_prob = probs[label]
            bin_no = int(pred_label_prob*self.n_bins)-(1 if pred_label_prob==1.0 else 0)
            self.samples_per_bin[bin_no] += 1
            self.correct_samples[bin_no] += (1 if label==ref else 0)
            self.tot_probs[bin_no] += pred_label_prob
        
    def compute(self) -> Dict[str, int]:
        """Returns a dict containing the ECE score and accuracy for all the
        predictions and references added till the call to this function.
    
        NOTE: This function resets the internal state of the object.
        """
        confidences = [0 if n_samples_in_bin==0 else tot_prob_in_bin/n_samples_in_bin 
                        for tot_prob_in_bin, n_samples_in_bin in 
                        zip(self.tot_probs, self.samples_per_bin) ]
        
        accuracies = [0 if n_samples_in_bin==0 else correct_samples_in_bin/n_samples_in_bin
                        for correct_samples_in_bin, n_samples_in_bin in
                        zip(self.correct_samples, self.samples_per_bin) ]
        
        total_samples = sum(self.samples_per_bin)

        ece = sum([n_samples_in_bin*abs(acc-conf) for (n_samples_in_bin, acc, conf) 
                    in zip(self.samples_per_bin, accuracies, confidences)])/total_samples
        
        acc = sum(self.correct_samples)/total_samples

        self.__init__(self.n_bins)
        
        return {"ECE": ece, "accuracy": acc}


from datasets import load_metric as orig_load_metric
def load_metric(metric_name, *args, **kwargs) -> Union[datasets.Metric, ECE]:
    """Wrapper for datasets.load_metric, to provide unified interface
    to load ECE as well as other metrics.
    
    Use ``from get_datasets import load_metric`` instead of ``from datasets import load_metric``, 
    in  all places where you need to load a metric.
    """
    if metric_name=="ECE":
        return ECE(*args, **kwargs)
    
    return orig_load_metric(metric_name, *args, **kwargs)