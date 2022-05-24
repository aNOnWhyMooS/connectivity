import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .get_datasets import get_hans_data, get_mnli_data, get_cola_data, get_qqp_data, get_paws_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loader(args,
               tokenizer,
               mnli_label_dict: Optional[Dict[str, int]]=None,
               heuristic_wise: Optional[List[str]] = ["lexical_overlap"],
               onlyNonEntailing: bool = True,
               n_parts: int=1, index:int=0) -> Iterable[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
    """Returns loader for MNLI/HANS datasets
    Args:
        args:   Namespace object containing the following fields:
                [dataset, split, base_model, batch_size, num_exs]. If num_exs is not provided,
                entire dataset split is returned. Providing all other fields is compulsory.
        
        mnli_label_dict:  A dictionary specifying which MNLI label is to be represented as which integer.
                          Passed to get_mnli_data(), if args.dataset=="mnli"
        heuristic_wise:   The heuristics to be used for HANS dataset. Used only if args.dataset=="hans".
        onlyNonEntailing: If True, only non-entailing samples of HANS are returned. Used only if 
                          args.dataset=="hans".
        n_parts:          The number of parts to break the dataset into. By default whole dataset is returned.
        index:            The index of the part of the dataset to return. By default the first part is returned.
    Returns:
        A loader that returns tuples of (input, target) where input is a dictionary containing
        input_ids, attention_mask, token_type_ids, of shape (batch_size, max_sequence_length) 
        and target is a torch.Tensor of shape (batch_size,) specifying the target label for each
        example in the batch. All returned tensors are on GPU, if it is available.
    """
    if args.dataset == 'mnli':
        dataset = get_mnli_data(split=args.split, labels_dict=mnli_label_dict, n_parts=n_parts, index=index)
    elif args.dataset == 'hans':
        dataset = get_hans_data(split=args.split, onlyNonEntailing=onlyNonEntailing, 
                                heuristic_wise=heuristic_wise, n_parts=n_parts, index=index)
    elif args.dataset=="cola":
        dataset = get_cola_data(split=args.split, n_parts=n_parts, index=index)
    elif args.dataset=="qqp":
        dataset = get_qqp_data(split=args.split, n_parts=n_parts, index=index)
    elif args.dataset=="paws":
        dataset = get_paws_data(data_dir=args.paws_data_dir, split=args.split, n_parts=n_parts, index=index)
    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))
    
    if not hasattr(args, "num_exs") or args.num_exs is None:
        num_exs = len(dataset)
        warnings.warn(f"Taking dataset of length: {num_exs} for {args.dataset}, {args.split} split.")
    else:
        num_exs = args.num_exs
    
    dataset = dataset.select(list(range(0, num_exs)))

    if args.dataset=="cola":
        dataset = dataset.map(lambda e: tokenizer(e["sentence"], truncation=True, 
                                                  padding="max_length", 
                                                  return_tensors="pt",))
    elif args.dataset=="qqp":
        dataset = dataset.map(lambda e: tokenizer(e['question1'], e['question2'],
                                                truncation=True, padding='max_length', 
                                                return_tensors='pt',),)
    elif args.dataset=="paws":
        dataset = dataset.map(lambda e: tokenizer(e['sentence1'], e['sentence2'],
                                                truncation=True, padding='max_length', 
                                                return_tensors='pt',),)
    else:
        dataset = dataset.map(lambda e: tokenizer(e['premise'], e['hypothesis'], 
                                                truncation=True, padding='max_length', 
                                                return_tensors='pt',),)
    
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 
                                              'token_type_ids', 'label'])
    
    dataset = dataset.map(lambda e1, e2, e3: {'input_ids': e1[0],
                                              'attention_mask': e2[0],
                                              'token_type_ids': e3[0]},
                          input_columns=['input_ids', 'attention_mask',
                                         'token_type_ids'])
    
    loader = DataLoader(dataset, batch_size=args.batch_size)
    
    class loader_wrapper:
        def __init__(self, loader):
            self.hf_loader = loader

        def __iter__(self):
            """Send samples to GPU and yield (input, target) tuples."""
            for elem in self.hf_loader:
                for k in elem:
                    elem[k] = elem[k].to(device)
                target = elem.pop("label")
                yield elem, target
    
    input_target_loader = loader_wrapper(loader)
    return input_target_loader


