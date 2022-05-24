from math import sqrt
import functools, os, sys
import shutil
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type
from urllib.error import HTTPError

import scipy
import torch
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
from git import Repo
from huggingface_hub import Repository
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers.modeling_outputs import SequenceClassifierOutput as SCO

from .get_datasets import get_hans_data, get_mnli_data, get_ml_dataset, get_mlm_data, get_img_dataset, get_cola_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_tfds(args, dataset, resolution):
    def pp(sample):
        """Simple image preprocessing."""
        img = tf.cast(sample["image"], float) / 255.0
        img = tf.image.resize(img, [resolution, resolution])
        return img, sample["label"]
    
    dataset = dataset.take(args.num_exs)
    dataset = dataset.map(pp)
    dataset = dataset.padded_batch(args.batch_size, drop_remainder=True)
    return [(images, labels) for (images, labels) in dataset.as_numpy_iterator()]

def get_imgData_loader(args, 
                      resolution,
                      *get_img_dataset_args,
                      **get_img_dataset_kwargs):
    
    dataset = get_img_dataset(dataset=args.dataset, split=args.split, 
                              manual_data_dir=args.manual_data_dir,
                              *get_img_dataset_args, **get_img_dataset_kwargs)
    
    
    if not hasattr(args, "num_exs"):
        num_exs = len(dataset)
        warnings.warn(f"Taking dataset of length: {num_exs} for {args.dataset}, {args.split} split.")
    else:
        num_exs = args.num_exs
    
    if args.dataset=="imagenet2012":
        return process_tfds(args, dataset, resolution)
    
    dataset = dataset.select(list(range(0, num_exs)))
    
    def pp(img, sz):
        """Simple image preprocessing."""
        img = img.getdata()
        img = np.array(img).reshape(img.size[0], img.size[1], 3)
        img = tf.cast(img, float) / 255.0
        img = tf.image.resize(img, [sz, sz])
        return np.array(img).reshape(-1)
    
    if args.dataset=="cifar100":
        label_key = "fine_label"
    
    dataset = dataset.map(lambda e: {"data_image": pp(e["img"], resolution), 
                                     "data_label": e[label_key],}, 
                          remove_columns=dataset.column_names)
    
    dataset.set_format(type="numpy", columns=["data_image", "data_label"])
    
    dl = DataLoader(dataset, batch_size=args.batch_size, 
                    drop_last=True)

    class loader_wrapper():
        def __init__(self, dl):
            self.dl = dl
        
        def __iter__(self):
            for batch in self.dl:
                images = batch["data_image"]
                labels = batch["data_label"]
                images = images.reshape(images.shape[0], 
                                        resolution, resolution, 3)
                yield jnp.array(images), jnp.array(labels)
    
    return loader_wrapper(dl)

def get_mlm_loader(args, *get_mlm_data_args, **get_mlm_data_kwargs):
    """Returns a DataLoader for masked language modeling on bookcorpus dataset.
    Args:
        args:  A namespace object containing the following fields:
                [base_model, batch_size, use_wwm, num_exs]. If num_exs is not provided,
                entire dataset split is returned. Providing other fields is compulsory.
                use_wwm is a boolean indicating whether to use whole-word masking.
        
        get_mlm_dataset_args:  Arguments to be passed to get_mlm_dataset()
        get_ml_dataset_kwargs: Keyword arguments to be passed to get_mlm_dataset()
    
    Returns:
        A loader that returns tuples of (input, labels). where:
        
        1. input is a dictionary containing input_ids(when use wwm), with additional 
           attention_mask, token_type_ids(in case not using wwm). labels included in
           input dictionary too. 
        
        2. labels is a tensor of shape [batch_size, seq_len]. With correct 
           token ids at the masked positions and -100 elsewhere.
    """
    dataset = get_mlm_data("bookcorpus", *get_mlm_data_args, **get_mlm_data_kwargs)
    if not hasattr(args, "num_exs"):
        num_exs = len(dataset)
        warnings.warn(f"Taking dataset of length: {num_exs} for {args.dataset}, {args.split} split.")
    else:
        num_exs = args.num_exs
    
    dataset = dataset.select(list(range(0, num_exs)))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    dataset = dataset.map(lambda e: tokenizer(e["text"], 
                                             return_tensors="pt", 
                                             truncation=True,
                                             return_special_tokens_mask=True),
                                             remove_columns=dataset.column_names)
    
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask',
                                              'token_type_ids', 'special_tokens_mask'])
    
    dataset = dataset.map(lambda e1, e2, e3, e4: {'input_ids': e1[0],
                                                  'attention_mask': e2[0],
                                                  'token_type_ids': e3[0],
                                                  'special_tokens_mask': e4[0],},
                          input_columns=['input_ids', 'attention_mask',
                                         'token_type_ids', 'special_tokens_mask',],)
    
    if args.use_wwm:
        col = DataCollatorForWholeWordMask(tokenizer)
    else:
        col = DataCollatorForLanguageModeling(tokenizer)
    
    class loader_wrapper:
        def __init__(self, dataset, col, b_sz):
            self.dataset = dataset
            self.collator = col
            self.b_sz = b_sz
        
        def __iter__(self):
            """Send samples to GPU and yield (input, target) tuples."""
            batch = []
            for elem in self.dataset:
                batch.append(elem)
                if len(batch)==self.b_sz:
                    collated = col(batch)
                    for k, v in collated.items():
                        collated[k] = v.to(device)
                    labels = collated["labels"]
                    yield collated, labels
                    batch = []
    
    input_target_loader = loader_wrapper(dataset, col, args.batch_size)
    
    return input_target_loader
        
def get_NA_loader(args, *get_ml_dataset_args, **get_ml_dataset_kwargs):
    """Returns the DataLoader for Marvin & Linzen's Number Agreement dataset provided at 
    https://github.com/yoavg/bert-syntax/blob/master/marvin_linzen_dataset.tsv . 
    Args:
        args:  A namespace object containing the following fields:
                [base_model, batch_size, num_exs]. If num_exs is not provided,
                entire dataset split is returned. Providing other fields is compulsory.
        
        get_ml_dataset_args:  Arguments to be passed to get_ml_dataset()
        get_ml_dataset_kwargs: Keyword arguments to be passed to get_ml_dataset()
    
    Returns:
        A loader that returns tuples of (input, correct_token_ids, incorrect_token_ids).
        where input is a dictionary containing input_ids, attention_mask, token_type_ids.
    """
    dataset = get_ml_dataset(*get_ml_dataset_args, **get_ml_dataset_kwargs)
    if not hasattr(args, "num_exs"):
        num_exs = len(dataset)
        warnings.warn(f"Taking dataset of length: {num_exs} for {args.dataset}, {args.split} split.")
    else:
        num_exs = args.num_exs
    
    dataset = dataset.select(list(range(0, num_exs)))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    def get_ids(tokens):
        ids = tokenizer.convert_tokens_to_ids(tokens)
        if ids[0]==tokenizer.unk_token_id and ids[1]==tokenizer.unk_token_id:
            raise ValueError(f"At least one of correct/incorrect answers must \
                be in the vocab. Both {tokens[0]} and {tokens[1]} are not in the vocab.")
        return {"correct_ans_id": ids[0], 
                "incorrect_ans_id": ids[1]}

    dataset = dataset.map(lambda e: get_ids([e["correct_ans"], e["incorrect_ans"]]))

    dataset = dataset.map(lambda e: tokenizer(e["sentence"], truncation=True,
                                              padding="max_length", return_tensors="pt"))
    
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask',
                                              'token_type_ids', 'correct_ans_id', 
                                              'incorrect_ans_id'])
    
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
                correct_ans_ids = elem.pop("correct_ans_id")
                incorrect_ans_ids = elem.pop("incorrect_ans_id")
                yield elem, correct_ans_ids, incorrect_ans_ids
    
    input_target_loader = loader_wrapper(loader)
    
    return input_target_loader

def get_loader(args,
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
    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))
    
    if not hasattr(args, "num_exs") or args.num_exs is None:
        num_exs = len(dataset)
        warnings.warn(f"Taking dataset of length: {num_exs} for {args.dataset}, {args.split} split.")
    else:
        num_exs = args.num_exs
    
    dataset = dataset.select(list(range(0, num_exs)))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    if args.dataset=="cola":
        dataset = dataset.map(lambda e: tokenizer(e["sentence"], truncation=True, 
                                                  padding="max_length", 
                                                  return_tensors="pt",))
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

def get_train_test_loader(args, *get_loader_args, **get_loader_kwargs):
    """
    Args:
        args:  A namespace object to pass to get_loader().
        *get_loader_args:  Arguments to pass to get_loader().
        **get_loader_kwargs:  Keyword arguments to pass to get_loader().
    Returns:
        Train and test data loader for the dataset specified in args.dataset.
    """
    args.split = "train"
    train_loader = get_loader(args, *get_loader_args, **get_loader_kwargs)
    args.split = "test"
    test_loader = get_loader(args, *get_loader_args, **get_loader_kwargs)
    del args.split
    return train_loader, test_loader

def get_train_test_loaders(args, *get_loader_args, **get_loader_kwargs):
    """To get multiple train loaders, and a single test loader.
    Args:
        args:  A namespace object to pass to get_loader().
        *get_loader_args:  Arguments to pass to get_loader().
        **get_loader_kwargs:  Keyword arguments to pass to get_loader().
    Returns:
        Train and test data loader for the dataset specified in args.dataset.
        Train data is broken into args.break_epoch train loaders, and a single 
        test loader with min(len(test_dataset), 8194) examples, are returned.
    """
    args.split = "train"
    train_loaders = []
    
    for i in range(args.break_epochs):
        args.split = "train"
        train_loader = get_loader(args, *get_loader_args, **get_loader_kwargs, 
                                  n_parts=args.break_epochs, index=i)
        train_loaders.append(train_loader)
    
    args.split = "test"
    test_loader = get_loader(args, *get_loader_args, **get_loader_kwargs)
    if len(test_loader.hf_loader.dataset) >= 8194:
        args.num_exs = 8194
        test_loader = get_loader(args, *get_loader_args, **get_loader_kwargs)
        del args.num_exs
    
    del args.split
    return train_loaders, test_loader

def get_logits_converter(mnli_label_dict: Dict[str, int], hans_label_dict: Dict[str, int], pool='max') -> Callable:
    """Returns the functions that converts MNLI logits to HANS.
    Static State Maintained:
            mnli_label_dict:  A dictionary specifying which MNLI label is to be represented as which integer.
            hans_label_dict:  A dictionary specifying which HANS label is to be represented as which integer.
            pool:             "max"/"sum" the pooling strategy to use to convert contradiction and neutral logits
                              to non-entailment logits.
    """

    def mnli_logits_to_hans(model_out: SCO) -> SCO:
        """Convert (sequence classification)logits for mnli finetuned model, to that for 
        binary labels of HANS dataset.
        PRE-CONDITIONS:
            1. Last dimension of model_out["logits"] tensor, spans the 3 mnli classes.
            2. The logit for neutral, contradiction and entailment appear at the locations 
               specified in mnli_label_dict.
        Args:
            model_out:  The output of model.forward() call for MNLI.
            pool:       Method to combine contradiction and neutal logits for HANS.
                        (in {'max', 'sum'})
        Returns:
            The model_out with logit for entailment and non-entailment for HANS sample, specified
            at the corresponding locations in the last dimension read from hans_label_dict.
            
        NOTE: 1. The shape of last dimension of model_out["logits"] gets changed from 3 to 2. 
              2. An additional "format" field is added to model_out, and set to "hans_logits" to indicate
                 the format of logits held by model_out.
        """
        if "format" in model_out and model_out["format"] == "hans_logits":
            return model_out
        
        contradiction_logits =  model_out["logits"][..., mnli_label_dict["contradiction"]]
        neutral_logits = model_out["logits"][..., mnli_label_dict["neutral"]]
        entailment_logits = model_out["logits"][..., mnli_label_dict["entailment"]]
        hans_logits = torch.zeros_like(model_out["logits"])
        if pool == "max":
            hans_logits[..., hans_label_dict["non-entailment"]] = torch.max(contradiction_logits,
                                                                            neutral_logits)
        elif pool == "sum":
            hans_logits[..., hans_label_dict["non-entailment"]] = contradiction_logits+neutral_logits
        else:
            raise ValueError("Unknown pooling {}".format(pool))
        hans_logits[..., hans_label_dict["entailment"]] = entailment_logits
        hans_logits = hans_logits[..., list(hans_label_dict.values())]
        model_out["logits"] = hans_logits
        model_out["format"] = "hans_logits"
        return model_out
    
    return mnli_logits_to_hans

def select_revision(path_or_name, num_steps: int, local_dir=None, tmp_dir=None):
    """Return the latest commit with num_steps in its commit message."""
    import string, random
    
    if num_steps is None:
        return None
    
    if tmp_dir is None:
        tmp_dir = "."+''.join(random.choices(string.ascii_uppercase+string.digits, k=20))
    if local_dir is not None:
        repo = Repo(local_dir)
    else:
        try:
            shutil.rmtree(tmp_dir)
        except FileNotFoundError:
            print(f"Creating {tmp_dir}, for loading in git data")
        repo = Repository(local_dir=tmp_dir, clone_from=path_or_name, skip_lfs_files=True)
        repo = Repo(tmp_dir)
    
    for commit in repo.iter_commits("main"):
        if str(num_steps) in commit.message:
            selected_commit=str(commit)
            break
    else:
        raise ValueError(f"Unable to find any commit with {num_steps} steps")
    
    return selected_commit

def get_model(path_or_name,
              base_model: Optional[str]="bert-base-uncased",
              from_flax: bool = False,
              model_type: Type[AutoModel]=AutoModel,
              **select_revision_kwargs) -> torch.nn.Module:
    """Returns a sequence classification model loaded from path_or_name. If it can't load 
    the model from path_or_name, it treats the path_or_name as a state dict file, and tries 
    to load it using torch.load() after appending ".pt" to it. 
    
    NOTE: Wraps the forward() call to accept a single input.
    """
    if len(select_revision_kwargs)!=0:
        if os.path.isdir(path_or_name):
            raise AssertionError(f"To load older commits, must fetch model from HuggingFace remote url repo,\
                but {path_or_name} is local directory!")
        revision = select_revision(path_or_name, **select_revision_kwargs)
    else:
        revision=None #===latest

    try:    
        model = model_type.from_pretrained(path_or_name, from_flax=from_flax, revision=revision)
    except (HTTPError, OSError, ValueError) as e:
        print("Encountered Error:", e, flush=True)
        print("Trying to load model from {}.pt".format(path_or_name), flush=True)
        model = model_type.from_pretrained(base_model, num_labels=3,)
        state_dict_path = path_or_name+".pt"
        try:
            model.load_state_dict(torch.load(state_dict_path))
            print("Model loaded from {}.pt".format(path_or_name), flush=True)
        except:
            raise e
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(X):
            return func(**X)
        return wrapper
    
    model.forward = decorator(model.forward)
    return model

get_sequence_classification_model = functools.partial(get_model,model_type=AutoModelForSequenceClassification)

get_model_for_masked_lm = functools.partial(get_model, model_type=AutoModelForMaskedLM)

def expand_resolution(params, resolution: int, patch_size: int, model_config):
    posemb = params["Transformer"]["posembed_input"]["pos_embedding"]   #Shape [1, 1+{res*res/(patch_size*patch_size)}, d_emb]
    orig_resolution = int(sqrt(posemb.shape[1]-1)*patch_size)
    if resolution!=orig_resolution:
        if resolution<orig_resolution:
            raise ValueError("Invalid operation: Trying to reduce reduce resolution.")
        if model_config.classifier=="token":
            posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        else:
            posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
        
        n_patches_per_axis = int((resolution/patch_size))
        n_patches_per_axis_old = int(orig_resolution/patch_size)
        zoom_ratio = n_patches_per_axis/n_patches_per_axis_old
        zoom = (zoom_ratio, zoom_ratio, 1)
        posemb_grid = posemb_grid.reshape(n_patches_per_axis_old, n_patches_per_axis_old, -1)
        posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
        posemb_grid = posemb_grid.reshape(1, n_patches_per_axis*n_patches_per_axis, -1)
        expanded_posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
        params["Transformer"]["posembed_input"]["pos_embedding"] = expanded_posemb
    return params

def get_criterion_fn(loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                     logit_converter_fn: Optional[Callable[[SCO], SCO]]=None,) -> Callable[[SCO, torch.Tensor], torch.Tensor]:
    """Returns a criterion to calculate loss for a sequence classification model.
    Args:
        loss_fn:  A function that takes (logits, labels) and returns a loss tensor.
        logit_converter_fn:  A function that takes model output and returns model output,
                             possibly with converted output logits. If not provided, logits
                             as given in original model output are used.
    Returns:
        A function that takes (model_out, labels) and returns a loss tensor, possibly with an
        additional logit conversion step, if logit_conversion_fn is provided.
    """
    def simple_criterion(model_out: SCO, target: torch.Tensor) -> torch.Tensor:
        return loss_fn(model_out["logits"], target)
    
    def converted_criterion(model_out: SCO, target: torch.Tensor) -> torch.Tensor:
        model_out = logit_converter_fn(model_out)
        return simple_criterion(model_out, target)
    
    if logit_converter_fn is None:
        return simple_criterion
    
    return converted_criterion

def get_pred_fn(pred_type: str = "argmax",
                logit_converter_fn: Optional[Callable[[SCO], SCO]] = None,) -> Callable[[SCO], torch.Tensor]:
    """Returns a prediction function for sequence classification model.
    Args:
        pred_type:          "max":    Return max logit values.
                            "argmax": Return max logit index.
                            "prob":   Return the probabilities calculated from logits. 
        
        logit_converter_fn:  A function to convert logits to a different format.
                             For e.g. see get_logits_converter().
    Returns:
        A function that takes model_out and returns a prediction.
    """
    supported_pred_types = ["max", "argmax", "prob"]
    if pred_type not in supported_pred_types:
        raise NotImplementedError(f"Prediction function for doing {pred_type} type predictions not implemented.\
                                    Choose from: {supported_pred_types}")
                
    def simple_pred(model_out: SCO) -> torch.Tensor:
        if pred_type == "argmax":
            return torch.max(model_out["logits"], dim=-1).indices
        elif pred_type == "max":
            return torch.max(model_out["logits"], dim=-1).values
        elif pred_type == "prob":
            return torch.softmax(model_out["logits"], dim=-1)
        
        raise AssertionError("Unreachable code")

    def converted_pred(model_out: SCO) -> torch.Tensor:
        model_out = logit_converter_fn(model_out)
        return simple_pred(model_out)
    
    if logit_converter_fn is None:
        return simple_pred
    
    return converted_pred
