from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
import pickle

import torch

from torch import nn

import transformers
import datasets


def eval(loader: Iterable[Tuple[Any, Any]],
         model: nn.Module, 
         criterion: Union[nn.Module, Callable],
         pred_fn: Optional[Callable]=None,
         metric:  Optional[datasets.Metric]=None) -> Dict[str, float]:
    """Evaluates the model on the data in the loader, using the criterion and metric provided.
    Args:
        loader:    A data loader for the model to be evaluated on. Should yield tuples 
                   of (input, target).
        model:     The model to be evaluated. The forward() method must take in input
                   yielded by loader.
        criterion: The loss function to be used for evaluation. Must take in output
                   of model.forward() and target yielded by loader.
        pred_fn:   A function that takes in the output of model.forward() and returns the predictions
                   that can be passed in metric().
        metric:    A class with an add_batch() method that takes in output of model.forward() 
                   and target yielded by loader. Additionally implements a compute() function 
                   that returns the value of metric over all the samples added using add_batch().
    Returns:
        A dictionary containing the loss and metric values of the model on the data in the loader.
    """
    if pred_fn is None and metric is not None:
        raise ValueError("Must provide pred_fn if metric is provided.")
    elif pred_fn is not None and metric is None:
        raise ValueError("Must provide metric if pred_fn is provided.")
    
    loss_sum = 0.0

    model.eval()

    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        loss_sum += loss.data.item() * input["input_ids"].size(0)
        if metric is not None:
            pred = pred_fn(output)
            metric.add_batch(predictions=pred,
                             references=target)
    
    metric_dict = {} if metric is None else metric.compute()
    
    metric_dict.update({
        'loss': loss_sum / len(loader.hf_loader.dataset),
    })

    return metric_dict

def store_outs(loader: Iterable[Tuple[Any, Any]],
               write_file: str,
               tokenizer: transformers.PreTrainedTokenizer,
               model: nn.Module, 
               pred_fn: Callable,
               label_dict: Dict[str, int]):
    """Writes the predictions of provided model, on every sample in loader 
    to write_file(in json format).
    Args:
        loader:    A data loader for the model to be evaluated on. Should yield tuples 
                   of (input, target).
        write_file:The file to which the predictions are to be written.
        tokenizer: The tokenizer used to encode the sentences, will be used to decode input_ids,
                   back to sentences, and sentences will be stored along their predictions.
        model:     The model to be evaluated. The forward() method must take in input
                   yielded by loader.
        pred_fn:   A function that takes in the output of model.forward() and returns the 
                   probabilities of various labels.
        label_dict:Dictionary specifying the integer labels for various classes, will be 
                   stored along the predictions on each sample.
    """
    
    evaluated_samples = {}
    evaluated_samples["label_dict"] = label_dict
    evaluated_samples["samples"] = []
    model.eval()
    
    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            output = model(input)
            preds = pred_fn(output)
        preds = preds.cpu()
        batch_logits = output["logits"].cpu()
        sents = tokenizer.batch_decode(input["input_ids"])
        for (probs, logits, label, sentence) in zip(preds, batch_logits, target, sents):
            evaluated_samples["samples"].append({"probabilities": probs.tolist(),
                                                 "logits": logits.tolist(),
                                                 "label": label.item(),
                                                 "sentence": sentence.replace(tokenizer.pad_token, "").strip()})
    
    with open(write_file, "wb") as f:
        pickle.dump(evaluated_samples, f)
    
    print(f"Wrote evaluation data to file {write_file}!", flush=True)
    
    return None
