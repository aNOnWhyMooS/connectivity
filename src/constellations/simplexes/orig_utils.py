from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
import jax
import json

import torch
import random
import jax.numpy as jnp

import flax
import flax.linen as fnn

from torch import nn
import torch.nn.functional as F

import transformers
import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # print(tensor.numel())
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def assign_pars(vector: torch.Tensor, model: nn.Module) -> None:
    """Assigns parameters specified in vector to model
    Args:
        vector: torch.Tensor of shape (n_pars,) where n_pars is the total
                number of parameters in the model. The parameters must be
                in the same order as those yielded by model.parameters().
        model:  A model to which the parameters will be copied to.

    NOTE:
        The gradients don't propagate backward through the .data assignment, i.e.,
        the gradients won't propagate through the model to the vector, but only
        till the model parameters, when loss.backward() is called.
    """
    new_pars = unflatten_like(vector, model.parameters())
    for old, new in zip(model.parameters(), new_pars):
        old.data = new.to(old.device).data

    return


def eval_vit(
    loader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]],
    params: flax.core.frozen_dict.FrozenDict,
    eval_batch_fn: Callable,
) -> Dict[str, float]:
    loss = 0.0
    correct_preds = 0
    total_preds = 0

    for images, labels in loader:
        batch_loss, correct, total = eval_batch_fn(
            params, jnp.array(images), jnp.array(labels)
        )
        loss += batch_loss.item()
        correct_preds += correct.item()
        total_preds += total.item()

    metrics_dict = {
        "loss": loss / total_preds,
        "total_samples": total_preds,
        "correct_preds": correct_preds,
        "accuracy": 100 * (correct_preds / total_preds),
    }

    return metrics_dict


def eval_mlm_task(
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    mask_token_id: int,
) -> Dict[str, float]:
    model.eval()
    total_masks = 0
    correct_preds = 0
    loss = 0.0

    for i, (input, labels) in enumerate(loader):
        with torch.no_grad():
            output = model(input)
            mask_pos = input["input_ids"] == mask_token_id
            total_masks += mask_pos.sum().cpu().item()
            loss += (output.loss * total_masks).cpu().item()
            preds = output.logits[mask_pos].max(axis=-1).indices
            gold_labels = labels[mask_pos].view(-1)
            correct_preds = torch.sum(gold_labels == preds).cpu().item()

    metric_dict = {
        "correct_preds": correct_preds,
        "total_samples": total_masks,
        "accuracy": 100 * (correct_preds / total_masks),
        "loss": loss / total_masks,
    }

    return metric_dict


def eval_na_task(
    loader: Iterable[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]],
    model: nn.Module,
    mask_token_id: int,
) -> Dict[str, float]:
    model.eval()
    tot_samples = 0
    correct_preds = 0
    loss = 0.0

    for i, (input, correct_ans_ids, incorrect_ans_ids) in enumerate(loader):
        with torch.no_grad():
            output_logits = model(input).logits[input["input_ids"] == mask_token_id]
            correct_logits = output_logits[
                torch.arange(output_logits.size(0)), correct_ans_ids
            ]
            incorrect_logits = output_logits[
                torch.arange(output_logits.size(0)), incorrect_ans_ids
            ]
            loss += torch.sum(incorrect_logits - correct_logits).detach().item()
            correct_preds += torch.sum(correct_logits > incorrect_logits).item()
            tot_samples += input["input_ids"].size(0)

    metric_dict = {
        "correct_preds": correct_preds,
        "total_samples": tot_samples,
        "accuracy": 100 * (correct_preds / tot_samples),
        "loss": loss / tot_samples,
    }

    return metric_dict


def store_outs(
    loader: Iterable[Tuple[Any, Any]],
    write_file: str,
    tokenizer: transformers.PreTrainedTokenizer,
    model: nn.Module,
    pred_fn: Callable,
    label_dict: Dict[str, int],
) -> Dict[str, float]:
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
    Returns:
        A dictionary containing the loss and metric values of the model on the data in the loader.
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
        for probs, logits, label, sentence in zip(preds, batch_logits, target, sents):
            evaluated_samples["samples"].append(
                {
                    "probabilities": probs.tolist(),
                    "logits": logits.tolist(),
                    "label": label.item(),
                    "sentence": sentence.replace(tokenizer.pad_token, "").strip(),
                }
            )

    with open(write_file, "w") as f:
        f.write(json.dumps(evaluated_samples, indent=4))

    print(f"Wrote evaluation data to file {write_file}!", flush=True)

    return None


def eval_model(
    loader: Iterable[Tuple[Any, Any]],
    model: nn.Module,
    criterion: Union[nn.Module, Callable],
    pred_fn: Optional[Callable] = None,
    metric: Optional[datasets.Metric] = None,
) -> Dict[str, float]:
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
            metric.add_batch(predictions=pred, references=target)

    metric_dict = {} if metric is None else metric.compute()

    metric_dict.update(
        {
            "loss": loss_sum / len(loader.hf_loader.dataset),
        }
    )

    return metric_dict


class BaseParamModifier(object):
    def __init__(self, model: nn.Module):
        self.params = []
        with torch.no_grad():
            model.apply(lambda module: self.get_params(module, self.params))

    @staticmethod
    def get_params(module: nn.Module, params_store: list):
        for name in list(module._parameters.keys()):
            if module._parameters[name] is None:
                continue
            params_store.append((module, name))


class ModelWeightClipper(BaseParamModifier):
    """Implements the weight clipping for constricting the model weights
    each weight is clamped between param_value-ɛ(param_value+1) and
    param_value+ɛ(param_value+1) on each call of the clip() method.

    NOTE: Maintains original parameter values on CPU, so that the limits to which each
          parameter is clamped, remain constant.
    """

    def __init__(self, model, epsilon):
        super().__init__(model)
        self.eps = epsilon
        self.orig_params = [
            (module, name, module._parameters[name].data.detach().clone().cpu())
            for module, name in self.params
        ]

    def __call__(self):
        with torch.no_grad():
            for module, name, orig_param in self.orig_params:
                param = module._parameters[name]
                clamp_bound = orig_param.to(param.device)
                lower_bounds = torch.min(
                    clamp_bound - self.eps * (clamp_bound + 1),
                    clamp_bound + self.eps * (clamp_bound + 1),
                )
                upper_bounds = torch.max(
                    clamp_bound - self.eps * (clamp_bound + 1),
                    clamp_bound + self.eps * (clamp_bound + 1),
                )
                param.clamp_(lower_bounds, upper_bounds)

    def clip(self):
        self()


class MaskGrads(BaseParamModifier):
    """Supply this class with a model and it will maintain the gradients of each parameter
    of the model. Gradients to all the model weights except a fixed random set of
    num_random_dirs parameters will be zeroed out, on each mask_grad() call."""

    def __init__(self, model: nn.Module, num_random_dirs: int):
        super().__init__(model)
        self.num_random_dirs = num_random_dirs
        total_dirs = sum(
            [module._parameters[name].numel() for (module, name) in self.params]
        )

        if self.num_random_dirs > total_dirs:
            raise ValueError(
                f"Number of random directions: {self.num_random_dirs} \
                is greater than total number of parameters: {total_dirs}. Kindly \
                    provide smaller random directions."
            )

        self.random_dirs = random.Random(42).sample(
            range(total_dirs), self.num_random_dirs
        )
        self.random_dirs.sort()
        self.construct_masks()

    def construct_masks(self):
        """Construct the mask for each parameter according to self.random_dirs list.
        Each mask will be used to multiply the gradients of the parameters.
        """
        self.masks = []
        elems_till_now = 0
        idx = 0
        for module, name in self.params:
            param = module._parameters[name]
            num_elems_in_param = param.numel()
            mask = torch.zeros((num_elems_in_param,))
            while (
                idx < len(self.random_dirs)
                and self.random_dirs[idx] < elems_till_now + num_elems_in_param
            ):
                mask[self.random_dirs[idx] - elems_till_now] = 1
                idx += 1
            elems_till_now += num_elems_in_param
            self.masks.append(mask.reshape(param.shape))

    def __call__(self):
        with torch.no_grad():
            for mask, (module, name) in zip(self.masks, self.params):
                param = module._parameters[name]
                param.grad.data = mask.to(param.device) * param.grad.data

    def mask_grads(self):
        self()


def flatness_measure(
    loader: Iterable[Tuple[Any, Any]],
    model: nn.Module,
    criterion: Union[nn.Module, Callable],
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int = 1,
    epsilon: float = 1e-5,
    num_random_dirs: Optional[int] = None,
) -> float:
    """Measures the ɛ-sharpness metric of https://arxiv.org/abs/1703.04933 as introduced in
    https://arxiv.org/abs/1609.04836.
    Args:
        loader:    A data loader for the model to be evaluated on. Should yield tuples
                   of (input, target).
        model:     The model to be evaluated. The forward() method must take in input
                   yielded by loader.
        criterion: The loss function to be used for evaluation. Must take in output
                   of model.forward() and target yielded by loader.

        optimizer: The optimizer to be used for updating the model parameters.
                   Can have all parameters of the model.

        gradient_accumulation_steps: The number of gradient steps to accumulate before
                                     each optimizer.step().
        epsilon:        The epsilon as specifed in the paper. Each parameter is allowed to vary b/w
                        -ɛ(initial_value+1)<param<ɛ(initial_value+1).

        num_random_dirs:  The number of random directions(only axis aligned directions supported)
                          which are updatable.
    Returns:
        The ɛ-sharpness of the model.
    """
    original_loss = eval_model(loader, model, criterion)["loss"]
    print("Original Loss:", original_loss, flush=True)

    model.train()

    clipper = ModelWeightClipper(model, epsilon)

    if num_random_dirs is not None:
        masker = MaskGrads(model, num_random_dirs)

    for i, (input, target) in enumerate(loader):
        output = model(input)
        loss = criterion(output, target)
        loss = -loss
        loss.backward()
        if (i + 1) % gradient_accumulation_steps == 0:
            if num_random_dirs is not None:
                masker.mask_grads()
            optimizer.step()
            optimizer.zero_grad()
            clipper.clip()

    maximized_loss = eval_model(loader, model, criterion)["loss"]

    print("Maximized loss:", maximized_loss, flush=True)

    return 100 * (maximized_loss - original_loss) / (1 + original_loss)


def train_epoch(
    loader: Iterable[Tuple[Any, Any]],
    model: nn.Module,
    criterion: Union[nn.Module, Callable],
    optimizer: torch.optim.Optimizer,
    metric: Optional[datasets.Metric] = None,
) -> Dict[str, float]:
    """Trains the model on the data in the loader, while logging the loss and metrics.
    Args:
        loader:    A data loader for the model to be trained on. Should yield tuples
                   of (input, target).

        model:     The model to be trained. The forward() method must take in input
                   yielded by loader.

        criterion: The loss function to be used for training. Must take in output
                   of model.forward() and target yielded by loader.

        optimizer: The optimizer to be used for updating the model parameters.

        metric:    A class with an add_batch() method that takes in output of model.forward()
                   and target yielded by loader. Additionally implements a compute() function
                   that returns the value of metric over all the samples added using add_batch().
                   By default, calculates accuracy.
    Returns:
        A dictionary containing the loss and metric values of the model on the data in the loader.
    """
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        "loss": loss_sum / len(loader.dataset),
        "accuracy": correct / len(loader.dataset) * 100.0,
    }


def train_epoch_volume(
    loader: Iterable[Tuple[Any, Any]],
    model: nn.Module,
    criterion: Union[nn.Module, Callable],
    optimizer: torch.optim.Optimizer,
    vol_reg: float,
    nsample: int,
    metric: Optional[datasets.Metric] = None,
) -> Dict[str, float]:
    """Trains a model(simplex of models) on the data in the loader with volume regularisation,
    while logging the loss and metrics.
    Args:
        loader:    A data loader for the model to be trained on. Should yield tuples
                   of (input, target).

        model:     The model to be trained. Must maintain a simplex of models. The forward()
                   method must take in input yielded by loader, and use (model-)weights
                   from a random point in the simplex. Must implement a total_volume() function
                   which returns the total volume of the simplex of models. A model wrapped in
                   SimplexNet or BasicSimplex class is suitable input.

        criterion: The loss function to be used for evaluation. Must take in output
                   of model.forward() and target yielded by loader.

        optimizer: The optimizer to be used for updating the model parameters.

        vol_reg:   The λ scaling factor for log of total volume of simplex.

        nsample:   The number of (model-)weights to sample from the simplex, and calculate loss on.
                   H in the paper.

        metric:    A class with an add_batch() method that takes in output of model.forward()
                   and target yielded by loader. Additionally implements a compute() function
                   that returns the value of metric over all the samples added using add_batch().
                   By default, calculates accuracy.
    Returns:
        A dictionary containing the loss and metric values of the model on the data in the loader.
    """
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)

        acc_loss = 0.0
        for _ in range(nsample):
            output = model(input)
            acc_loss = acc_loss + criterion(output, target)
        acc_loss.div(nsample)

        vol = model.total_volume()
        log_vol = (vol + 1e-4).log()

        loss = acc_loss - vol_reg * log_vol

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        "loss": loss_sum / len(loader.dataset),
        "accuracy": correct / len(loader.dataset) * 100.0,
    }


def train_epoch_multi_sample(
    loader: Iterable[Tuple[Any, Any]],
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    nsample: int,
    metric: Optional[datasets.Metric] = None,
) -> Dict[str, int]:
    """Trains a simplex of models on the data in the loader without volume regularisation,
    while logging the loss and metrics.
    Args:
        loader:    A data loader for the model to be trained on. Should yield tuples
                   of (input, target).

        model:     The model to be trained. Must maintain a simplex of models. The forward()
                   method must take in input yielded by loader, and use (model-)weights
                   from a random point in the simplex. Must implement a total_volume() function
                   which returns the total volume of the simplex of models. A model wrapped in
                   SimplexNet or BasicSimplex class is suitable input.

        criterion: The loss function to be used for evaluation. Must take in output
                   of model.forward() and target yielded by loader.

        optimizer: The optimizer to be used for updating the model parameters.

        nsample:   The number of (model-)weights to sample from the simplex, and calculate loss on.
                   H in the paper.

        metric:    A class with an add_batch() method that takes in output of model.forward()
                   and target yielded by loader. Additionally implements a compute() function
                   that returns the value of metric over all the samples added using add_batch().
                   By default, calculates accuracy.
    Returns:
        A dictionary containing the loss and metric values of the model on the data in the loader.
    """
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        acc_loss = 0.0
        for _ in range(nsample):
            output = model(input_var)
            acc_loss += criterion(output, target_var)
        acc_loss.div(nsample)

        loss = acc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        "loss": loss_sum / len(loader.dataset),
        "accuracy": correct / len(loader.dataset) * 100.0,
    }


def train_transformer_epoch(
    loader: Iterable[Tuple[Any, Any]],
    model: transformers.PreTrainedModel,
    criterion: Union[nn.Module, Callable],
    optimizer: torch.optim.Optimizer,
    nsample: int,
    vol_reg: float = 1e-5,
    gradient_accumulation_steps: int = 1,
    pred_fn: Optional[Callable] = None,
    metric: Optional[datasets.Metric] = None,
) -> Dict[str, float]:
    """Trains a simplex of models on the data in the loader with volume regularisation,
    while logging the loss and metrics.
    Args:
        loader:    A data loader for the model to be trained on. Should yield tuples
                   of (input, target).

        model:     The model to be trained. Must maintain a simplex of models. The forward()
                   method must take in input yielded by loader, and use (model-)weights
                   from a random point in the simplex. Must implement a total_volume() function
                   which returns the total volume of the simplex of models. A model wrapped in
                   SimplexNet or BasicSimplex class is suitable input.

        criterion: The loss function to be used for evaluation. Must take in output
                   of model.forward() and target yielded by loader.

        optimizer: The optimizer to be used for updating the model parameters.

        nsample:   The number of (model-)weights to sample from the simplex, and calculate loss on.
                   H in the paper.

        pred_fn:   A function that takes in the output of model.forward() and returns the predictions
                   that can be passed in metric().

        metric:    A class with an add_batch() method that takes in output of model.forward()
                   and target yielded by loader. Additionally implements a compute() function
                   that returns the value of metric over all the samples added using add_batch().
                   By default, calculates accuracy.
    Returns:
        A dictionary containing the loss and metric values of the model on the data in the loader.
    """

    if pred_fn is None and metric is not None:
        raise ValueError("Must provide pred_fn if metric is provided.")
    elif pred_fn is not None and metric is None:
        raise ValueError("Must provide metric if pred_fn is provided.")

    loss_sum = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        if i % 20 == 0:
            print(i, "batches completed", flush=True)
        torch.cuda.empty_cache()

        # Backpropagate Volume Loss
        if vol_reg != 0:
            vol = model.total_volume()
            log_vol = vol_reg * (vol + 1e-4).log()
            vol_loss = -log_vol
            if gradient_accumulation_steps > 1:
                vol_loss = vol_loss / gradient_accumulation_steps
            vol_loss.backward()
            torch.cuda.empty_cache()

        for _ in range(nsample):
            output = model(input)
            loss = criterion(output, target)
            loss = loss / nsample
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            loss_sum += loss.item() * input["input_ids"].size(0)
            torch.cuda.empty_cache()

        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        if metric is not None:
            pred = pred_fn(output)
            metric.add_batch(predictions=pred, references=target)

    print("Completed loader.")
    metric_dict = {} if metric is None else metric.compute()

    metric_dict.update(
        {
            "loss": loss_sum / len(loader.hf_loader.dataset),
        }
    )

    print("Computed metrics:", metric_dict)

    return metric_dict
