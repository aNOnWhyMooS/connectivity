from typing import Callable, Optional, Dict
import torch
from transformers.modeling_outputs import SequenceClassifierOutput as SCO

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
