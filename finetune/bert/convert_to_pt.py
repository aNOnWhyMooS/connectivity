import os, sys
import shutil
import torch

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from huggingface_hub import Repository

# Function from: https://github.com/huggingface/transformers/blob/215e0681e4c3f6ade6e219d022a5e640b42fcb76/src/transformers/models/bert/modeling_bert.py#L109
def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    
    tf_path = os.path.abspath(tf_checkpoint_path)
    print(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    
    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            print(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if (scope_names[0]=="output_weights" or scope_names[0]=="output_bias" 
                and hasattr(pointer, "classifier")):
                pointer = getattr(pointer, "classifier")
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    print(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model

if __name__=="__main__":
    model_num, hf_auth_token = sys.argv[1], sys.argv[2]

    config = BertConfig.from_json_file("./uncased_L-12_H-768_A-12/bert_config.json")
    bert = BertForSequenceClassification(config=config)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    hf_repo_dir = f"bert_ft_qqp-{model_num}"
    assert not os.path.isdir(hf_repo_dir)

    tokenizer.push_to_hub(hf_repo_dir, 
                          commit_message="Saving tokenizer",
                          use_auth_token=hf_auth_token)
    
    model_save_dir = f"qqp_save_{model_num}"
    
    for steps in [15000, 20000, 25000, 30000, 34110]:    
        ckpt = f"{model_save_dir}/model.ckpt-{steps}"
        bert = load_tf_weights_in_bert(bert, config, ckpt)
        
        bert.push_to_hub(hf_repo_dir,
                         commit_message=f"Saving weights and logs of step {steps}",
                         use_auth_token=hf_auth_token)
    
    
    repo = Repository(local_dir=hf_repo_dir, use_auth_token=hf_auth_token)
    
    for file in os.listdir(model_save_dir):
        filepath = os.path.join(model_save_dir, file)
        if os.path.isfile(filepath) and "events.out.tfevents" in filepath:
            shutil.copy(filepath, hf_repo_dir)
    repo.push_to_hub(commit_message="Added training logs")
