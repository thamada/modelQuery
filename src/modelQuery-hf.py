#!/usr/bin/env python3

"""
example:
./modelQuery-hf.py --hf tokyotech-llm/Llama-3-Swallow-8B-v0.1
"""

import argparse
import torch
from transformers import AutoModelForCausalLM

def get_n_params(_tensor_):
    n = 1
    for i in range(_tensor_.dim()):
        n *= _tensor_.shape[i]
    return n

def replace_key_names(key):
    # Define your mapping here
    replacements = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers": "layers",
        "self_attn.q_proj.weight": "attention.wq.weight",
        "self_attn.k_proj.weight": "attention.wk.weight",
        "self_attn.v_proj.weight": "attention.wv.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight"
    }

    # Replace the keys based on the mapping
    for old_key, new_key in replacements.items():
        key = key.replace(old_key, new_key)
    return key

def load_hf_model(model_path):
    # load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    m = hf_model.state_dict()
    print("")
    print(hf_model.config)
    print("")

    total_params = 0
    total_bytes = 0

    for i, key in enumerate(m.keys()):
        n_param = get_n_params(m[key])
        n_bytes = n_param * m[key].element_size()
        total_params += n_param
        total_bytes += n_bytes
        s_size = str(m[key].size()).replace('torch.Size','').replace('(','').replace(')','')

        # Apply key replacements
        replaced_key = replace_key_names(key)

        print ('% 4d' % i,
               ': % 10d' % get_n_params(m[key]),
               ': %035s' % replaced_key.ljust(35),
               ": %015s" % s_size.ljust(15),
               ": %s" % str(m[key].dtype)
               )

    print ('Total number of parameters: %d' % total_params)
    print ('%.4f B' % (total_params / 1000000000.))
    print ('%d Bytes' % total_bytes)
    print ('%.4f GB' % (total_bytes / (1024.*1024.*1024.)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hf", type=str, help="huggingface model path")
    args = parser.parse_args()
    load_hf_model(args.hf)
