#!/usr/bin/env python3

"""
example:
./modelQuery-hf.py --hf tokyotech-llm/Llama-3-Swallow-8B-v0.1
"""

import argparse
from transformers import AutoModelForCausalLM

def get_n_params(_tensor_):
    n = 1
    for i in range(_tensor_.dim()):
        n *= _tensor_.shape[i]
    return n

def load_hf_model(model_path):
    # load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    m = hf_model.state_dict()
    print("")
    print('-' * 80)
    for key, value in hf_model.config.to_dict().items():
        if key != "torch_dtype":
            print(f"{key.rjust(35)}: {value}")
    print('-' * 80)
    print("")

    total_params = 0
    total_bytes = 0

    for i, key in enumerate(m.keys()):

        n_param = get_n_params(m[key])
        n_bytes = n_param * m[key].element_size()
        total_params += n_param
        total_bytes += n_bytes
        s_size = str(m[key].size()).replace('torch.Size','').replace('(','').replace(')','')

        print ('% 4d' % i,
               ': % 10d' % get_n_params(m[key]),
               ': %050s' % key.ljust(50),
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
