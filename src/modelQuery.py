#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import json
import torch

def get_n_params(_tensor_):
    n = 1
    for i in range(_tensor_.dim()):
        n *= _tensor_.shape[i]
    return n

def load_modelfile(model_path):

    params_path = os.path.join(model_path, 'params.json')
    with open(params_path) as f:
        params = json.load(f)
        print(params)

    model_paths = sorted(list(Path(model_path).glob('consolidated.*.pth')))

    total_params = 0
    total_bytes = 0

    for fname in model_paths:
        m = torch.load(fname, map_location='cpu', weights_only=True)
        for i, key in enumerate(m.keys()):
            n_param = get_n_params(m[key])
            n_bytes = n_param * m[key].element_size()

            total_params += n_param
            total_bytes += n_bytes

            print ('% 4d' % i,
                   ':',
                   '% 15d:' % get_n_params(m[key]),
                   key,
                   ":\t\t",
                   m[key].size(),
                   )

            '''
            print (i,
                   key,
                   type(m[key]), 
                   m[key].size(),
                   ': ',
                   m[key].dim(),
                   m[key].dtype,
                   m[key].element_size()
                   )
            '''

    print ('Total number of parameters: %d' % total_params)
    print ('%.4f B' % (total_params / 1000000000.))
    print ('%d Bytes' % total_bytes)
    print ('%.4f GB' % (total_bytes / (1024.*1024.*1024.)))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('[Llama model folder path]')
        exit()

    model_path = sys.argv[1]

    load_modelfile(model_path)
