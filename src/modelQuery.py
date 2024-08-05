#!/usr/bin/env python3

import os
import sys
import time
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

    # for Llama-3-405B or larger models
    if int(params['dim']) >= 16384:
        print ('DIM=', params['dim'])
        model_paths = sorted(list(Path(model_path).glob('consolidated.*/consolidated*.pth')))

    for i, s in enumerate(model_paths):
        print ("%03d: %s" % (i, s))

    total_params = 0
    total_bytes = 0
    time_total = 0.0
    time_lap = 0.0
    time_prev = time.time()

    for fi, fname in enumerate(model_paths):
        print ('\n', '-' * 30)
        print ("%d params in total." % total_params)
        print ("%d bytes in total." % total_bytes)
        time_lap = time.time() - time_prev
        time_prev = time.time()
        time_total += time_lap

        if (total_bytes > 0):
            bw = (total_bytes / time_total) / (1024. * 1024.) # MB/sec
            print("%.2f sec, %.2f sec, %.2f MB/s" % (time_total, time_lap, bw))

        print ("[%d/%d]: Loading %s" % (1+fi, len(model_paths),fname)) 
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
