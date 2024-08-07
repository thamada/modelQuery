#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

token = os.getenv('HF_TOKEN')

model = {'hf': 'openai-community/openai-gpt', 'dir': 'gpt-1'}
model = {'hf': 'openai-community/gpt2-xl',    'dir': 'gpt2-xl'}
model = {'hf': 'meta-llama/Llama-2-7b',       'dir': 'llama-2-7b'}
model = {'hf': 'meta-llama/Llama-2-13b',      'dir': 'llama-2-13b'}
model = {'hf': 'meta-llama/Llama-2-7b',       'dir': 'llama-2-7b'}
model = {'hf': 'meta-llama/Meta-Llama-3-8B',              'dir': 'llama3-8b'}
model = {'hf': 'meta-llama/Meta-Llama-3-8B-Instruct',     'dir': 'llama3-8b-inst'}
model = {'hf': 'meta-llama/Meta-Llama-3-70B',             'dir': 'llama3-70b'}
model = {'hf': 'meta-llama/Meta-Llama-3.1-8B',            'dir': 'llama3.1-8b'}
model = {'hf': 'meta-llama/Meta-Llama-3.1-70B-Instruct',  'dir': 'llama-3.1-70b-inst'}
model = {'hf': 'meta-llama/Meta-Llama-3.1-405B-Instruct', 'dir': 'llama-3.1-405b-inst'}
model = {'hf': 'cyberagent/calm3-22b-chat',    'dir': 'cyberagent-calm3-22b-chat'}
model = {'hf': 'cyberagent/open-calm-small',   'dir': 'cyberagent-small'}

path = snapshot_download(repo_id=model['hf'], cache_dir=model['dir'], use_auth_token=token)

print("_\n" * 20)
print(path)
