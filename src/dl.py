#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

token = os.getenv('HF_TOKEN')

#path = snapshot_download(repo_id="meta-llama/Llama-2-7b", cache_dir="Llama-2-7b", use_auth_token=token)
#path = snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", cache_dir="Llama3-8b", use_auth_token=token)
#path = snapshot_download(repo_id="cyberagent/open-calm-small", cache_dir="cyberagent-small", use_auth_token=token)
#path = snapshot_download(repo_id="cyberagent/calm3-22b-chat", cache_dir="cyberagent-calm3-22b-chat", use_auth_token=token)
#path = snapshot_download(repo_id="meta-llama/Meta-Llama-3.1-70B-Instruct", cache_dir="Meta-Llama-3.1-70B-Instruct", use_auth_token=token)
#path = snapshot_download(repo_id="meta-llama/Meta-Llama-3.1-405B-Instruct", cache_dir="Meta-Llama-3.1-405B-Instruct", use_auth_token=token)
#path = snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B", cache_dir="/mnt/models/Llama3-8b", use_auth_token=token)
#path = snapshot_download(repo_id="meta-llama/Meta-Llama-3.1-8B", cache_dir="/mnt/models/Llama3.1-8b", use_auth_token=token)

path = snapshot_download(repo_id="meta-llama/Meta-Llama-3.1-70B-Instruct", cache_dir="/mnt/models/Meta-Llama-3.1-70B-Instruct", use_auth_token=token)



print("_\n" * 20)
print(path)
