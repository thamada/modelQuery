# modelQuery

Query the internals of the Meta's Llama model.

## Instructions

The `modelQuery.py` reads the pth files and retrieves information. Specify the directory where your model's params.json resides. 
For example:

```:shell
git clone https://github.com/thamada/modelQuery.git
python3 ./modelQuery/src/modelQuery.py ~/models/Meta-Llama-3.1-405B-Instruct/models--meta-llama--Meta-Llama-3.1-405B-Instruct/snapshots/cfe126a16d4108374a6f9cda6d117c0d08b99e23/original/mp16/
```

## Queries

| model | params | params (B) | size (Bytes) | size (GB) | layers | dim  | etc        | 
| ----- | -----: | ---------: | -----------: | --------: | -----: | ---: | :--------: |
| [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | 8030261248 | 8.0303 | 16060522496 | 14.9575 | 32 | 4096 | [*](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/original/params.json) |
| [Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) | 8030261248 | 8.0303 | 16060522496 | 14.9575 | 32 | 4096 | [*](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/original/params.json) |
| [Llama-2-13B](https://huggingface.co/meta-llama/Llama-2-13b) | 13016279168 | 13.0163 | 26032558592 | 24.2447 | 40 | 5120 | [*](https://huggingface.co/meta-llama/Llama-2-13b/blob/main/params.json) |
| [Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)     | 70562938880 | 70.5629 | 141125877760 | 131.4337 | 80 | 8192 | [*](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/blob/main/original/params.json) |
| [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) | 70562938880 | 70.5629 | 141125877760 | 131.4337 | 80 | 8192 | [*](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct/blob/main/original/params.json) |
| [Llama-3.1-405B](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B) | 410143424512 | 410.1434 | 820286849024 | 763.9517 | 126 | 16384 | [*](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B/blob/main/original/mp16/params.json) |



