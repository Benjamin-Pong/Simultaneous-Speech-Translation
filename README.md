# Simultaneous-Speech-Translation
This repository contains the code to built a simultaneous speech translation model using dynamic masking. A folder containing the scripts to build a Docker Image is also provided.

Benjamin Pong  
benpongmca@gmail.com


## Method
- **Base model**: SeamlessM4T-medium (facebook/hf-seamless-m4t-medium) with schedulers for dynamic masking
- **Training**: LoRA finetuning 
- **Inference**: Sliding window retranslation


## Files
- mcif-dev log files
- docker
    - `dynamic_sliding_window.py` — custom speech processor class
    - `model_sm4t.py` — DynamicSeamlessM4T model with scheduler MLPs
    - `dynamic_sliding_window.yaml` — server configuration
    - `Dockerfile` — Docker build file


## Command to Build Dockerimage
```shell
docker build \
  --build-context simulstream_base=/path/to/simulstream \
  -t benjamin_sst .
```

## Run dockerimage
```shell
docker run --rm --gpus=all -p 8080:8080 benjamin_sst
```

## Docker Hub 
Alternatively, the dockerimage can be pulled from dockerhub 
```shell
docker pull benjaminpwh/benjamin_sst
```

## Inference
```shell
simulstream_inference \
    --speech-processor-config config/http_proxy_processor.yaml \
    --wav-list-file $WAV_LIST \
    --tgt-lang $TGT_LANG \
    --src-lang eng \
    --metrics-log-file $OUTPUT_JSONL
```

Supported target languages: `deu`, `cmn`, `ita`
