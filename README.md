# SpecEE
Repo for SpecEE: Accelerating Large Language Model Inference with Speculative Early Exiting (ISCA25)
Paper website: https://arxiv.org/pdf/2504.08850

# Acknowledgement
The SpecEE is implemented based on the HuggingFace framework in the cloud scenario and the llama.cpp scenario in the edge scenario. We modify the part of code  to support the SpecEE, and we will conduct the faster research iteration in the future.

# Demo
Left: SpecEE & Right: llama.cpp
<div align="center">
    <img src="./SpecEE.gif" alt="SpecEE" width="400">
    <img src="./llama.cpp.gif" alt="llama.cpp" width="400">
</div>

# SpecEE for Edge Scenario
please enter in the SpecEE_PC directory 

## Software Dependencies
torch~=2.2.1
numpy~=1.26.4
sentencepiece~=0.2.0
transformers>=4.45.1,<5.0.0
gguf>=0.1.0
protobuf>=4.21.0,<5.0.0


## Environment Set up
```bash
cd SpecEE_PC
conda create -n specee python=3.12
conda activate specee
pip install -r  ./requirements/requirements-convert_hf_to_gguf.txt
```
## Download model weights from Huggingface (Recommended)
```bash
huggingface-cli login
huggingface-cli download YYDH2333/SpecEE-7b-chat-hf
```

## Build model weights from scratch
1. Download [Llama-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and [EAGLE-llama2-chat-7B](https://huggingface.co/yuhuili/EAGLE-llama2-chat-7B)
```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-chat-hf
huggingface-cli download yuhuili/EAGLE-llama2-chat-7B
```
2. Copy Llama-7b-chat-hf weights to ./specee_weights
```bash
cp Llama-7b-chat-hf/config.json specee_weights/ specee_weights/
cp Llama-7b-chat-hf/generation_config.json specee_weights/
cp Llama-7b-chat-hf/model-00001-of-00002.safetensors specee_weights/
cp Llama-7b-chat-hf/model-00002-of-00002.safetensors specee_weights/
cp Llama-7b-chat-hf/special_tokens_map.json specee_weights/
cp Llama-7b-chat-hf/tokenizer.json specee_weights/
cp Llama-7b-chat-hf/tokenizer.model specee_weights/
cp Llama-7b-chat-hf/tokenizer_config.json specee_weights/
```
3. Run ./specee_weights/add_eagle.py
```bash
python ./specee_weights/add_eagle.py
```
4. Convert safetensors to gguf
```bash 
python convert_hf_to_gguf.py Llama-2-7b-chat-hf --outtype f16 --outfile models/Llama-2-7b-chat-hf.gguf --verbose 
python convert_hf_to_gguf.py specee_weights --outtype f16 --outfile models/SpecEE-7b-chat-hf.gguf --verbose 
```

## Compilation
```bash
make GGML_CUDA=1 -j$(nproc)
```
## Running Example
```bash
CUDA_VISIBLE_DEVICES=0 \
./llama-cli \
-m models/SpecEE-7b-chat-hf.gguf \
-p "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions." \
-e -ngl 16 -t 4 -n 256 -c 512 -s 8 --top_k 0 --temp 0
```

## Eval on Alpaca dataset
```bash
python eval_on_alpaca_dataset.py
```

## Eval on GSM8k dataset
```bash
python eval_on_gsm8k_dataset.py
```

## Eval on HumanEval dataset
```bash
python eval_on_humaneval_dataset.py
```

## Eval on MT-Bench dataset
```bash
python eval_on_mt-bench_dataset.py
```

## Eval on commonsenseQA dataset
```bash
python eval_on_qa_dataset.py
```

## Eval on SUM dataset
```bash
python eval_on_sum_dataset.py
```

# SpecEE for Cloud Scenario
please enter in the SpecEE_cloud directory 

## Setup
First setup the environments.

We need to setup three environments to evaluate SpecEE in huggingface and awq.
### Huggingface + SpecEE
```bash
cd SpecEE-cloud
conda create -n SpecEE python==3.10
conda activate SpecEE
pip install -r requirements.txt
```

### Raw awq
```bash
cd SpecEE-cloud
conda create -n awq python==3.10
conda activate awq
pip install -r requirements_awq.txt
```

### Awq + SpecEE
```bash
cd SpecEE-cloud
conda create -n specee_awq python==3.10
conda activate specee_awq
pip install -r requirements_awq.txt
cd AutoAWQ-0.2.6
pip install -e .
```

Second download the models needed.
```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-chat-hf
huggingface-cli download TheBloke/Llama-2-7B-Chat-AWQ
huggingface-cli download yuhuili/EAGLE-llama2-chat-7B
```

## Evaluation

### Huggingface + SpecEE
In our  code, we can evaluate the speed and accuracy performance based on Huggingface.


Firstly, activate SpecEE environment.
```bash
conda activate SpecEE
```

The example command to evaluate speed.


```
CUDA_VISIBLE_DEVICES=0 python EEInference.py --base-model-path meta-llama/Llama-2-7b-chat-hf --draft-model-path yuhuili/EAGLE-llama2-chat-7B --dataset mt_bench --task speed --predictor-path [the local path of ./llama-7b] 
```

The example command to evaluate accuracy.
```
CUDA_VISIBLE_DEVICES=0 python EEInference.py --base-model-path meta-llama/Llama-2-7b-chat-hf --draft-model-path yuhuili/EAGLE-llama2-chat-7B --dataset sst2 --task accuracy --predictor-path [the local path of ./llama-7b]  
```

You can replace the model path information in run_hf_specee.sh to complete all AE experiments replication in one go.
```bash
./run_hf_specee.sh
```
### Awq + SpecEE
In our AE code, we can evaluate the speed and accuracy performance based on awq.
### Accuracy
Firstly, activate SpecEE environment.
```bash
conda activate SpecEE
```
The example command.
```
CUDA_VISIBLE_DEVICES=0 python EEInference_awq.py --base-model-path TheBloke/Llama-2-7B-Chat-AWQ --draft-model-path yuhuili/EAGLE-llama2-chat-7B --dataset commonsenseqa --task accuracy --predictor-path [the local path of ./llama-7b]  
```
### Speed
Firstly, activate awq environment.
```bash
conda activate awq
```
You can get all awq speed via:
```bash
CUDA_VISIBLE_DEVICES=0 python AwqInference.py --base-model-path TheBloke/Llama-2-7B-Chat-AWQ 
```

Next, activate specee_awq environment.
```bash
conda activate specee_awq
```
You can get all awq+specee speed via:
```bash
CUDA_VISIBLE_DEVICES=0 python AwqEEInference.py --base-model-path TheBloke/Llama-2-7B-Chat-AWQ --draft-model-path yuhuili/EAGLE-llama2-chat-7B
```
The speed evaluation results of awq and awq+specee are in raw_awq.json and specee_awq.json.

You can run calculate_awq_speedup.py to get speedup ratio of awq+specee.
```
> python calculate_awq_speedup.py
```
