# SpecEE

## Note
This repository is a reproduction of the llama.cpp portion in the paper's Figure 16: (a) The speedup and throughput of Llama2-7B on a Lenovo PC compared with llama.cpp .

### Hardware Dependencies
Lenovo Legion Y7000 with i7-13650HX CPU and NVIDIA RTX 4060 Laptop 8GB GPU for PC

### Software Dependencies
torch~=2.2.1
numpy~=1.26.4
sentencepiece~=0.2.0
transformers>=4.45.1,<5.0.0
gguf>=0.1.0
protobuf>=4.21.0,<5.0.0

## Usage
### Environment Set up
```bash
cd SpecEE_PC
conda create -n specee python=3.12
conda activate specee
pip install -r  ./requirements/requirements-convert_hf_to_gguf.txt
```
### Download model weights from Huggingface (Recommended)
```bash
huggingface-cli login
huggingface-cli download YYDH2333/SpecEE-7b-chat-hf
```

### Build model weights from scratch
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

### Compilation
```bash
make GGML_CUDA=1 -j$(nproc)
```
### Running Example
```bash
CUDA_VISIBLE_DEVICES=0 \
./llama-cli \
-m models/SpecEE-7b-chat-hf.gguf \
-p "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions." \
-e -ngl 16 -t 4 -n 256 -c 512 -s 8 --top_k 0 --temp 0
```

### Eval on Alpaca dataset
```bash
python eval_on_alpaca_dataset.py
```

### Eval on GSM8k dataset
```bash
python eval_on_gsm8k_dataset.py
```

### Eval on HumanEval dataset
```bash
python eval_on_humaneval_dataset.py
```

### Eval on MT-Bench dataset
```bash
python eval_on_mt-bench_dataset.py
```

### Eval on commonsenseQA dataset
```bash
python eval_on_qa_dataset.py
```

### Eval on SUM dataset
```bash
python eval_on_sum_dataset.py
```

## Code Introduction
### Model Components
In short, the model weights of specee consist of three parts: LLM, EAGLE, predictor.

Specifically, we use [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [EAGLE-llama2-chat-7B](https://huggingface.co/yuhuili/EAGLE-llama2-chat-7B), and our own trained predictor (weights in ./autoee).

### Data Flow
#### Prefill Phase
The LLM pre-fills the prompt of n tokens and predicts the next token.

EAGLE receives the original n tokens prompt plus the next token predicted by LLM, making it n+1 tokens in total.

EAGLE discards the 0th token and uses the remaining n tokens to complete the prefill and predict the next token.

It derives the final logits and finds the token_id of the top 4 words with the highest logit.

The predictor does not make predictions during the prefill phase.

#### Decode Phase
Pass the token_id of the top 4 words with the highest logits into the LLM.

The LLM decodes and gets the next token.

During decoding, the predictor judges whether to exit early based on the current layer's hidden_states and previously passed lm_head.

EAGLE receives the token predicted by LLM and predicts the next token.

## Changes in the Code
### build_eauto() in llama.cpp
This function builds the computation graph for specee.

### ggml_backend_sched_compute_splits() in ggml/src/ggml-backend.cpp
This is the function for executing the computation graph, running all ops in a for loop.

We execute all ops up to pred-il each time to get the value of pred.

If pred<0.5 (pred_greater_than_threshold returns false), indicating no exit, we skip the subsequent two OPs: MUL and ARGMAX.

If pred>0.5, we execute the following two OPs: multiply lm_head, argmax to get token_id,

fast_find_target_tensor checks if the currently predicted token_id is among the 4 candidate word token_ids passed in.

If it is included, exit, i.e., do not perform several layers of calculations behind the LLM, directly find the OP calculated by EAGLE and execute.

If not, continue with the original loop.

is_debug controls whether to print intermediate tensors.

### llm_load_tensors() in llama.cpp
Here, the weights of EAGLE and predictor are hung onto the output context and ensure that the weights are on the GPU.

Adjusted the original offload gpu method, placing the first few layers on the GPU and the last few layers on the CPU.

### llama_decode_internal() in llama.cpp
After completing the execution of the computation graph each time,
get logits from result_output tensor
```cpp
} else if (model.arch == LLM_ARCH_EAUTO){
    for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
        if (strcmp(ggml_graph_node(gf, i)->name, "result_output") == 0) {
            res = ggml_graph_node(gf, i);
            break;
        }
    }
```
Get candidate_token_ids from backend, which are the predicted four words.


### llama_set_eauto_token_intputs() in llama.cpp
Here, we pass the four candidate words predicted by EAGLE in the previous round into the current LLM input.
```cpp
tokens_data[0] = lctx.candidate_token_ids_last[0];
tokens_data[1] = lctx.candidate_token_ids_last[1];
tokens_data[2] = lctx.candidate_token_ids_last[2];
tokens_data[3] = lctx.candidate_token_ids_last[3];
```