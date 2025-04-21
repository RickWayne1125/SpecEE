from datasets import load_dataset


import re
import json
import shlex
from tqdm import tqdm
from subprocess import run, PIPE

# Set environment variable for CUDA device visibility
env = {'CUDA_VISIBLE_DEVICES':'2'}

# Configuration parameters
LLAMA_MODEL_PATH = "models/Llama-2-7b-chat-hf.gguf"  # Path to the original LLaMa model
SPECEE_MODEL_PATH = "models/SpecEE-7b-chat-hf.gguf"  # Path to the SpecEE model


def generate_baseline_command(prompt, model_path):
    """
    Generate a baseline command (using heredoc for safely handling long text inputs)
    """
    return [
        "./llama-cli",
        "-m", model_path,
        "-p", prompt, 
        "-n", "256", "-t", "4", "-e", "-s", "8",
        "--top_k", "0", "--temp", "0", "-ngl", "16","-ub","1024"
    ]


# Generate commands for both models based on prompts from JSON lines file
specee_commands = []
llama_commands = []
dataset = load_dataset("philschmid/mt-bench")['train'] 
for data in dataset:
    prompt = data['turns'][0][:1024]
    llama_commands.append(generate_baseline_command(prompt, LLAMA_MODEL_PATH))
    specee_commands.append(generate_baseline_command(prompt, SPECEE_MODEL_PATH))

llama_summary, specee_summary = {}, {}

# Evaluate both models and collect inference times and token counts
for model_name,commands in [('specee',specee_commands),('llama',llama_commands)]:
    print(f'Start evaluating on sum dataset for {model_name} model ...')
    results = []
    for i, cmd in tqdm(enumerate(commands), total=len(commands)):
        shell_cmd = " ".join(shlex.quote(arg) for arg in cmd)
        # print(shell_cmd)
        result = run(cmd, stdout=PIPE, stderr=PIPE, text=True, env=env,timeout=300)
        
        # Use regex to find relevant performance information
        pattern = r'llama_perf_context_print:\s*total time\s*=\s*(\d+\.?\d*)\s*ms\s*/\s*(\d+)\s*tokens'
        match = re.search(pattern, result.stderr)
        if match:
            inference_time = match.group(1)  # Extract inference time
            tokens = match.group(2)  # Extract number of tokens
        else:
            raise Exception("No match found in stderr output")
        
        results.append({
            "inference_time": inference_time,
            "tokens": tokens,
            "command": " ".join(cmd),
            "output": result.stdout,
            "log": result.stderr,
        })

    # Summarize inference time and tokens for each model
    total_time = sum(float(result['inference_time']) for result in results)
    total_tokens = sum(int(result['tokens']) for result in results)
    summary = {
        'Model': model_name,
        'Inference time': f'{total_time:.2f} ms',
        'Tokens': f'{total_tokens}', 
        'Tokens per second':f'{total_tokens/total_time*1000:.2f}'
    }
    # print(f'Model: {model_name}\nInference time: {total_time:.2f} ms\nTokens: {total_tokens} \n{total_time/total_tokens:.2f} ms per token')
    print(summary)
    results = [summary] + results
    if model_name == 'llama':
        llama_summary = summary
    else:
        specee_summary = summary
    
    # Save results to a JSON file
    with open(f'timing_results_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=4)

# Print comparison between models
print(f'LLAMA: {llama_summary} ms\nSpecEE: {specee_summary} ms\nSpeedup: {float(specee_summary["Tokens per second"])/float(llama_summary["Tokens per second"]):.2f}x')