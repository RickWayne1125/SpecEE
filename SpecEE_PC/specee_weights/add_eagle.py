import torch
import torch.nn as nn
from safetensors.torch import save_file

eagle_model_path = './EAGLE-llama2-chat-7B/pytorch_model.bin'
state_dict = torch.load(eagle_model_path, map_location=torch.device('cpu'))

# Parameter name: layers.0.self_attn.q_proj.weight, Shape: torch.Size([4096, 4096])
# Parameter name: layers.0.self_attn.k_proj.weight, Shape: torch.Size([4096, 4096])
# Parameter name: layers.0.self_attn.v_proj.weight, Shape: torch.Size([4096, 4096])
# Parameter name: layers.0.self_attn.o_proj.weight, Shape: torch.Size([4096, 4096])
# Parameter name: layers.0.mlp.gate_proj.weight, Shape: torch.Size([11008, 4096])
# Parameter name: layers.0.mlp.up_proj.weight, Shape: torch.Size([11008, 4096])
# Parameter name: layers.0.mlp.down_proj.weight, Shape: torch.Size([4096, 11008])
# Parameter name: layers.0.post_attention_layernorm.weight, Shape: torch.Size([4096])
# Parameter name: embed_tokens.weight, Shape: torch.Size([32000, 4096])
# Parameter name: fc.weight, Shape: torch.Size([4096, 8192])
# Parameter name: fc.bias, Shape: torch.Size([4096])

tensor_dict = {}
for name, param in state_dict.items():
    tensor_dict[name] = param

# 定义保存的文件路径
output_file = "specee_weights/model-eagle.safetensors"

# 保存为 safetensors 格式
save_file(tensor_dict, output_file)
print(f"Tensor weights saved as {output_file}")

