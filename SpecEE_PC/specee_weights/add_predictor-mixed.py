import json
import torch
import torch.nn as nn
from safetensors.torch import save_file


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


hidden_dim = 4096
n_layers = 32

predictors = [torch.load(f'/share/public/LLMs/Llama-2-7b-chat-eauto/mixed/model{i}.pth', map_location=torch.device('cpu')).to(torch.float16) for i in range(n_layers)]
    
tensor_dict = {}
for i in range(n_layers):
    tensor_dict[f'model.layers.{i}.pred.up.weight'] = predictors[i].state_dict()['fc1.weight']
    tensor_dict[f'model.layers.{i}.pred.up.bias'] = predictors[i].state_dict()['fc1.bias']
    tensor_dict[f'model.layers.{i}.pred.down.weight'] = predictors[i].state_dict()['fc2.weight']
    tensor_dict[f'model.layers.{i}.pred.down.bias'] = predictors[i].state_dict()['fc2.bias']

    print('fc1.weight',tensor_dict[f'model.layers.{i}.pred.up.weight'].shape)
    print('fc2.bias',tensor_dict[f'model.layers.{i}.pred.up.bias'].shape)
    print('fc1.weight',tensor_dict[f'model.layers.{i}.pred.down.weight'].shape)
    print('fc2.bias',tensor_dict[f'model.layers.{i}.pred.down.bias'].shape)

output_file = "specee_weights/model-predictor-mixed.safetensors"

save_file(tensor_dict, output_file)
print(f"Tensor weights saved as {output_file}")



safetensor_json_path = "/share/public/LLMs/ReluLLaMA-7B/pytorch_model.bin.index.json"

with open(safetensor_json_path, "r", encoding="utf-8") as f:
    safetensor_json = json.load(f)
    print(safetensor_json)
    predictor_safetensor_json = {key:output_file for key in tensor_dict.keys()}
    safetensor_json['weight_map'].update(predictor_safetensor_json)

save_safetensor_json_path = '/share/public/LLMs/ReluLLaMA-7B/pytorch_model.bin.index.json'

with open(save_safetensor_json_path, "w", encoding="utf-8") as f:
    json.dump(safetensor_json, f, ensure_ascii=False, indent=4)
print(safetensor_json)
print(f"Updated GGUF file saved to {save_safetensor_json_path}")

