# 令牌生成性能故障排除

## 验证模型是否在使用 CUDA 的 GPU 上运行
确保根据[此指南](/docs/build.md#cuda)正确设置了环境变量来编译 llama，以便 llama 接受 `-ngl N`（或 `--n-gpu-layers N`）参数。运行 llama 时，您可以将 `N` 设置得非常大，llama 会将尽可能多的层卸载到 GPU，即使卸载的层数少于您设置的数量。例如：
```shell
./llama-cli -m "path/to/model.gguf" -ngl 200000 -p "Please sir, may I have some "
```

运行 llama 时，在开始推理工作之前，它会输出诊断信息，显示 cuBLAS 是否正在将工作卸载到 GPU。请查找以下行：
```shell
llama_model_load_internal: [cublas] offloading 60 layers to GPU
llama_model_load_internal: [cublas] offloading output layer to GPU
llama_model_load_internal: [cublas] total VRAM used: 17223 MB
... rest of inference
```

如果看到这些行，则说明正在使用 GPU。

## 验证 CPU 是否未超负荷
llama 接受 `-t N`（或 `--threads N`）参数。确保此参数设置得不太大非常重要。如果您的令牌生成速度非常慢，尝试将此参数设置为 1。如果这显著提高了令牌生成速度，那么您的 CPU 正在超负荷，您需要明确将此参数设置为您机器的物理 CPU 核心数量（即使您使用 GPU）。如果不确定，从 1 开始，然后将数量加倍，直到遇到性能瓶颈，再减少数量。

## 运行时标志对推理速度基准的影响示例
以下测试运行在这台机器上：
- GPU：A6000（48GB VRAM）
- CPU：7 个物理核心
- RAM：32GB

模型：`TheBloke_Wizard-Vicuna-30B-Uncensored-GGML/Wizard-Vicuna-30B-Uncensored.q4_0.gguf`（30B 参数，4 位量化，GGML）

运行命令：`./llama-cli -m "path/to/model.gguf" -p "An extremely detailed description of the 10 best ethnic dishes will follow, with recipes: " -n 1000 [附加的基准标志]`

结果：

| 命令 | 令牌/秒（值越高越好） |
| - | - |
| -ngl 2000000 | N/A（少于 0.1） |
| -t 7 | 1.7 |
| -t 1 -ngl 2000000 | 5.5 |
| -t 7 -ngl 2000000 | 8.7 |
| -t 4 -ngl 2000000 | 9.1 |