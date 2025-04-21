conda activate awq
CUDA_VISIBLE_DEVICES=0 python AwqInference.py
conda activate specee_awq
CUDA_VISIBLE_DEVICES=0 python AwqEEInference.py
python calculate_awq_speedup.py