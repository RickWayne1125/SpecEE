#include "common.cuh"

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_argtopk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
