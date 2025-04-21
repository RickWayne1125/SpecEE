#include "common.cuh"
#include "argmax.cuh"
#include "sum.cuh"

#include <cstdint>

#define BLOCK_SIZE 256
#define GRID_SIZE 16

static __global__ void argmax_f32(
    const float * x, int32_t * dst, const int64_t ncols, const int64_t nrows) {

    int argmax_thread = 0;
    const int64_t row0 = (int64_t)blockIdx.x*WARP_SIZE;

#pragma unroll
    for (int64_t row1 = 0; row1 < WARP_SIZE; ++row1) {
        const int64_t row = row0 + row1;

        if (row >= nrows) {
            break;
        }

        float maxval = -FLT_MAX;
        int   argmax = -1;

        for (int32_t col = threadIdx.x; col < ncols; col += WARP_SIZE) {
            const float val        = x[row*ncols + col];
            const int   bigger     = val > maxval;
            const int   not_bigger = bigger ^ 0x00000001;

            maxval = maxval*not_bigger + val*bigger;
            argmax = argmax*not_bigger + col*bigger;
        }

#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float val        = __shfl_xor_sync(0xFFFFFFFF, maxval, mask, WARP_SIZE);
            const int   col        = __shfl_xor_sync(0xFFFFFFFF, argmax, mask, WARP_SIZE);
            const int   bigger     = val > maxval;
            const int   not_bigger = bigger ^ 0x00000001;

            maxval = maxval*not_bigger + val*bigger;
            argmax = argmax*not_bigger + col*bigger;
        }

        const int store = row1 == threadIdx.x;
        argmax_thread += store*argmax;
    }

    const int row = row0 + threadIdx.x;

    if (row >= nrows) {
        return;
    }

    dst[row] = argmax_thread;
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t num_blocks = (nrows + WARP_SIZE - 1) / WARP_SIZE;

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00, nrows);
}

static __global__ void argmax_f32(const float * x, int * dst, const int ncols) {
    int col = threadIdx.x;
    int row = blockIdx.y;
    const float * x_row = x + row * ncols;
    int * dst_row = dst + row * GRID_SIZE;

    __shared__ int ken[BLOCK_SIZE];
    int top_idx = INT_MIN;
    for (int idx = col + blockDim.x * blockIdx.x; idx < ncols; idx += blockDim.x * gridDim.x) {
        if ((top_idx == INT_MIN) || (x_row[top_idx] < x_row[idx])) {
            top_idx = idx;
        }
    }
    ken[col] = top_idx;
    __syncthreads();

    for (int i = BLOCK_SIZE/2; i >= 1; i /= 2) {
        if ((col < i) && (ken[col+i] != INT_MIN) && ((ken[col] == INT_MIN) || (x_row[ken[col]] < x_row[ken[col + i]]))) {
            ken[col] = ken[col + i];
        }
        __syncthreads();
    }
    if (col == 0) {
        dst_row[blockIdx.x] = ken[0];
    }
}

static __global__ void argmax_f32(const float * x, int * first_pass, int * dst, const int ncols) {
    int col = threadIdx.x;
    int row = blockIdx.y;
    const float * x_row = x + row * ncols;
    int * fp_row = first_pass + row * GRID_SIZE;
    int * dst_row = dst + row;

    __shared__ int ken[BLOCK_SIZE];
    ken[col] = INT_MIN;
    if (col < GRID_SIZE) {
        ken[col] = fp_row[col];
    }
    __syncthreads();

    for (int i = BLOCK_SIZE/2; i >= 1; i /= 2) {
        if ((col < i) && (ken[col+i] != INT_MIN) && ((ken[col] == INT_MIN) || (x_row[ken[col]] < x_row[ken[col + i]]))) {
            ken[col] = ken[col + i];
        }
        __syncthreads();
    }
    if (col == 0) {
        dst_row[0] = ken[0];
    }
}

static void argmax_f32_cuda(const float * x, int * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const dim3 block_dims(BLOCK_SIZE, 1, 1);
    const dim3 block_nums(GRID_SIZE, nrows, 1);
    const dim3 block_nums2(1, nrows, 1);
    const size_t shared_mem = BLOCK_SIZE * sizeof(int);

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    int * first_pass;
    cudaMalloc((void **) &first_pass, nrows * GRID_SIZE * sizeof(float));

    argmax_f32<<<block_nums, block_dims, shared_mem, stream>>>(x, first_pass, ncols);
    argmax_f32<<<block_nums2, block_dims, shared_mem, stream>>>(x, first_pass, dst, ncols);

    cudaFree(first_pass);
}

void ggml_cuda_op_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    argmax_f32_cuda(src0_d, (int *)dst_d, ncols, nrows, stream);
}
