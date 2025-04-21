#include "argsort.cuh"
#define BLOCK_SIZE 256
#define GRID_SIZE 16
#define K_TOPK 4

template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<ggml_sort_order order>
static __global__ void k_argsort_f32_i32(const float * x, int * dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.y;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void argsort_f32_i32_cuda(const float * x, int * dst, const int ncols, const int nrows, ggml_sort_order order, cudaStream_t stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(1, nrows, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    // auto code = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH_QUICKSORT);
    // GGML_ASSERT(code == cudaSuccess);

    // const dim3 block_nums(1, nrows, 1);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
        // quicksort<GGML_SORT_ORDER_ASC><<<block_nums, 1>>>(x, dst, 0, ncols-1, 1);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
        // quicksort<GGML_SORT_ORDER_DESC><<<block_nums, 1>>>(x, dst, 0, ncols-1, 1);
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_cuda(src0_d, (int *)dst_d, ncols, nrows, order, stream);
}

static __device__ __host__ void insert_value(const float * x, int * ken, int data) {
    if (data == INT_MIN) {
        return;
    }
    int i = K_TOPK-1;
    for (; i >= 0; --i) {
        if (ken[i] == INT_MIN) {
            continue;
        } else if (x[ken[i]] > x[data]) {
            break;
        } else if (i < K_TOPK-1) {
            ken[i+1] = ken[i];
        }
    }
    if (i < K_TOPK-1) {
        ken[i+1] = data;
    }
}

static __global__ void k_argtopk_f32_i32(const float * x, int * dst, const int ncols) {
    int col = threadIdx.x;
    int row = blockIdx.y;
    const float * x_row = x + row * ncols;
    int * dst_row = dst + row * GRID_SIZE * K_TOPK;

    __shared__ int ken[BLOCK_SIZE * K_TOPK];
    int top_array[K_TOPK];
    for (int i = 0; i < K_TOPK; ++i) {
        top_array[i] = INT_MIN;
    }
    for (int idx = col + blockDim.x * blockIdx.x; idx < ncols; idx += blockDim.x * gridDim.x) {
        insert_value(x_row, top_array, idx);
    }
    for (int i = 0; i < K_TOPK; ++i) {
        ken[K_TOPK * col + i] = top_array[i];
    }
    __syncthreads();

    for (int i = BLOCK_SIZE/2; i >= 1; i /= 2) {
        if (col < i) {
            for (int m = 0; m < K_TOPK; ++m) {
                insert_value(x_row, top_array, ken[K_TOPK * (col + i) + m]);
            }
        }
        __syncthreads();
        if (col < i) {
            for (int m = 0; m < K_TOPK; ++m) {
                ken[K_TOPK * col + m] = top_array[m];
            }
        }
        __syncthreads();
    }
    if (col == 0) {
        for (int i = 0; i < K_TOPK; ++i) {
            dst_row[K_TOPK * blockIdx.x + i] = ken[i];
        }
    }
}

static __global__ void k_argtopk_f32_i32(const float * x, int * first_pass, int * dst, const int ncols, int k) {
    int col = threadIdx.x;
    int row = blockIdx.y;
    const float * x_row = x + row * ncols;
    int * fp_row = first_pass + row * K_TOPK * GRID_SIZE;
    int * dst_row = dst + row * k;

    __shared__ int ken[BLOCK_SIZE * K_TOPK];
    int top_array[K_TOPK];
    for (int i = 0; i < K_TOPK; ++i) {
        top_array[i] = INT_MIN;
    }
    for (int idx = col; idx < GRID_SIZE * K_TOPK; idx += blockDim.x) {
        insert_value(x_row, top_array, fp_row[idx]);
    }
    for (int i = 0; i < K_TOPK; ++i) {
        ken[K_TOPK * col + i] = top_array[i];
    }
    __syncthreads();

    for (int i = BLOCK_SIZE/2; i >= 1; i /= 2) {
        if (col < i) {
            for (int m = 0; m < K_TOPK; ++m) {
                insert_value(x_row, top_array, ken[K_TOPK * (col + i) + m]);
            }
        }
        __syncthreads();
        if (col < i) {
            for (int m = 0; m < K_TOPK; ++m) {
                ken[K_TOPK * col + m] = top_array[m];
            }
        }
        __syncthreads();
    }
    if (col == 0) {
        for (int i = 0; i < k; ++i) {
            dst_row[i] = ken[i];
        }
    }
}

static void argtopk_f32_i32_cuda(const float * x, int * dst, const int ncols, const int nrows, cudaStream_t stream, int k) {
    const dim3 block_dims(BLOCK_SIZE, 1, 1);
    const dim3 block_nums(GRID_SIZE, nrows, 1);
    const dim3 block_nums2(1, nrows, 1);
    const size_t shared_mem = BLOCK_SIZE * K_TOPK * sizeof(int);

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);
    GGML_ASSERT(k <= 8);

    int * first_pass;
    cudaMalloc((void **) &first_pass, nrows * GRID_SIZE * K_TOPK * sizeof(float));

    k_argtopk_f32_i32<<<block_nums, block_dims, shared_mem, stream>>>(x, first_pass, ncols);
    k_argtopk_f32_i32<<<block_nums2, block_dims, shared_mem, stream>>>(x, first_pass, dst, ncols, k);

    cudaFree(first_pass);
}

void ggml_cuda_op_argtopk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    int k = dst->op_params[0];

    argtopk_f32_i32_cuda(src0_d, (int *)dst_d, ncols, nrows, stream, k);
}

// test

// int main() {
//     float x[32000];
//     int dst[4];
//     int k = 4, ncols = 32000;
//     int fp_size = GRID_SIZE * K_TOPK;
//     int fp[GRID_SIZE * K_TOPK];

//     for (int i = 0; i < 32000; ++i) {
//         x[i] = i;
//     }

//     float * x_d;
//     int * dst_d, * fp_d;

//     cudaMalloc((void**)&x_d, 32000*sizeof(float));
//     cudaMalloc((void**)&dst_d, 4*sizeof(int));
//     cudaMalloc((void**)&fp_d, fp_size*sizeof(int));

//     cudaMemcpy(x_d, x, 32000*sizeof(float), cudaMemcpyHostToDevice);

//     const dim3 block_dims(BLOCK_SIZE, 1, 1);
//     const dim3 block_nums(GRID_SIZE, 1, 1);
//     const dim3 block_nums2(1, 1, 1);

//     k_argtopk_f32_i32<<<block_nums, block_dims>>>(x_d, fp_d, ncols);
//     k_argtopk_f32_i32<<<block_nums2, block_dims>>>(x_d, fp_d, dst_d, ncols, k);

//     cudaMemcpy(dst, dst_d, 4*sizeof(int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(fp, fp_d, fp_size*sizeof(int), cudaMemcpyDeviceToHost);

//     cudaFree(x_d);
//     cudaFree(dst_d);
//     cudaFree(fp_d);

//     for (int te = 0; te < fp_size; ++te) {
//         printf("%d ", fp[te]);
//     }
//     printf("\n%d %d %d %d\n", dst[0], dst[1], dst[2], dst[3]);
// }
