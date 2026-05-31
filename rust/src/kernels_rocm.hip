#include <hip/hip_runtime.h>

__global__ void fast_relu_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (x[idx] < 0.0f) {
            x[idx] = 0.0f;
        }
    }
}

__global__ void block_diagonal_attn_bias_kernel(float* attn_mask, int seq_len, int block_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < seq_len && col < seq_len) {
        int row_block = row / block_size;
        int col_block = col / block_size;
        if (row_block != col_block) {
            attn_mask[row * seq_len + col] = -1e9f; // Mask out non-diagonal blocks
        }
    }
}

extern "C" void launch_fast_relu(float* x, int size) {
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    hipLaunchKernelGGL(fast_relu_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0, x, size);
    hipDeviceSynchronize();
}

extern "C" void launch_block_diagonal_attn_bias(float* attn_mask, int seq_len, int block_size) {
    dim3 threads(16, 16);
    dim3 blocks((seq_len + 15) / 16, (seq_len + 15) / 16);
    hipLaunchKernelGGL(block_diagonal_attn_bias_kernel, dim3(blocks), dim3(threads), 0, 0, attn_mask, seq_len, block_size);
    hipDeviceSynchronize();
}
