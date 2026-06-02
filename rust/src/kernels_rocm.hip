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

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void conv2d_kernel(const float* input, const float* weight, float* output,
                              int N, int C, int H, int W,
                              int K, int kh, int kw,
                              int sy, int sx, int py, int px,
                              int out_h, int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * K * out_h * out_w;
    if (idx < total_elements) {
        int ow = idx % out_w;
        int oh = (idx / out_w) % out_h;
        int ok = (idx / (out_w * out_h)) % K;
        int on = idx / (out_w * out_h * K);

        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            for (int ky = 0; ky < kh; ++ky) {
                int iy = oh * sy - py + ky;
                for (int kx = 0; kx < kw; ++kx) {
                    int ix = ow * sx - px + kx;
                    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                        int input_idx = ((on * C + c) * H + iy) * W + ix;
                        int weight_idx = ((ok * C + c) * kh + ky) * kw + kx;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

extern "C" void launch_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    hipLaunchKernelGGL(gemm_kernel, dim3(blocks), dim3(threads), 0, 0, A, B, C, M, N, K);
    hipDeviceSynchronize();
}

extern "C" void launch_conv2d(const float* input, const float* weight, float* output,
                              int N, int C, int H, int W,
                              int K, int kh, int kw,
                              int sy, int sx, int py, int px,
                              int out_h, int out_w) {
    int total_elements = N * K * out_h * out_w;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    hipLaunchKernelGGL(conv2d_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                       input, weight, output, N, C, H, W, K, kh, kw, sy, sx, py, px, out_h, out_w);
    hipDeviceSynchronize();
}
