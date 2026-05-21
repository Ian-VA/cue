#include "cuda_ops.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int THREADS = 256;

inline void cuda_check(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error (") + what +
                             "): " + cudaGetErrorString(err));
  }
}

inline int grid_size(size_t n, int block) {
  return static_cast<int>((n + block - 1) / block);
}

// element-wise operations

__global__ void add_k(const float *a, const float *b, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + b[i];
}

__global__ void sub_k(const float *a, const float *b, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] - b[i];
}

__global__ void mul_k(const float *a, const float *b, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * b[i];
}

__global__ void div_k(const float *a, const float *b, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] / b[i];
}

__global__ void add_scalar_k(const float *a, float s, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + s;
}

__global__ void mul_scalar_k(const float *a, float s, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * s;
}

// activations

__global__ void relu_k(const float *a, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = a[i];
    out[i] = v > 0.0f ? v : 0.0f;
  }
}

__global__ void relu_backward_k(const float *grad, const float *input,
                                float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = input[i] > 0.0f ? grad[i] : 0.0f;
}

__global__ void sigmoid_k(const float *a, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = 1.0f / (1.0f + __expf(-a[i]));
}

__global__ void tanh_k(const float *a, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = tanhf(a[i]);
}

__global__ void exp_k(const float *a, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = expf(a[i]);
}

__global__ void log_k(const float *a, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = logf(a[i]);
}

__global__ void softmax_last_dim_k(const float *in, float *out, size_t outer,
                                   size_t inner) {
  extern __shared__ float shared[];
  size_t row = blockIdx.x;
  if (row >= outer)
    return;
  const float *in_row = in + row * inner;
  float *out_row = out + row * inner;

  float local_max = -INFINITY;
  for (size_t i = threadIdx.x; i < inner; i += blockDim.x) {
    local_max = fmaxf(local_max, in_row[i]);
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
    }
    __syncthreads();
  }
  float row_max = shared[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (size_t i = threadIdx.x; i < inner; i += blockDim.x) {
    float e = expf(in_row[i] - row_max);
    out_row[i] = e;
    local_sum += e;
  }
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) {
      shared[threadIdx.x] += shared[threadIdx.x + s];
    }
    __syncthreads();
  }
  float row_sum = shared[0];

  for (size_t i = threadIdx.x; i < inner; i += blockDim.x) {
    out_row[i] /= row_sum;
  }
}

// linear algebra
__global__ void matmul_k(const float *A, const float *B, float *C, size_t M,
                         size_t K, size_t N) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N)
    return;

  float acc = 0.0f;
  for (size_t k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = acc;
}

__global__ void transpose2d_k(const float *in, float *out, size_t rows,
                              size_t cols) {
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows || j >= cols)
    return;
  out[j * rows + i] = in[i * cols + j];
}

// reductions

__global__ void sum_partial_k(const float *in, float *partials, size_t n) {
  extern __shared__ float sdata[];
  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * (blockDim.x * 2) + tid;

  float v = 0.0f;
  if (i < n)
    v += in[i];
  if (i + blockDim.x < n)
    v += in[i + blockDim.x];
  sdata[tid] = v;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if ((int)tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    partials[blockIdx.x] = sdata[0];
}

// conv2d
//
__global__ void conv2d_forward_k(const float *in, const float *ker, float *out,
                                 size_t N, size_t Cin, size_t H, size_t W,
                                 size_t Cout, size_t kH, size_t kW, size_t oH,
                                 size_t oW, size_t stride, size_t padding) {
  size_t total = N * Cout * oH * oW;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  size_t ow = idx % oW;
  size_t oh = (idx / oW) % oH;
  size_t co = (idx / (oW * oH)) % Cout;
  size_t n = idx / (oW * oH * Cout);

  float acc = 0.0f;
  for (size_t ci = 0; ci < Cin; ++ci) {
    for (size_t ki = 0; ki < kH; ++ki) {
      for (size_t kj = 0; kj < kW; ++kj) {
        long long ih =
            (long long)oh * stride + (long long)ki - (long long)padding;
        long long iw =
            (long long)ow * stride + (long long)kj - (long long)padding;
        if (ih < 0 || ih >= (long long)H)
          continue;
        if (iw < 0 || iw >= (long long)W)
          continue;
        size_t in_idx = ((n * Cin + ci) * H + (size_t)ih) * W + (size_t)iw;
        size_t k_idx = ((co * Cin + ci) * kH + ki) * kW + kj;
        acc += in[in_idx] * ker[k_idx];
      }
    }
  }
  out[idx] = acc;
}

// pooling

__global__ void max_pool2d_forward_k(const float *in, float *out, size_t N,
                                     size_t C, size_t H, size_t W, size_t kH,
                                     size_t kW, size_t oH, size_t oW,
                                     size_t stride, size_t padding) {
  size_t total = N * C * oH * oW;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  size_t ow = idx % oW;
  size_t oh = (idx / oW) % oH;
  size_t c = (idx / (oW * oH)) % C;
  size_t n = idx / (oW * oH * C);

  float m = -INFINITY;
  for (size_t ki = 0; ki < kH; ++ki) {
    for (size_t kj = 0; kj < kW; ++kj) {
      long long ih =
          (long long)oh * stride + (long long)ki - (long long)padding;
      long long iw =
          (long long)ow * stride + (long long)kj - (long long)padding;
      if (ih < 0 || ih >= (long long)H)
        continue;
      if (iw < 0 || iw >= (long long)W)
        continue;
      float v = in[((n * C + c) * H + (size_t)ih) * W + (size_t)iw];
      if (v > m)
        m = v;
    }
  }
  out[idx] = isinf(m) ? 0.0f : m;
}

__global__ void avg_pool2d_forward_k(const float *in, float *out, size_t N,
                                     size_t C, size_t H, size_t W, size_t kH,
                                     size_t kW, size_t oH, size_t oW,
                                     size_t stride, size_t padding) {
  size_t total = N * C * oH * oW;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  size_t ow = idx % oW;
  size_t oh = (idx / oW) % oH;
  size_t c = (idx / (oW * oH)) % C;
  size_t n = idx / (oW * oH * C);

  float acc = 0.0f;
  size_t count = 0;
  for (size_t ki = 0; ki < kH; ++ki) {
    for (size_t kj = 0; kj < kW; ++kj) {
      long long ih =
          (long long)oh * stride + (long long)ki - (long long)padding;
      long long iw =
          (long long)ow * stride + (long long)kj - (long long)padding;
      if (ih < 0 || ih >= (long long)H)
        continue;
      if (iw < 0 || iw >= (long long)W)
        continue;
      acc += in[((n * C + c) * H + (size_t)ih) * W + (size_t)iw];
      ++count;
    }
  }
  out[idx] = count > 0 ? acc / (float)count : 0.0f;
}

// channel-wise bias add

__global__ void bias_add_channel_k(const float *in, const float *bias,
                                   float *out, size_t N, size_t C, size_t HW) {
  size_t total = N * C * HW;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;
  size_t c = (idx / HW) % C;
  out[idx] = in[idx] + bias[c];
}

} // namespace

namespace cue::cuda {

// memory
void *malloc_bytes(size_t bytes) {
  void *p = nullptr;
  cuda_check(cudaMalloc(&p, bytes), "malloc");
  return p;
}

void free_bytes(void *ptr) {
  if (ptr)
    cudaFree(ptr);
}

void memcpy_h2d(void *dst, const void *src, size_t bytes) {
  cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "memcpy_h2d");
}

void memcpy_d2h(void *dst, const void *src, size_t bytes) {
  cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "memcpy_d2h");
}

void memcpy_d2d(void *dst, const void *src, size_t bytes) {
  cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice),
             "memcpy_d2d");
}

void memset_zero(void *dst, size_t bytes) {
  cuda_check(cudaMemset(dst, 0, bytes), "memset_zero");
}

void device_synchronize() {
  cuda_check(cudaDeviceSynchronize(), "device_synchronize");
}

// element-wise operations

void add(const float *a, const float *b, float *out, size_t n) {
  add_k<<<grid_size(n, THREADS), THREADS>>>(a, b, out, n);
  cuda_check(cudaGetLastError(), "add");
}

void sub(const float *a, const float *b, float *out, size_t n) {
  sub_k<<<grid_size(n, THREADS), THREADS>>>(a, b, out, n);
  cuda_check(cudaGetLastError(), "sub");
}

void mul(const float *a, const float *b, float *out, size_t n) {
  mul_k<<<grid_size(n, THREADS), THREADS>>>(a, b, out, n);
  cuda_check(cudaGetLastError(), "mul");
}

void div(const float *a, const float *b, float *out, size_t n) {
  div_k<<<grid_size(n, THREADS), THREADS>>>(a, b, out, n);
  cuda_check(cudaGetLastError(), "div");
}

void add_scalar(const float *a, float s, float *out, size_t n) {
  add_scalar_k<<<grid_size(n, THREADS), THREADS>>>(a, s, out, n);
  cuda_check(cudaGetLastError(), "add_scalar");
}

void mul_scalar(const float *a, float s, float *out, size_t n) {
  mul_scalar_k<<<grid_size(n, THREADS), THREADS>>>(a, s, out, n);
  cuda_check(cudaGetLastError(), "mul_scalar");
}

// activations

void relu(const float *a, float *out, size_t n) {
  relu_k<<<grid_size(n, THREADS), THREADS>>>(a, out, n);
  cuda_check(cudaGetLastError(), "relu");
}

void relu_backward(const float *grad, const float *input, float *out,
                   size_t n) {
  relu_backward_k<<<grid_size(n, THREADS), THREADS>>>(grad, input, out, n);
  cuda_check(cudaGetLastError(), "relu_backward");
}

void sigmoid(const float *a, float *out, size_t n) {
  sigmoid_k<<<grid_size(n, THREADS), THREADS>>>(a, out, n);
  cuda_check(cudaGetLastError(), "sigmoid");
}

void tanh_act(const float *a, float *out, size_t n) {
  tanh_k<<<grid_size(n, THREADS), THREADS>>>(a, out, n);
  cuda_check(cudaGetLastError(), "tanh");
}

void exp_act(const float *a, float *out, size_t n) {
  exp_k<<<grid_size(n, THREADS), THREADS>>>(a, out, n);
  cuda_check(cudaGetLastError(), "exp");
}

void log_act(const float *a, float *out, size_t n) {
  log_k<<<grid_size(n, THREADS), THREADS>>>(a, out, n);
  cuda_check(cudaGetLastError(), "log");
}

void softmax_last_dim(const float *in, float *out, size_t outer, size_t inner) {
  softmax_last_dim_k<<<(int)outer, THREADS, THREADS * sizeof(float)>>>(
      in, out, outer, inner);
  cuda_check(cudaGetLastError(), "softmax_last_dim");
}

// linear algebra
void matmul(const float *a, const float *b, float *out, size_t m, size_t k,
            size_t n) {
  dim3 block(16, 16);
  dim3 grid(static_cast<unsigned>((n + 15) / 16),
            static_cast<unsigned>((m + 15) / 16));
  matmul_k<<<grid, block>>>(a, b, out, m, k, n);
  cuda_check(cudaGetLastError(), "matmul");
}

void transpose2d(const float *a, float *out, size_t rows, size_t cols) {
  dim3 block(16, 16);
  dim3 grid(static_cast<unsigned>((cols + 15) / 16),
            static_cast<unsigned>((rows + 15) / 16));
  transpose2d_k<<<grid, block>>>(a, out, rows, cols);
  cuda_check(cudaGetLastError(), "transpose2d");
}

// reductions
float sum(const float *a, size_t n) {
  if (n == 0)
    return 0.0f;
  int threads = THREADS;
  int blocks =
      static_cast<int>((n + (size_t)threads * 2 - 1) / ((size_t)threads * 2));

  float *partials = nullptr;
  cuda_check(cudaMalloc(&partials, blocks * sizeof(float)), "sum.alloc");
  sum_partial_k<<<blocks, threads, threads * sizeof(float)>>>(a, partials, n);
  cuda_check(cudaGetLastError(), "sum.partial");

  std::vector<float> host(blocks);
  cuda_check(cudaMemcpy(host.data(), partials, blocks * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "sum.d2h");
  cudaFree(partials);

  double total = 0.0;
  for (float v : host)
    total += v;
  return static_cast<float>(total);
}

// conv / pool

void conv2d_forward(const float *input, const float *kernel, float *out,
                    size_t N, size_t Cin, size_t H, size_t W, size_t Cout,
                    size_t kH, size_t kW, size_t stride, size_t padding) {
  size_t oH = (H + 2 * padding - kH) / stride + 1;
  size_t oW = (W + 2 * padding - kW) / stride + 1;
  size_t total = N * Cout * oH * oW;
  conv2d_forward_k<<<grid_size(total, THREADS), THREADS>>>(
      input, kernel, out, N, Cin, H, W, Cout, kH, kW, oH, oW, stride, padding);
  cuda_check(cudaGetLastError(), "conv2d");
}

void max_pool2d_forward(const float *input, float *out, size_t N, size_t C,
                        size_t H, size_t W, size_t kH, size_t kW, size_t stride,
                        size_t padding) {
  size_t oH = (H + 2 * padding - kH) / stride + 1;
  size_t oW = (W + 2 * padding - kW) / stride + 1;
  size_t total = N * C * oH * oW;
  max_pool2d_forward_k<<<grid_size(total, THREADS), THREADS>>>(
      input, out, N, C, H, W, kH, kW, oH, oW, stride, padding);
  cuda_check(cudaGetLastError(), "max_pool2d");
}

void avg_pool2d_forward(const float *input, float *out, size_t N, size_t C,
                        size_t H, size_t W, size_t kH, size_t kW, size_t stride,
                        size_t padding) {
  size_t oH = (H + 2 * padding - kH) / stride + 1;
  size_t oW = (W + 2 * padding - kW) / stride + 1;
  size_t total = N * C * oH * oW;
  avg_pool2d_forward_k<<<grid_size(total, THREADS), THREADS>>>(
      input, out, N, C, H, W, kH, kW, oH, oW, stride, padding);
  cuda_check(cudaGetLastError(), "avg_pool2d");
}

// bias
void bias_add_channel(const float *input, const float *bias, float *out,
                      size_t N, size_t C, size_t HW) {
  size_t total = N * C * HW;
  bias_add_channel_k<<<grid_size(total, THREADS), THREADS>>>(input, bias, out,
                                                             N, C, HW);
  cuda_check(cudaGetLastError(), "bias_add_channel");
}

} // namespace cue::cuda
