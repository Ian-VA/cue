#include "cuda_ops.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int THREADS = 256;
constexpr int TILE = 16;

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

// each output element decodes its coordinates from the flat index then walks the
// per input strides which are 0 on broadcast axes so a size 1 dim reads the same
// element repeatedly
__global__ void broadcast_binop_k(const float *a, const float *b, float *out,
                                  size_t n, cue::cuda::BroadcastDims d, int op) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  size_t rem = idx, a_off = 0, b_off = 0;
  for (int i = d.rank - 1; i >= 0; --i) {
    size_t coord = rem % d.out_shape[i];
    rem /= d.out_shape[i];
    a_off += coord * d.a_stride[i];
    b_off += coord * d.b_stride[i];
  }

  float av = a[a_off], bv = b[b_off];
  float r = 0.0f;
  switch (op) {
  case 0: r = av + bv; break;
  case 1: r = av - bv; break;
  case 2: r = av * bv; break;
  case 3: r = av / bv; break;
  }
  out[idx] = r;
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

// binary tree reduction sums
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

// tiled matmul each block loads a TILE x TILE square of A and B into shared
// memory and reuses it across the inner product so global memory is read once
// per tile instead of once per multiply
__global__ void matmul_k(const float *A, const float *B, float *C, size_t M,
                         size_t K, size_t N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  size_t row = (size_t)blockIdx.y * TILE + threadIdx.y;
  size_t col = (size_t)blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;
  size_t tiles = (K + TILE - 1) / TILE;
  for (size_t t = 0; t < tiles; ++t) {
    size_t a_col = t * TILE + threadIdx.x;
    size_t b_row = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE; ++k)
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = acc;
}

// tiled transpose stages a TILE x TILE block in shared memory so both the read
// and the write to global memory are coalesced the +1 pads each row to avoid
// shared memory bank conflicts
__global__ void transpose2d_k(const float *in, float *out, size_t rows,
                              size_t cols) {
  __shared__ float tile[TILE][TILE + 1];

  size_t x = (size_t)blockIdx.x * TILE + threadIdx.x;
  size_t y = (size_t)blockIdx.y * TILE + threadIdx.y;
  if (x < cols && y < rows)
    tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  __syncthreads();

  size_t tx = (size_t)blockIdx.y * TILE + threadIdx.x;
  size_t ty = (size_t)blockIdx.x * TILE + threadIdx.y;
  if (tx < rows && ty < cols)
    out[ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
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

__global__ void sum_to_channel_k(const float *in, float *out, size_t N,
                                 size_t C, size_t HW) {
  size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C)
    return;
  float acc = 0.0f;
  for (size_t n = 0; n < N; ++n) {
    const float *row = in + (n * C + c) * HW;
    for (size_t i = 0; i < HW; ++i)
      acc += row[i];
  }
  out[c] = acc;
}

__global__ void sum_axis0_k(const float *in, float *out, size_t N, size_t M) {
  size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= M)
    return;
  float acc = 0.0f;
  for (size_t i = 0; i < N; ++i)
    acc += in[i * M + j];
  out[j] = acc;
}

__global__ void gather_rows_k(const float *src, const size_t *idx, float *out,
                              size_t n, size_t per_sample) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * per_sample)
    return;
  size_t row = t / per_sample;
  size_t col = t % per_sample;
  out[t] = src[idx[row] * per_sample + col];
}

// conv2d / pool backwards

__global__ void conv2d_backward_input_k(const float *grad_out, const float *ker,
                                        float *grad_in, size_t N, size_t Cin,
                                        size_t H, size_t W, size_t Cout,
                                        size_t kH, size_t kW, size_t oH,
                                        size_t oW, size_t stride,
                                        size_t padding) {
  size_t total = N * Cin * H * W;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  size_t iw = idx % W;
  size_t ih = (idx / W) % H;
  size_t ci = (idx / (W * H)) % Cin;
  size_t n = idx / (W * H * Cin);

  float acc = 0.0f;
  for (size_t co = 0; co < Cout; ++co) {
    for (size_t ki = 0; ki < kH; ++ki) {
      for (size_t kj = 0; kj < kW; ++kj) {
        long long num_h = (long long)ih + (long long)padding - (long long)ki;
        long long num_w = (long long)iw + (long long)padding - (long long)kj;
        if (num_h < 0 || num_w < 0)
          continue;
        if ((size_t)num_h % stride != 0 || (size_t)num_w % stride != 0)
          continue;
        size_t oh = (size_t)num_h / stride;
        size_t ow = (size_t)num_w / stride;
        if (oh >= oH || ow >= oW)
          continue;
        size_t g_idx = ((n * Cout + co) * oH + oh) * oW + ow;
        size_t k_idx = ((co * Cin + ci) * kH + ki) * kW + kj;
        acc += grad_out[g_idx] * ker[k_idx];
      }
    }
  }
  grad_in[idx] = acc;
}

__global__ void conv2d_backward_kernel_k(const float *grad_out, const float *in,
                                         float *grad_ker, size_t N, size_t Cin,
                                         size_t H, size_t W, size_t Cout,
                                         size_t kH, size_t kW, size_t oH,
                                         size_t oW, size_t stride,
                                         size_t padding) {
  size_t total = Cout * Cin * kH * kW;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  size_t kj = idx % kW;
  size_t ki = (idx / kW) % kH;
  size_t ci = (idx / (kW * kH)) % Cin;
  size_t co = idx / (kW * kH * Cin);

  float acc = 0.0f;
  for (size_t n = 0; n < N; ++n) {
    for (size_t oh = 0; oh < oH; ++oh) {
      for (size_t ow = 0; ow < oW; ++ow) {
        long long ih =
            (long long)oh * stride + (long long)ki - (long long)padding;
        long long iw =
            (long long)ow * stride + (long long)kj - (long long)padding;
        if (ih < 0 || ih >= (long long)H)
          continue;
        if (iw < 0 || iw >= (long long)W)
          continue;
        size_t in_idx = ((n * Cin + ci) * H + (size_t)ih) * W + (size_t)iw;
        size_t g_idx = ((n * Cout + co) * oH + oh) * oW + ow;
        acc += in[in_idx] * grad_out[g_idx];
      }
    }
  }
  grad_ker[idx] = acc;
}

__global__ void max_pool2d_backward_k(const float *grad_out, const float *in,
                                      float *grad_in, size_t N, size_t C,
                                      size_t H, size_t W, size_t kH, size_t kW,
                                      size_t oH, size_t oW, size_t stride,
                                      size_t padding) {
  size_t total = N * C * oH * oW;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  size_t ow = idx % oW;
  size_t oh = (idx / oW) % oH;
  size_t c = (idx / (oW * oH)) % C;
  size_t n = idx / (oW * oH * C);

  float m = -INFINITY;
  long long best_ih = -1, best_iw = -1;
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
      if (v > m) {
        m = v;
        best_ih = ih;
        best_iw = iw;
      }
    }
  }
  if (best_ih < 0)
    return;
  size_t in_idx = ((n * C + c) * H + (size_t)best_ih) * W + (size_t)best_iw;
  atomicAdd(&grad_in[in_idx], grad_out[idx]);
}

__global__ void avg_pool2d_backward_k(const float *grad_out, float *grad_in,
                                      size_t N, size_t C, size_t H, size_t W,
                                      size_t kH, size_t kW, size_t oH,
                                      size_t oW, size_t stride,
                                      size_t padding) {
  size_t total = N * C * oH * oW;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  size_t ow = idx % oW;
  size_t oh = (idx / oW) % oH;
  size_t c = (idx / (oW * oH)) % C;
  size_t n = idx / (oW * oH * C);

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
      ++count;
    }
  }
  if (count == 0)
    return;
  float share = grad_out[idx] / (float)count;
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
      size_t in_idx = ((n * C + c) * H + (size_t)ih) * W + (size_t)iw;
      atomicAdd(&grad_in[in_idx], share);
    }
  }
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

void broadcast_binop(const float *a, const float *b, float *out, size_t n,
                     const BroadcastDims &dims, BinOp op) {
  broadcast_binop_k<<<grid_size(n, THREADS), THREADS>>>(a, b, out, n, dims,
                                                        (int)op);
  cuda_check(cudaGetLastError(), "broadcast_binop");
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
  dim3 block(TILE, TILE);
  dim3 grid(grid_size(n, TILE), grid_size(m, TILE));
  matmul_k<<<grid, block>>>(a, b, out, m, k, n);
  cuda_check(cudaGetLastError(), "matmul");
}

void transpose2d(const float *a, float *out, size_t rows, size_t cols) {
  dim3 block(TILE, TILE);
  dim3 grid(grid_size(cols, TILE), grid_size(rows, TILE));
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

void conv2d_backward_input(const float *grad_out, const float *kernel,
                           float *grad_in, size_t N, size_t Cin, size_t H,
                           size_t W, size_t Cout, size_t kH, size_t kW,
                           size_t stride, size_t padding) {
  size_t oH = (H + 2 * padding - kH) / stride + 1;
  size_t oW = (W + 2 * padding - kW) / stride + 1;
  size_t total = N * Cin * H * W;
  conv2d_backward_input_k<<<grid_size(total, THREADS), THREADS>>>(
      grad_out, kernel, grad_in, N, Cin, H, W, Cout, kH, kW, oH, oW, stride,
      padding);
  cuda_check(cudaGetLastError(), "conv2d_backward_input");
}

void conv2d_backward_kernel(const float *grad_out, const float *input,
                            float *grad_kernel, size_t N, size_t Cin, size_t H,
                            size_t W, size_t Cout, size_t kH, size_t kW,
                            size_t stride, size_t padding) {
  size_t oH = (H + 2 * padding - kH) / stride + 1;
  size_t oW = (W + 2 * padding - kW) / stride + 1;
  size_t total = Cout * Cin * kH * kW;
  conv2d_backward_kernel_k<<<grid_size(total, THREADS), THREADS>>>(
      grad_out, input, grad_kernel, N, Cin, H, W, Cout, kH, kW, oH, oW, stride,
      padding);
  cuda_check(cudaGetLastError(), "conv2d_backward_kernel");
}

void max_pool2d_backward(const float *grad_out, const float *input,
                         float *grad_in, size_t N, size_t C, size_t H, size_t W,
                         size_t kH, size_t kW, size_t stride, size_t padding) {
  size_t oH = (H + 2 * padding - kH) / stride + 1;
  size_t oW = (W + 2 * padding - kW) / stride + 1;
  cuda_check(cudaMemset(grad_in, 0, N * C * H * W * sizeof(float)),
             "max_pool2d_backward.memset");
  size_t total = N * C * oH * oW;
  max_pool2d_backward_k<<<grid_size(total, THREADS), THREADS>>>(
      grad_out, input, grad_in, N, C, H, W, kH, kW, oH, oW, stride, padding);
  cuda_check(cudaGetLastError(), "max_pool2d_backward");
}

void avg_pool2d_backward(const float *grad_out, float *grad_in, size_t N,
                         size_t C, size_t H, size_t W, size_t kH, size_t kW,
                         size_t stride, size_t padding) {
  size_t oH = (H + 2 * padding - kH) / stride + 1;
  size_t oW = (W + 2 * padding - kW) / stride + 1;
  cuda_check(cudaMemset(grad_in, 0, N * C * H * W * sizeof(float)),
             "avg_pool2d_backward.memset");
  size_t total = N * C * oH * oW;
  avg_pool2d_backward_k<<<grid_size(total, THREADS), THREADS>>>(
      grad_out, grad_in, N, C, H, W, kH, kW, oH, oW, stride, padding);
  cuda_check(cudaGetLastError(), "avg_pool2d_backward");
}

// bias
void bias_add_channel(const float *input, const float *bias, float *out,
                      size_t N, size_t C, size_t HW) {
  size_t total = N * C * HW;
  bias_add_channel_k<<<grid_size(total, THREADS), THREADS>>>(input, bias, out,
                                                             N, C, HW);
  cuda_check(cudaGetLastError(), "bias_add_channel");
}

void sum_to_channel(const float *in, float *out, size_t N, size_t C,
                    size_t HW) {
  sum_to_channel_k<<<grid_size(C, THREADS), THREADS>>>(in, out, N, C, HW);
  cuda_check(cudaGetLastError(), "sum_to_channel");
}

void sum_axis0(const float *in, float *out, size_t N, size_t M) {
  sum_axis0_k<<<grid_size(M, THREADS), THREADS>>>(in, out, N, M);
  cuda_check(cudaGetLastError(), "sum_axis0");
}

void gather_rows(const float *src, const size_t *indices, float *out,
                 size_t n, size_t per_sample) {
  if (n == 0 || per_sample == 0)
    return;
  size_t *idx_dev = nullptr;
  cuda_check(cudaMalloc(&idx_dev, n * sizeof(size_t)), "gather_rows alloc");
  cuda_check(cudaMemcpy(idx_dev, indices, n * sizeof(size_t),
                        cudaMemcpyHostToDevice),
             "gather_rows copy");
  size_t total = n * per_sample;
  gather_rows_k<<<grid_size(total, THREADS), THREADS>>>(src, idx_dev, out, n,
                                                        per_sample);
  cudaError_t err = cudaGetLastError();
  cudaFree(idx_dev);
  cuda_check(err, "gather_rows");
}

} // namespace cue::cuda
