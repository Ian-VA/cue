#include "cuda_ops.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {

constexpr int THREADS = 256;
constexpr int TILE = 16;
// upper bound on the grid for the grid-stride kernels: enough blocks to fill a
// modern GPU while keeping the launch resident so each thread amortises its
// setup over several elements on large tensors
constexpr int MAX_BLOCKS = 4096;

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
//
// Element-wise and activation kernels are bandwidth-bound, so they share two
// launch helpers below. Both use a grid-stride loop (the launch is decoupled
// from the problem size, so a capped grid stays resident and each thread
// amortises its setup over several elements) and, when every buffer is 16-byte
// aligned, a float4 path that moves 128 bits per thread per memory transaction
// instead of 32. A scalar kernel mops up the < 4 element tail.

template <class Op>
__global__ void unary_vec_k(const float4 *a, float4 *out, size_t n4, Op op) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += stride) {
    float4 v = a[i];
    out[i] = make_float4(op(v.x), op(v.y), op(v.z), op(v.w));
  }
}

template <class Op>
__global__ void unary_scalar_k(const float *a, float *out, size_t n, Op op) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride)
    out[i] = op(a[i]);
}

template <class Op>
__global__ void binary_vec_k(const float4 *a, const float4 *b, float4 *out,
                             size_t n4, Op op) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += stride) {
    float4 x = a[i], y = b[i];
    out[i] =
        make_float4(op(x.x, y.x), op(x.y, y.y), op(x.z, y.z), op(x.w, y.w));
  }
}

template <class Op>
__global__ void binary_scalar_k(const float *a, const float *b, float *out,
                                size_t n, Op op) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride)
    out[i] = op(a[i], b[i]);
}

inline bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

inline int capped_blocks(size_t units) {
  int b = grid_size(units, THREADS);
  if (b < 1)
    b = 1;
  return b < MAX_BLOCKS ? b : MAX_BLOCKS;
}

template <class Op>
inline void launch_unary(const float *a, float *out, size_t n, Op op) {
  if (n == 0)
    return;
  size_t n4 = n / 4;
  if (n4 && aligned16(a) && aligned16(out)) {
    unary_vec_k<<<capped_blocks(n4), THREADS>>>(
        reinterpret_cast<const float4 *>(a), reinterpret_cast<float4 *>(out),
        n4, op);
    size_t done = n4 * 4;
    if (done < n)
      unary_scalar_k<<<1, THREADS>>>(a + done, out + done, n - done, op);
  } else {
    unary_scalar_k<<<capped_blocks(n), THREADS>>>(a, out, n, op);
  }
}

template <class Op>
inline void launch_binary(const float *a, const float *b, float *out, size_t n,
                          Op op) {
  if (n == 0)
    return;
  size_t n4 = n / 4;
  if (n4 && aligned16(a) && aligned16(b) && aligned16(out)) {
    binary_vec_k<<<capped_blocks(n4), THREADS>>>(
        reinterpret_cast<const float4 *>(a),
        reinterpret_cast<const float4 *>(b), reinterpret_cast<float4 *>(out),
        n4, op);
    size_t done = n4 * 4;
    if (done < n)
      binary_scalar_k<<<1, THREADS>>>(a + done, b + done, out + done, n - done,
                                      op);
  } else {
    binary_scalar_k<<<capped_blocks(n), THREADS>>>(a, b, out, n, op);
  }
}

// functors fed to the launch helpers above

struct AddOp {
  __device__ float operator()(float a, float b) const { return a + b; }
};
struct SubOp {
  __device__ float operator()(float a, float b) const { return a - b; }
};
struct MulOp {
  __device__ float operator()(float a, float b) const { return a * b; }
};
struct DivOp {
  __device__ float operator()(float a, float b) const { return a / b; }
};
struct AddScalarOp {
  float s;
  __device__ float operator()(float a) const { return a + s; }
};
struct MulScalarOp {
  float s;
  __device__ float operator()(float a) const { return a * s; }
};

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

// activations (run through the launch helpers above via these functors)

struct ReluOp {
  __device__ float operator()(float a) const { return a > 0.0f ? a : 0.0f; }
};
struct ReluBackOp {
  // a = upstream gradient, b = the layer input that produced this element
  __device__ float operator()(float a, float b) const {
    return b > 0.0f ? a : 0.0f;
  }
};
struct SigmoidOp {
  __device__ float operator()(float a) const {
    return 1.0f / (1.0f + __expf(-a));
  }
};
struct TanhOp {
  __device__ float operator()(float a) const { return tanhf(a); }
};
struct ExpOp {
  __device__ float operator()(float a) const { return expf(a); }
};
struct LogOp {
  __device__ float operator()(float a) const { return logf(a); }
};

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

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, offset);
  return v;
}

// Grid-stride accumulate into a register, then a two-level block reduction:
// a warp-shuffle reduce within each warp (no shared memory, no divergence),
// one shared exchange of per-warp totals, and a final warp-shuffle reduce.
// Each block writes one partial; running the kernel again over the partials
// with a single block finishes the whole reduction on the GPU, so only one
// float is ever copied back to the host.
__global__ void reduce_sum_k(const float *in, float *out, size_t n) {
  float v = 0.0f;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride)
    v += in[i];

  __shared__ float warp_sums[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0)
    warp_sums[wid] = v;
  __syncthreads();

  int n_warps = (blockDim.x + 31) >> 5;
  if (wid == 0) {
    v = lane < n_warps ? warp_sums[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0)
      out[blockIdx.x] = v;
  }
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
  launch_binary(a, b, out, n, AddOp{});
  cuda_check(cudaGetLastError(), "add");
}

void sub(const float *a, const float *b, float *out, size_t n) {
  launch_binary(a, b, out, n, SubOp{});
  cuda_check(cudaGetLastError(), "sub");
}

void mul(const float *a, const float *b, float *out, size_t n) {
  launch_binary(a, b, out, n, MulOp{});
  cuda_check(cudaGetLastError(), "mul");
}

void div(const float *a, const float *b, float *out, size_t n) {
  launch_binary(a, b, out, n, DivOp{});
  cuda_check(cudaGetLastError(), "div");
}

void add_scalar(const float *a, float s, float *out, size_t n) {
  launch_unary(a, out, n, AddScalarOp{s});
  cuda_check(cudaGetLastError(), "add_scalar");
}

void mul_scalar(const float *a, float s, float *out, size_t n) {
  launch_unary(a, out, n, MulScalarOp{s});
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
  launch_unary(a, out, n, ReluOp{});
  cuda_check(cudaGetLastError(), "relu");
}

void relu_backward(const float *grad, const float *input, float *out,
                   size_t n) {
  launch_binary(grad, input, out, n, ReluBackOp{});
  cuda_check(cudaGetLastError(), "relu_backward");
}

void sigmoid(const float *a, float *out, size_t n) {
  launch_unary(a, out, n, SigmoidOp{});
  cuda_check(cudaGetLastError(), "sigmoid");
}

void tanh_act(const float *a, float *out, size_t n) {
  launch_unary(a, out, n, TanhOp{});
  cuda_check(cudaGetLastError(), "tanh");
}

void exp_act(const float *a, float *out, size_t n) {
  launch_unary(a, out, n, ExpOp{});
  cuda_check(cudaGetLastError(), "exp");
}

void log_act(const float *a, float *out, size_t n) {
  launch_unary(a, out, n, LogOp{});
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
  int blocks = capped_blocks(n);

  // one scratch buffer holds the per-block partials in [0, blocks) and the
  // final scalar at [blocks]; the second pass reduces the partials in place on
  // the GPU so exactly one float crosses the bus
  float *scratch = nullptr;
  cuda_check(cudaMalloc(&scratch, ((size_t)blocks + 1) * sizeof(float)),
             "sum.alloc");

  reduce_sum_k<<<blocks, THREADS>>>(a, scratch, n);
  cuda_check(cudaGetLastError(), "sum.pass1");
  reduce_sum_k<<<1, THREADS>>>(scratch, scratch + blocks, (size_t)blocks);
  cuda_check(cudaGetLastError(), "sum.pass2");

  float result = 0.0f;
  cuda_check(cudaMemcpy(&result, scratch + blocks, sizeof(float),
                        cudaMemcpyDeviceToHost),
             "sum.d2h");
  cudaFree(scratch);
  return result;
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
