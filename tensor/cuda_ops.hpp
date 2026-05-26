#pragma once

#include <cstddef>

// CUDA dispatch for the tensor module

namespace cue::cuda {

    //  raw memory 

    void* malloc_bytes(size_t bytes);
    void  free_bytes(void* ptr);
    void  memcpy_h2d(void* dst, const void* src, size_t bytes);
    void  memcpy_d2h(void* dst, const void* src, size_t bytes);
    void  memcpy_d2d(void* dst, const void* src, size_t bytes);
    void  memset_zero(void* dst, size_t bytes);
    void  device_synchronize();

    //  element-wise operations

    void add(const float* a, const float* b, float* out, size_t n);
    void sub(const float* a, const float* b, float* out, size_t n);
    void mul(const float* a, const float* b, float* out, size_t n);
    void div(const float* a, const float* b, float* out, size_t n);
    void add_scalar(const float* a, float s, float* out, size_t n);
    void mul_scalar(const float* a, float s, float* out, size_t n);

    //  activations 
//
    void relu(const float* a, float* out, size_t n);
    void relu_backward(const float* grad, const float* input, float* out, size_t n);
    void sigmoid(const float* a, float* out, size_t n);
    void tanh_act(const float* a, float* out, size_t n);
    void exp_act(const float* a, float* out, size_t n);
    void log_act(const float* a, float* out, size_t n);
    void softmax_last_dim(const float* in, float* out, size_t outer, size_t inner);

    //  linear algebra 

    void matmul(const float* a, const float* b, float* out,
                size_t m, size_t k, size_t n);
    void transpose2d(const float* a, float* out, size_t rows, size_t cols);

    //  reductions 

    float sum(const float* a, size_t n);

    //  convolution / pooling (NCHW layout)

    void conv2d_forward(const float* input, const float* kernel, float* out,
                        size_t N, size_t Cin, size_t H, size_t W,
                        size_t Cout, size_t kH, size_t kW,
                        size_t stride, size_t padding);
    void conv2d_backward_input(const float* grad_out, const float* kernel, float* grad_in,
                               size_t N, size_t Cin, size_t H, size_t W,
                               size_t Cout, size_t kH, size_t kW,
                               size_t stride, size_t padding);
    void conv2d_backward_kernel(const float* grad_out, const float* input, float* grad_kernel,
                                size_t N, size_t Cin, size_t H, size_t W,
                                size_t Cout, size_t kH, size_t kW,
                                size_t stride, size_t padding);
    void max_pool2d_forward(const float* input, float* out,
                            size_t N, size_t C, size_t H, size_t W,
                            size_t kH, size_t kW,
                            size_t stride, size_t padding);
    void max_pool2d_backward(const float* grad_out, const float* input, float* grad_in,
                             size_t N, size_t C, size_t H, size_t W,
                             size_t kH, size_t kW,
                             size_t stride, size_t padding);
    void avg_pool2d_forward(const float* input, float* out,
                            size_t N, size_t C, size_t H, size_t W,
                            size_t kH, size_t kW,
                            size_t stride, size_t padding);
    void avg_pool2d_backward(const float* grad_out, float* grad_in,
                             size_t N, size_t C, size_t H, size_t W,
                             size_t kH, size_t kW,
                             size_t stride, size_t padding);

    //  broadcasting helpers

    void bias_add_channel(const float* input, const float* bias, float* out,
                          size_t N, size_t C, size_t HW);
    void sum_to_channel(const float* in, float* out,
                        size_t N, size_t C, size_t HW);
    void sum_axis0(const float* in, float* out, size_t N, size_t M);
}
