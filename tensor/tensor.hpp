#pragma once

#include <vector>
#include <cstdint>
#include <array>
#include <memory>
#include <initializer_list>
#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <random>
#include <type_traits>
#include <limits>
#include <string>

#include "cuda_ops.hpp"

using Index = size_t;

template<typename Val>
using Scalar = Val;

template<typename Val, size_t N>
using Vec = std::array<Val, N>;

enum class Device : uint8_t {
    CPU,
    CUDA
};

template<typename Val>
class TensorStorage {
    public:

        TensorStorage(Index size, Device device) : _size(size), _device(device) {
            if (size == 0) {
                _data = nullptr;
                return;
            }
            if (device == Device::CPU) {
                _data = new Val[size]();
            } else {
                if constexpr (std::is_same_v<Val, float>) {
                    _data = static_cast<Val*>(cue::cuda::malloc_bytes(size * sizeof(Val)));
                    cue::cuda::memset_zero(_data, size * sizeof(Val));
                } else {
                    throw std::runtime_error("CUDA tensors only support float");
                }
            }
        }

        ~TensorStorage() {
            if (_data == nullptr) return;
            if (_device == Device::CPU) {
                delete[] _data;
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::free_bytes(_data);
            }
        }

        TensorStorage(const TensorStorage&)            = delete;
        TensorStorage& operator=(const TensorStorage&) = delete;

        Val*       data()         noexcept { return _data; }
        const Val* data()   const noexcept { return _data; }
        Index      size()   const noexcept { return _size; }
        Device     device() const noexcept { return _device; }

    private:
        Val*   _data {nullptr};
        Index  _size;
        Device _device;
};


template<typename Val>
class Tensor {
    public:

        Tensor() = default;

        explicit Tensor(const std::vector<Index> &shape, Device device = Device::CPU)
                : _shape(shape), _device(device) {
            init();
        }

        explicit Tensor(std::initializer_list<Val> row) : _device(Device::CPU) {
            _shape = {row.size()};
            init();
            std::copy(row.begin(), row.end(), _storage->data());
        }

        explicit Tensor(std::initializer_list<std::initializer_list<Val>> rows) : _device(Device::CPU) {
            if (rows.size() == 0) {
                throw std::invalid_argument("Rows in 2D Tensor() declaration cannot be empty");
            }

            Index cols = rows.begin()->size();

            for (const auto& row : rows) {
                if (row.size() != cols) {
                    throw std::invalid_argument("Rows must have the same length in 2D Tensor() declaration");
                }
            }

            _shape = {rows.size(), cols};
            init();
            Val* it = _storage->data();

            for (const auto& row : rows) {
                for (const auto& item : row) {
                    *it++ = item;
                }
            }
        }

        //  factories 

        static Tensor<Val> zeros(const std::vector<Index>& shape, Device device = Device::CPU) {
            return Tensor<Val>(shape, device);
        }

        static Tensor<Val> from_values(std::initializer_list<Val> row) {
            return Tensor<Val>(row);
        }

        static Tensor<Val> from_values(std::initializer_list<std::initializer_list<Val>> rows) {
            return Tensor<Val>(rows);
        }

        static Tensor<Val> ones(const std::vector<Index>& shape, Device device = Device::CPU) {
            return full(shape, Val(1), device);
        }

        static Tensor<Val> full(const std::vector<Index>& shape, Val value, Device device = Device::CPU) {
            Tensor<Val> cpu(shape, Device::CPU);
            std::fill_n(cpu._storage->data(), cpu.size(), value);
            return device == Device::CPU ? cpu : cpu.to_cuda();
        }

        static Tensor<Val> randn(const std::vector<Index>& shape,
                                 Val mean = Val(0), Val stddev = Val(1),
                                 Device device = Device::CPU) {
            Tensor<Val> cpu(shape, Device::CPU);
            static thread_local std::mt19937 gen{std::random_device{}()};
            std::normal_distribution<double> dist(static_cast<double>(mean),
                                                  static_cast<double>(stddev));
            for (Index i = 0; i < cpu.size(); ++i) {
                cpu._storage->data()[i] = static_cast<Val>(dist(gen));
            }
            return device == Device::CPU ? cpu : cpu.to_cuda();
        }

        //  introspection 

        const std::vector<Index>& shape()   const noexcept { return _shape; }
        const std::vector<Index>& stride()  const noexcept { return _stride; }
        Index                     rank()    const noexcept { return _shape.size(); }
        Index                     size()    const noexcept { return _storage ? _storage->size() : 0; }
        Device                    device()  const noexcept { return _device; }
        bool                      on_cuda() const noexcept { return _device == Device::CUDA; }

        Val*       data()       noexcept { return _storage->data(); }
        const Val* data() const noexcept { return _storage->data(); }

        //  device transfer 

        Tensor<Val> to(Device device) const {
            if (device == _device) return *this;
            return device == Device::CUDA ? to_cuda() : to_cpu();
        }

        Tensor<Val> to_cuda() const {
            if (_device == Device::CUDA) return *this;
            if constexpr (!std::is_same_v<Val, float>) {
                throw std::runtime_error("CUDA tensors only support float");
            } else {
                Tensor<Val> out(_shape, Device::CUDA);
                if (size() > 0) {
                    cue::cuda::memcpy_h2d(out._storage->data(), _storage->data(),
                                          size() * sizeof(Val));
                }
                return out;
            }
        }

        Tensor<Val> to_cpu() const {
            if (_device == Device::CPU) return *this;
            if constexpr (!std::is_same_v<Val, float>) {
                throw std::runtime_error("CUDA tensors only support float");
            } else {
                Tensor<Val> out(_shape, Device::CPU);
                if (size() > 0) {
                    cue::cuda::memcpy_d2h(out._storage->data(), _storage->data(),
                                          size() * sizeof(Val));
                }
                return out;
            }
        }

        Tensor<Val> clone() const {
            Tensor<Val> out(_shape, _device);
            if (size() == 0) return out;
            if (_device == Device::CPU) {
                std::copy_n(_storage->data(), size(), out._storage->data());
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::memcpy_d2d(out._storage->data(), _storage->data(),
                                      size() * sizeof(Val));
            }
            return out;
        }

        //  shape manipulation 

        void reshape(const std::vector<Index>& shape) {
            Index product = 1;
            for (Index dim : shape) {
                product *= dim;
            }

            if (product != size()) {
                throw std::invalid_argument("Cannot reshape tensor without altering data");
            }

            _shape = shape;
            compute_assign_stride();
        }

        Tensor<Val> view(const std::vector<Index>& shape) const {
            Index product = 1;
            for (Index dim : shape) {
                product *= dim;
            }
            if (product != size()) {
                throw std::invalid_argument("View shape does not match tensor size");
            }

            Tensor<Val> out = *this;
            out._shape = shape;
            out.compute_assign_stride();
            return out;
        }

        Tensor<Val> flatten(Index start_dim = 1) const {
            if (start_dim >= rank()) return *this;
            std::vector<Index> new_shape(_shape.begin(), _shape.begin() + start_dim);
            Index tail = 1;
            for (Index d = start_dim; d < rank(); ++d) {
                tail *= _shape[d];
            }
            new_shape.push_back(tail);
            return view(new_shape);
        }

        Tensor<Val> transpose(Index dim_a, Index dim_b) const {
            if (dim_a >= rank() || dim_b >= rank()) {
                throw std::out_of_range("Transpose dim out of range");
            }
            if (dim_a == dim_b) return clone();

            if (rank() == 2 && on_cuda()) {
                if constexpr (std::is_same_v<Val, float>) {
                    Tensor<Val> out({_shape[1], _shape[0]}, Device::CUDA);
                    cue::cuda::transpose2d(_storage->data(), out._storage->data(),
                                           _shape[0], _shape[1]);
                    return out;
                }
            }

            Tensor<Val> src = on_cuda() ? to_cpu() : *this;
            std::vector<Index> new_shape = _shape;
            std::swap(new_shape[dim_a], new_shape[dim_b]);
            Tensor<Val> out(new_shape, Device::CPU);

            std::vector<Index> coord(rank(), 0);
            for (Index i = 0; i < src.size(); ++i) {
                Index tmp = i;
                for (Index d = 0; d < rank(); ++d) {
                    coord[d] = tmp / src._stride[d];
                    tmp      = tmp % src._stride[d];
                }
                std::swap(coord[dim_a], coord[dim_b]);
                Index dst = 0;
                for (Index d = 0; d < rank(); ++d) {
                    dst += coord[d] * out._stride[d];
                }
                out._storage->data()[dst] = src._storage->data()[i];
            }

            return on_cuda() ? out.to_cuda() : out;
        }

        //  element access 

        Val operator[](std::initializer_list<Index> coord) const {
            require_cpu("operator[]");
            return _storage->data()[index(coord)];
        }

        Val& at(std::initializer_list<Index> coord) {
            require_cpu("at()");
            return _storage->data()[index(coord)];
        }

        //  element-wise arithmetic 

        Tensor<Val> add(const Tensor<Val>& other) const { return elementwise(other, Op::Add); }
        Tensor<Val> sub(const Tensor<Val>& other) const { return elementwise(other, Op::Sub); }
        Tensor<Val> mul(const Tensor<Val>& other) const { return elementwise(other, Op::Mul); }
        Tensor<Val> div(const Tensor<Val>& other) const { return elementwise(other, Op::Div); }

        Tensor<Val> add(Val s) const { return scalar_op(s, Op::Add); }
        Tensor<Val> sub(Val s) const { return scalar_op(Val(0) - s, Op::Add); }
        Tensor<Val> mul(Val s) const { return scalar_op(s, Op::Mul); }

        Tensor<Val> neg() const { return mul(Val(0) - Val(1)); }

        Tensor<Val> operator+(const Tensor<Val>& other) const { return add(other); }
        Tensor<Val> operator-(const Tensor<Val>& other) const { return sub(other); }
        Tensor<Val> operator*(const Tensor<Val>& other) const { return mul(other); }
        Tensor<Val> operator/(const Tensor<Val>& other) const { return div(other); }
        Tensor<Val> operator-() const { return neg(); }

        Tensor<Val> operator+(Val s) const { return add(s); }
        Tensor<Val> operator-(Val s) const { return sub(s); }
        Tensor<Val> operator*(Val s) const { return mul(s); }

        Tensor<Val>& operator+=(const Tensor<Val>& other) { *this = add(other); return *this; }
        Tensor<Val>& operator-=(const Tensor<Val>& other) { *this = sub(other); return *this; }
        Tensor<Val>& operator*=(const Tensor<Val>& other) { *this = mul(other); return *this; }
        Tensor<Val>& operator/=(const Tensor<Val>& other) { *this = div(other); return *this; }

        //  linear algebra 

        Tensor<Val> matmul(const Tensor<Val>& other) const {
            if (rank() != 2 || other.rank() != 2) {
                throw std::invalid_argument("matmul requires 2D operands");
            }
            if (_shape[1] != other._shape[0]) {
                throw std::invalid_argument("matmul inner-dim mismatch");
            }
            check_same_device(other);

            Index M = _shape[0];
            Index K = _shape[1];
            Index N = other._shape[1];
            Tensor<Val> out({M, N}, _device);

            if (_device == Device::CPU) {
                const Val* A = _storage->data();
                const Val* B = other._storage->data();
                Val*       O = out._storage->data();
                for (Index i = 0; i < M; ++i) {
                    for (Index k = 0; k < K; ++k) {
                        Val a = A[i*K + k];
                        for (Index j = 0; j < N; ++j) {
                            O[i*N + j] += a * B[k*N + j];
                        }
                    }
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::matmul(_storage->data(), other._storage->data(),
                                  out._storage->data(), M, K, N);
            }
            return out;
        }

        //  reductions 

        Val sum() const {
            if (size() == 0) return Val(0);
            if (_device == Device::CPU) {
                const Val* d = _storage->data();
                Val s = Val(0);
                for (Index i = 0; i < size(); ++i) s += d[i];
                return s;
            }
            if constexpr (std::is_same_v<Val, float>) {
                return cue::cuda::sum(_storage->data(), size());
            }
            return Val(0);
        }

        Val mean() const {
            return size() == 0 ? Val(0) : sum() / Val(size());
        }

        //  activations 

        Tensor<Val> relu() const {
            Tensor<Val> out(_shape, _device);
            if (_device == Device::CPU) {
                for (Index i = 0; i < size(); ++i) {
                    Val v = _storage->data()[i];
                    out._storage->data()[i] = v > Val(0) ? v : Val(0);
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::relu(_storage->data(), out._storage->data(), size());
            }
            return out;
        }

        Tensor<Val> relu_backward(const Tensor<Val>& input) const {
            check_same_shape(input);
            check_same_device(input);
            Tensor<Val> out(_shape, _device);
            if (_device == Device::CPU) {
                for (Index i = 0; i < size(); ++i) {
                    out._storage->data()[i] = input._storage->data()[i] > Val(0)
                                                ? _storage->data()[i]
                                                : Val(0);
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::relu_backward(_storage->data(), input._storage->data(),
                                         out._storage->data(), size());
            }
            return out;
        }

        Tensor<Val> sigmoid() const {
            Tensor<Val> out(_shape, _device);
            if (_device == Device::CPU) {
                for (Index i = 0; i < size(); ++i) {
                    Val v = _storage->data()[i];
                    out._storage->data()[i] = Val(1) / (Val(1) + std::exp(-v));
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::sigmoid(_storage->data(), out._storage->data(), size());
            }
            return out;
        }

        Tensor<Val> tanh() const {
            Tensor<Val> out(_shape, _device);
            if (_device == Device::CPU) {
                for (Index i = 0; i < size(); ++i) {
                    out._storage->data()[i] = std::tanh(_storage->data()[i]);
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::tanh_act(_storage->data(), out._storage->data(), size());
            }
            return out;
        }

        Tensor<Val> exp() const {
            Tensor<Val> out(_shape, _device);
            if (_device == Device::CPU) {
                for (Index i = 0; i < size(); ++i) {
                    out._storage->data()[i] = std::exp(_storage->data()[i]);
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::exp_act(_storage->data(), out._storage->data(), size());
            }
            return out;
        }

        Tensor<Val> log() const {
            Tensor<Val> out(_shape, _device);
            if (_device == Device::CPU) {
                for (Index i = 0; i < size(); ++i) {
                    out._storage->data()[i] = std::log(_storage->data()[i]);
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::log_act(_storage->data(), out._storage->data(), size());
            }
            return out;
        }

        Tensor<Val> softmax() const {
            if (rank() == 0) {
                throw std::invalid_argument("Cannot softmax 0-d tensor");
            }
            Index inner = _shape.back();
            Index outer = size() / inner;

            if (on_cuda()) {
                if constexpr (std::is_same_v<Val, float>) {
                    Tensor<Val> out(_shape, Device::CUDA);
                    cue::cuda::softmax_last_dim(_storage->data(), out._storage->data(),
                                                outer, inner);
                    return out;
                }
            }

            Tensor<Val> out(_shape, Device::CPU);
            const Val* in_ = _storage->data();
            Val*       o_  = out._storage->data();
            for (Index r = 0; r < outer; ++r) {
                Val m = in_[r*inner];
                for (Index i = 1; i < inner; ++i) m = std::max(m, in_[r*inner + i]);
                Val s = Val(0);
                for (Index i = 0; i < inner; ++i) {
                    Val e = std::exp(in_[r*inner + i] - m);
                    o_[r*inner + i] = e;
                    s += e;
                }
                for (Index i = 0; i < inner; ++i) o_[r*inner + i] /= s;
            }
            return out;
        }

        //  convolution / pooling 

        Tensor<Val> conv2d(const Tensor<Val>& kernel,
                           Index stride = 1, Index padding = 0) const {
            if (rank() != 4 || kernel.rank() != 4) {
                throw std::invalid_argument("conv2d expects 4D operands");
            }
            if (_shape[1] != kernel._shape[1]) {
                throw std::invalid_argument("conv2d input/kernel channel mismatch");
            }
            if (stride == 0) {
                throw std::invalid_argument("conv2d stride must be > 0");
            }
            check_same_device(kernel);

            Index N    = _shape[0];
            Index Cin  = _shape[1];
            Index H    = _shape[2];
            Index W    = _shape[3];
            Index Cout = kernel._shape[0];
            Index kH   = kernel._shape[2];
            Index kW   = kernel._shape[3];

            if (H + 2*padding < kH || W + 2*padding < kW) {
                throw std::invalid_argument("conv2d kernel does not fit in padded input");
            }
            Index oH = (H + 2*padding - kH) / stride + 1;
            Index oW = (W + 2*padding - kW) / stride + 1;

            Tensor<Val> out({N, Cout, oH, oW}, _device);

            if (_device == Device::CPU) {
                conv2d_cpu(out, kernel, N, Cin, H, W, Cout, kH, kW, oH, oW, stride, padding);
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::conv2d_forward(
                    _storage->data(), kernel._storage->data(), out._storage->data(),
                    N, Cin, H, W, Cout, kH, kW, stride, padding);
            }
            return out;
        }

        Tensor<Val> max_pool2d(Index kH, Index kW,
                               Index stride = 0, Index padding = 0) const {
            return pool2d(kH, kW, stride, padding, /*max_mode=*/true);
        }

        Tensor<Val> avg_pool2d(Index kH, Index kW,
                               Index stride = 0, Index padding = 0) const {
            return pool2d(kH, kW, stride, padding, /*max_mode=*/false);
        }

        Tensor<Val> conv2d_backward_input(const Tensor<Val>& kernel,
                                          const std::vector<Index>& input_shape,
                                          Index stride = 1, Index padding = 0) const {
            if (rank() != 4 || kernel.rank() != 4) {
                throw std::invalid_argument("conv2d_backward_input expects 4D operands");
            }
            if (input_shape.size() != 4) {
                throw std::invalid_argument("input_shape must be 4D");
            }
            check_same_device(kernel);

            Index N    = input_shape[0];
            Index Cin  = input_shape[1];
            Index H    = input_shape[2];
            Index W    = input_shape[3];
            Index Cout = kernel._shape[0];
            Index kH   = kernel._shape[2];
            Index kW   = kernel._shape[3];
            Index oH   = _shape[2];
            Index oW   = _shape[3];

            Tensor<Val> grad_in(input_shape, _device);

            if (_device == Device::CPU) {
                const Val* g  = _storage->data();
                const Val* ke = kernel._storage->data();
                Val*       gi = grad_in._storage->data();
                std::fill_n(gi, grad_in.size(), Val(0));
                for (Index n = 0; n < N; ++n) {
                    for (Index co = 0; co < Cout; ++co) {
                        for (Index oh = 0; oh < oH; ++oh) {
                            for (Index ow = 0; ow < oW; ++ow) {
                                Val g_val = g[((n*Cout + co)*oH + oh)*oW + ow];
                                for (Index ci = 0; ci < Cin; ++ci) {
                                    for (Index ki = 0; ki < kH; ++ki) {
                                        for (Index kj = 0; kj < kW; ++kj) {
                                            long long ih = (long long)oh*stride + (long long)ki - (long long)padding;
                                            long long iw = (long long)ow*stride + (long long)kj - (long long)padding;
                                            if (ih < 0 || ih >= (long long)H) continue;
                                            if (iw < 0 || iw >= (long long)W) continue;
                                            Index in_idx = ((n*Cin + ci)*H + (Index)ih)*W + (Index)iw;
                                            Index k_idx  = ((co*Cin + ci)*kH + ki)*kW + kj;
                                            gi[in_idx] += g_val * ke[k_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::conv2d_backward_input(
                    _storage->data(), kernel._storage->data(),
                    grad_in._storage->data(),
                    N, Cin, H, W, Cout, kH, kW, stride, padding);
            }
            return grad_in;
        }

        Tensor<Val> conv2d_backward_kernel(const Tensor<Val>& input,
                                           const std::vector<Index>& kernel_shape,
                                           Index stride = 1, Index padding = 0) const {
            if (rank() != 4 || input.rank() != 4) {
                throw std::invalid_argument("conv2d_backward_kernel expects 4D operands");
            }
            if (kernel_shape.size() != 4) {
                throw std::invalid_argument("kernel_shape must be 4D");
            }
            check_same_device(input);

            Index N    = input._shape[0];
            Index Cin  = input._shape[1];
            Index H    = input._shape[2];
            Index W    = input._shape[3];
            Index Cout = kernel_shape[0];
            Index kH   = kernel_shape[2];
            Index kW   = kernel_shape[3];
            Index oH   = _shape[2];
            Index oW   = _shape[3];

            Tensor<Val> grad_k(kernel_shape, _device);

            if (_device == Device::CPU) {
                const Val* g  = _storage->data();
                const Val* in_ = input._storage->data();
                Val*       gk = grad_k._storage->data();
                std::fill_n(gk, grad_k.size(), Val(0));
                for (Index co = 0; co < Cout; ++co) {
                    for (Index ci = 0; ci < Cin; ++ci) {
                        for (Index ki = 0; ki < kH; ++ki) {
                            for (Index kj = 0; kj < kW; ++kj) {
                                Val acc = Val(0);
                                for (Index n = 0; n < N; ++n) {
                                    for (Index oh = 0; oh < oH; ++oh) {
                                        for (Index ow = 0; ow < oW; ++ow) {
                                            long long ih = (long long)oh*stride + (long long)ki - (long long)padding;
                                            long long iw = (long long)ow*stride + (long long)kj - (long long)padding;
                                            if (ih < 0 || ih >= (long long)H) continue;
                                            if (iw < 0 || iw >= (long long)W) continue;
                                            Index in_idx = ((n*Cin + ci)*H + (Index)ih)*W + (Index)iw;
                                            Index g_idx  = ((n*Cout + co)*oH + oh)*oW + ow;
                                            acc += in_[in_idx] * g[g_idx];
                                        }
                                    }
                                }
                                gk[((co*Cin + ci)*kH + ki)*kW + kj] = acc;
                            }
                        }
                    }
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::conv2d_backward_kernel(
                    _storage->data(), input._storage->data(),
                    grad_k._storage->data(),
                    N, Cin, H, W, Cout, kH, kW, stride, padding);
            }
            return grad_k;
        }

        Tensor<Val> max_pool2d_backward(const Tensor<Val>& input,
                                        Index kH, Index kW,
                                        Index stride = 0, Index padding = 0) const {
            return pool2d_backward(input, kH, kW, stride, padding, /*max_mode=*/true);
        }

        Tensor<Val> avg_pool2d_backward(const std::vector<Index>& input_shape,
                                        Index kH, Index kW,
                                        Index stride = 0, Index padding = 0) const {
            if (stride == 0) stride = kH;

            Index N = input_shape[0];
            Index C = input_shape[1];
            Index H = input_shape[2];
            Index W = input_shape[3];
            Index oH = _shape[2];
            Index oW = _shape[3];

            Tensor<Val> grad_in(input_shape, _device);

            if (_device == Device::CPU) {
                const Val* g  = _storage->data();
                Val*       gi = grad_in._storage->data();
                std::fill_n(gi, grad_in.size(), Val(0));
                for (Index n = 0; n < N; ++n) {
                    for (Index c = 0; c < C; ++c) {
                        for (Index oh = 0; oh < oH; ++oh) {
                            for (Index ow = 0; ow < oW; ++ow) {
                                Index count = 0;
                                for (Index ki = 0; ki < kH; ++ki) {
                                    for (Index kj = 0; kj < kW; ++kj) {
                                        long long ih = (long long)oh*stride + (long long)ki - (long long)padding;
                                        long long iw = (long long)ow*stride + (long long)kj - (long long)padding;
                                        if (ih < 0 || ih >= (long long)H) continue;
                                        if (iw < 0 || iw >= (long long)W) continue;
                                        ++count;
                                    }
                                }
                                if (count == 0) continue;
                                Val share = g[((n*C + c)*oH + oh)*oW + ow] / Val(count);
                                for (Index ki = 0; ki < kH; ++ki) {
                                    for (Index kj = 0; kj < kW; ++kj) {
                                        long long ih = (long long)oh*stride + (long long)ki - (long long)padding;
                                        long long iw = (long long)ow*stride + (long long)kj - (long long)padding;
                                        if (ih < 0 || ih >= (long long)H) continue;
                                        if (iw < 0 || iw >= (long long)W) continue;
                                        gi[((n*C + c)*H + (Index)ih)*W + (Index)iw] += share;
                                    }
                                }
                            }
                        }
                    }
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::avg_pool2d_backward(_storage->data(),
                                               grad_in._storage->data(),
                                               N, C, H, W, kH, kW, stride, padding);
            }
            return grad_in;
        }

        Tensor<Val> sum_to_channel() const {
            if (rank() < 2) {
                throw std::invalid_argument("sum_to_channel requires rank >= 2");
            }
            Index N = _shape[0];
            Index C = _shape[1];
            Index HW = 1;
            for (Index d = 2; d < rank(); ++d) HW *= _shape[d];

            Tensor<Val> out({C}, _device);
            if (_device == Device::CPU) {
                const Val* in_  = _storage->data();
                Val*       out_ = out._storage->data();
                std::fill_n(out_, C, Val(0));
                for (Index n = 0; n < N; ++n) {
                    for (Index c = 0; c < C; ++c) {
                        const Val* row = in_ + (n*C + c) * HW;
                        Val acc = Val(0);
                        for (Index i = 0; i < HW; ++i) acc += row[i];
                        out_[c] += acc;
                    }
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::sum_to_channel(_storage->data(), out._storage->data(),
                                          N, C, HW);
            }
            return out;
        }

        Tensor<Val> sum_axis0() const {
            if (rank() == 0 || size() == 0) {
                throw std::invalid_argument("sum_axis0 requires a non-empty tensor");
            }
            Index N = _shape[0];
            Index M = size() / N;

            std::vector<Index> out_shape;
            if (rank() == 1) {
                out_shape.push_back(1);
            } else {
                out_shape.reserve(rank() - 1);
                for (Index d = 1; d < rank(); ++d) out_shape.push_back(_shape[d]);
            }
            Tensor<Val> out(out_shape, _device);

            if (_device == Device::CPU) {
                const Val* in_  = _storage->data();
                Val*       out_ = out._storage->data();
                std::fill_n(out_, M, Val(0));
                for (Index i = 0; i < N; ++i) {
                    for (Index j = 0; j < M; ++j) out_[j] += in_[i*M + j];
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::sum_axis0(_storage->data(), out._storage->data(), N, M);
            }
            return out;
        }

        Tensor<Val> add_bias(const Tensor<Val>& bias) const {
            if (bias.rank() != 1) {
                throw std::invalid_argument("bias must be 1D");
            }
            if (rank() < 2) {
                throw std::invalid_argument("add_bias needs an input of rank >= 2");
            }
            if (_shape[1] != bias._shape[0]) {
                throw std::invalid_argument("bias length must match channel dim");
            }
            check_same_device(bias);

            Index N = _shape[0];
            Index C = _shape[1];
            Index HW = 1;
            for (Index d = 2; d < rank(); ++d) HW *= _shape[d];

            Tensor<Val> out(_shape, _device);

            if (_device == Device::CPU) {
                const Val* in_   = _storage->data();
                const Val* bias_ = bias._storage->data();
                Val*       out_  = out._storage->data();
                for (Index n = 0; n < N; ++n) {
                    for (Index c = 0; c < C; ++c) {
                        Val        b      = bias_[c];
                        Index      base   = (n*C + c) * HW;
                        const Val* in_row = in_  + base;
                        Val*       o_row  = out_ + base;
                        for (Index i = 0; i < HW; ++i) o_row[i] = in_row[i] + b;
                    }
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::bias_add_channel(_storage->data(), bias._storage->data(),
                                            out._storage->data(), N, C, HW);
            }
            return out;
        }

    private:

        enum class Op { Add, Sub, Mul, Div };

        Tensor<Val> elementwise(const Tensor<Val>& other, Op op) const {
            check_same_device(other);

            // shapes differ but are broadcast compatible take the general path
            if (_shape != other._shape) {
                return broadcast_elementwise(other, op);
            }

            Tensor<Val> out(_shape, _device);
            Index n = size();

            if (_device == Device::CPU) {
                const Val* a = _storage->data();
                const Val* b = other._storage->data();
                Val*       o = out._storage->data();
                switch (op) {
                    case Op::Add: for (Index i = 0; i < n; ++i) o[i] = a[i] + b[i]; break;
                    case Op::Sub: for (Index i = 0; i < n; ++i) o[i] = a[i] - b[i]; break;
                    case Op::Mul: for (Index i = 0; i < n; ++i) o[i] = a[i] * b[i]; break;
                    case Op::Div: for (Index i = 0; i < n; ++i) o[i] = a[i] / b[i]; break;
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                const float* a = _storage->data();
                const float* b = other._storage->data();
                float*       o = out._storage->data();
                switch (op) {
                    case Op::Add: cue::cuda::add(a, b, o, n); break;
                    case Op::Sub: cue::cuda::sub(a, b, o, n); break;
                    case Op::Mul: cue::cuda::mul(a, b, o, n); break;
                    case Op::Div: cue::cuda::div(a, b, o, n); break;
                }
            }
            return out;
        }

        // numpy style broadcasting align shapes from the right a dim is
        // compatible if equal or one of them is 1 the size 1 dim is stretched
        static std::vector<Index> broadcast_shape(const std::vector<Index>& a,
                                                  const std::vector<Index>& b) {
            Index r  = std::max(a.size(), b.size());
            Index pa = r - a.size();
            Index pb = r - b.size();
            std::vector<Index> out(r);
            for (Index i = 0; i < r; ++i) {
                Index ad = i < pa ? 1 : a[i - pa];
                Index bd = i < pb ? 1 : b[i - pb];
                if (ad != bd && ad != 1 && bd != 1) {
                    throw std::invalid_argument("Tensor shapes are not broadcastable");
                }
                out[i] = std::max(ad, bd);
            }
            return out;
        }

        // row major strides for shape left padded to out_rank with 0 on any axis
        // that is broadcast (size 1 or missing) so it reads one element repeatedly
        static std::vector<Index> broadcast_strides(const std::vector<Index>& shape,
                                                    Index out_rank) {
            Index r = shape.size();
            std::vector<Index> stride(r, 1);
            Index p = 1;
            for (int i = (int)r - 1; i >= 0; --i) { stride[i] = p; p *= shape[i]; }
            std::vector<Index> out(out_rank, 0);
            for (Index i = 0; i < r; ++i) {
                out[out_rank - r + i] = shape[i] == 1 ? 0 : stride[i];
            }
            return out;
        }

        Tensor<Val> broadcast_elementwise(const Tensor<Val>& other, Op op) const {
            std::vector<Index> out_shape = broadcast_shape(_shape, other._shape);
            Index out_rank = out_shape.size();
            std::vector<Index> a_stride = broadcast_strides(_shape, out_rank);
            std::vector<Index> b_stride = broadcast_strides(other._shape, out_rank);

            Tensor<Val> out(out_shape, _device);
            Index n = out.size();

            if (_device == Device::CPU) {
                const Val* a = _storage->data();
                const Val* b = other._storage->data();
                Val*       o = out._storage->data();
                for (Index idx = 0; idx < n; ++idx) {
                    Index rem = idx, a_off = 0, b_off = 0;
                    for (int i = (int)out_rank - 1; i >= 0; --i) {
                        Index coord = rem % out_shape[i];
                        rem        /= out_shape[i];
                        a_off      += coord * a_stride[i];
                        b_off      += coord * b_stride[i];
                    }
                    switch (op) {
                        case Op::Add: o[idx] = a[a_off] + b[b_off]; break;
                        case Op::Sub: o[idx] = a[a_off] - b[b_off]; break;
                        case Op::Mul: o[idx] = a[a_off] * b[b_off]; break;
                        case Op::Div: o[idx] = a[a_off] / b[b_off]; break;
                    }
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                if (out_rank > cue::cuda::MAX_DIMS) {
                    throw std::invalid_argument("broadcast rank exceeds CUDA MAX_DIMS");
                }
                cue::cuda::BroadcastDims d{};
                d.rank = (int)out_rank;
                for (Index i = 0; i < out_rank; ++i) {
                    d.out_shape[i] = out_shape[i];
                    d.a_stride[i]  = a_stride[i];
                    d.b_stride[i]  = b_stride[i];
                }
                cue::cuda::BinOp cop = cue::cuda::BinOp::Add;
                switch (op) {
                    case Op::Add: cop = cue::cuda::BinOp::Add; break;
                    case Op::Sub: cop = cue::cuda::BinOp::Sub; break;
                    case Op::Mul: cop = cue::cuda::BinOp::Mul; break;
                    case Op::Div: cop = cue::cuda::BinOp::Div; break;
                }
                cue::cuda::broadcast_binop(_storage->data(), other._storage->data(),
                                           out._storage->data(), n, d, cop);
            }
            return out;
        }

        Tensor<Val> scalar_op(Val s, Op op) const {
            Tensor<Val> out(_shape, _device);
            Index n = size();

            if (_device == Device::CPU) {
                const Val* a = _storage->data();
                Val*       o = out._storage->data();
                switch (op) {
                    case Op::Add: for (Index i = 0; i < n; ++i) o[i] = a[i] + s; break;
                    case Op::Mul: for (Index i = 0; i < n; ++i) o[i] = a[i] * s; break;
                    default:      throw std::logic_error("scalar_op only supports Add and Mul");
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                const float* a = _storage->data();
                float*       o = out._storage->data();
                switch (op) {
                    case Op::Add: cue::cuda::add_scalar(a, s, o, n); break;
                    case Op::Mul: cue::cuda::mul_scalar(a, s, o, n); break;
                    default:      throw std::logic_error("scalar_op only supports Add and Mul");
                }
            }
            return out;
        }

        void conv2d_cpu(Tensor<Val>& out, const Tensor<Val>& kernel,
                        Index N, Index Cin, Index H, Index W,
                        Index Cout, Index kH, Index kW,
                        Index oH, Index oW,
                        Index stride, Index padding) const {
            const Val* in_  = _storage->data();
            const Val* ker_ = kernel._storage->data();
            Val*       out_ = out._storage->data();

            for (Index n = 0; n < N; ++n) {
                for (Index co = 0; co < Cout; ++co) {
                    for (Index oh = 0; oh < oH; ++oh) {
                        for (Index ow = 0; ow < oW; ++ow) {
                            Val acc = Val(0);
                            for (Index ci = 0; ci < Cin; ++ci) {
                                for (Index ki = 0; ki < kH; ++ki) {
                                    for (Index kj = 0; kj < kW; ++kj) {
                                        long long ih = (long long)oh*stride + (long long)ki - (long long)padding;
                                        long long iw = (long long)ow*stride + (long long)kj - (long long)padding;
                                        if (ih < 0 || ih >= (long long)H) continue;
                                        if (iw < 0 || iw >= (long long)W) continue;
                                        Index in_idx = ((n*Cin + ci)*H + (Index)ih)*W + (Index)iw;
                                        Index k_idx  = ((co*Cin + ci)*kH + ki)*kW + kj;
                                        acc += in_[in_idx] * ker_[k_idx];
                                    }
                                }
                            }
                            out_[((n*Cout + co)*oH + oh)*oW + ow] = acc;
                        }
                    }
                }
            }
        }

        Tensor<Val> pool2d(Index kH, Index kW, Index stride, Index padding,
                           bool max_mode) const {
            if (rank() != 4) {
                throw std::invalid_argument("pool2d expects a 4D input");
            }
            if (stride == 0) stride = kH;

            Index N = _shape[0];
            Index C = _shape[1];
            Index H = _shape[2];
            Index W = _shape[3];

            if (H + 2*padding < kH || W + 2*padding < kW) {
                throw std::invalid_argument("pool kernel does not fit in padded input");
            }
            Index oH = (H + 2*padding - kH) / stride + 1;
            Index oW = (W + 2*padding - kW) / stride + 1;

            Tensor<Val> out({N, C, oH, oW}, _device);

            if (_device == Device::CPU) {
                const Val* in_  = _storage->data();
                Val*       out_ = out._storage->data();
                for (Index n = 0; n < N; ++n) {
                    for (Index c = 0; c < C; ++c) {
                        for (Index oh = 0; oh < oH; ++oh) {
                            for (Index ow = 0; ow < oW; ++ow) {
                                Val   acc   = max_mode ? std::numeric_limits<Val>::lowest() : Val(0);
                                Index count = 0;
                                for (Index ki = 0; ki < kH; ++ki) {
                                    for (Index kj = 0; kj < kW; ++kj) {
                                        long long ih = (long long)oh*stride + (long long)ki - (long long)padding;
                                        long long iw = (long long)ow*stride + (long long)kj - (long long)padding;
                                        if (ih < 0 || ih >= (long long)H) continue;
                                        if (iw < 0 || iw >= (long long)W) continue;
                                        Val v = in_[((n*C + c)*H + (Index)ih)*W + (Index)iw];
                                        if (max_mode) acc = std::max(acc, v);
                                        else        { acc += v; ++count; }
                                    }
                                }
                                if (!max_mode) acc = count > 0 ? acc / Val(count) : Val(0);
                                else if (acc == std::numeric_limits<Val>::lowest()) acc = Val(0);
                                out_[((n*C + c)*oH + oh)*oW + ow] = acc;
                            }
                        }
                    }
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                if (max_mode) {
                    cue::cuda::max_pool2d_forward(_storage->data(), out._storage->data(),
                                                  N, C, H, W, kH, kW, stride, padding);
                } else {
                    cue::cuda::avg_pool2d_forward(_storage->data(), out._storage->data(),
                                                  N, C, H, W, kH, kW, stride, padding);
                }
            }
            return out;
        }

        Tensor<Val> pool2d_backward(const Tensor<Val>& input,
                                    Index kH, Index kW,
                                    Index stride, Index padding,
                                    bool max_mode) const {
            if (input.rank() != 4) {
                throw std::invalid_argument("pool2d_backward expects a 4D input");
            }
            check_same_device(input);
            if (stride == 0) stride = kH;

            Index N = input._shape[0];
            Index C = input._shape[1];
            Index H = input._shape[2];
            Index W = input._shape[3];
            Index oH = _shape[2];
            Index oW = _shape[3];

            Tensor<Val> grad_in(input._shape, _device);

            if (_device == Device::CPU) {
                const Val* g    = _storage->data();
                const Val* in_  = input._storage->data();
                Val*       gi   = grad_in._storage->data();
                std::fill_n(gi, grad_in.size(), Val(0));
                for (Index n = 0; n < N; ++n) {
                    for (Index c = 0; c < C; ++c) {
                        for (Index oh = 0; oh < oH; ++oh) {
                            for (Index ow = 0; ow < oW; ++ow) {
                                Val   m       = std::numeric_limits<Val>::lowest();
                                long long best_ih = -1, best_iw = -1;
                                if (max_mode) {
                                    for (Index ki = 0; ki < kH; ++ki) {
                                        for (Index kj = 0; kj < kW; ++kj) {
                                            long long ih = (long long)oh*stride + (long long)ki - (long long)padding;
                                            long long iw = (long long)ow*stride + (long long)kj - (long long)padding;
                                            if (ih < 0 || ih >= (long long)H) continue;
                                            if (iw < 0 || iw >= (long long)W) continue;
                                            Val v = in_[((n*C + c)*H + (Index)ih)*W + (Index)iw];
                                            if (v > m) {
                                                m       = v;
                                                best_ih = ih;
                                                best_iw = iw;
                                            }
                                        }
                                    }
                                    if (best_ih < 0) continue;
                                    Index in_idx = ((n*C + c)*H + (Index)best_ih)*W + (Index)best_iw;
                                    gi[in_idx] += g[((n*C + c)*oH + oh)*oW + ow];
                                }
                            }
                        }
                    }
                }
            } else if constexpr (std::is_same_v<Val, float>) {
                cue::cuda::max_pool2d_backward(_storage->data(), input._storage->data(),
                                               grad_in._storage->data(),
                                               N, C, H, W, kH, kW, stride, padding);
            }
            return grad_in;
        }

        Index index(std::initializer_list<Index> coord) const {
            if (coord.size() != _shape.size()) {
                throw std::out_of_range("Coordinate rank does not match tensor rank");
            }

            Index flat = 0;
            Index dim  = 0;

            for (Index c : coord) {
                if (c >= _shape[dim]) {
                    throw std::out_of_range("Tensor coordinate out of bounds");
                }

                flat += c * _stride[dim];
                ++dim;
            }

            return flat;
        }

        Index compute_assign_stride() {
            _stride.assign(_shape.size(), 1);

            Index product = 1;
            for (int i = static_cast<int>(_shape.size()) - 1; i >= 0; --i) {
                _stride[i] = product;
                product *= _shape[i];
            }

            return product;
        }

        void init() {
            Index product = compute_assign_stride();
            _storage = std::make_shared<TensorStorage<Val>>(product, _device);
        }

        void check_same_shape(const Tensor<Val>& other) const {
            if (_shape != other._shape) {
                throw std::invalid_argument("Tensor shape mismatch");
            }
        }

        void check_same_device(const Tensor<Val>& other) const {
            if (_device != other._device) {
                throw std::invalid_argument("Tensor devices differ");
            }
        }

        void require_cpu(const char* what) const {
            if (_device != Device::CPU) {
                throw std::runtime_error(std::string(what) +
                                         " requires a CPU tensor; call to_cpu() first");
            }
        }

        std::shared_ptr<TensorStorage<Val>> _storage;
        std::vector<Index>                  _shape;
        std::vector<Index>                  _stride;
        Device                              _device {Device::CPU};
};
