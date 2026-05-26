#include "nn.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cue::nn {

static Tensor<float> kaiming_normal(const std::vector<Index>& shape,
                                    Index fan_in, Device d) {
    float stddev = std::sqrt(2.0f / (float)fan_in);
    return Tensor<float>::randn(shape, 0.0f, stddev, d);
}

// Linear

Linear::Linear(Index in_features, Index out_features, Device d) {
    _weight = Parameter(kaiming_normal({in_features, out_features}, in_features, d));
    _bias   = Parameter(Tensor<float>::zeros({out_features}, d));
}

Tensor<float> Linear::forward(const Tensor<float>& x) {
    _input = x;
    return x.matmul(_weight.value).add_bias(_bias.value);
}

Tensor<float> Linear::backward(const Tensor<float>& grad_out) {
    auto wt = _weight.value.transpose(0, 1);
    auto xt = _input.transpose(0, 1);

    _weight.grad = xt.matmul(grad_out);
    _bias.grad   = grad_out.sum_axis0();

    return grad_out.matmul(wt);
}

std::vector<Parameter*> Linear::parameters() {
    return {&_weight, &_bias};
}

void Linear::to(Device d) {
    _weight.to(d);
    _bias.to(d);
}

// Conv2d

Conv2d::Conv2d(Index in_channels, Index out_channels, Index kernel_size,
               Index stride, Index padding, Device d)
        : _stride(stride), _padding(padding) {
    Index fan_in = in_channels * kernel_size * kernel_size;
    _weight = Parameter(kaiming_normal({out_channels, in_channels, kernel_size, kernel_size},
                                       fan_in, d));
    _bias   = Parameter(Tensor<float>::zeros({out_channels}, d));
}

Tensor<float> Conv2d::forward(const Tensor<float>& x) {
    _input = x;
    return x.conv2d(_weight.value, _stride, _padding).add_bias(_bias.value);
}

Tensor<float> Conv2d::backward(const Tensor<float>& grad_out) {
    _bias.grad   = grad_out.sum_to_channel();
    _weight.grad = grad_out.conv2d_backward_kernel(_input, _weight.value.shape(),
                                                   _stride, _padding);
    return grad_out.conv2d_backward_input(_weight.value, _input.shape(),
                                          _stride, _padding);
}

std::vector<Parameter*> Conv2d::parameters() {
    return {&_weight, &_bias};
}

void Conv2d::to(Device d) {
    _weight.to(d);
    _bias.to(d);
}

// ReLU

Tensor<float> ReLU::forward(const Tensor<float>& x) {
    _input = x;
    return x.relu();
}

Tensor<float> ReLU::backward(const Tensor<float>& grad_out) {
    return grad_out.relu_backward(_input);
}

// MaxPool2d

MaxPool2d::MaxPool2d(Index kH, Index kW, Index stride, Index padding)
        : _kH(kH), _kW(kW),
          _stride(stride == 0 ? kH : stride),
          _padding(padding) {}

Tensor<float> MaxPool2d::forward(const Tensor<float>& x) {
    _input = x;
    return x.max_pool2d(_kH, _kW, _stride, _padding);
}

Tensor<float> MaxPool2d::backward(const Tensor<float>& grad_out) {
    return grad_out.max_pool2d_backward(_input, _kH, _kW, _stride, _padding);
}

// Flatten

Flatten::Flatten(Index start_dim) : _start_dim(start_dim) {}

Tensor<float> Flatten::forward(const Tensor<float>& x) {
    _orig_shape = x.shape();
    return x.flatten(_start_dim);
}

Tensor<float> Flatten::backward(const Tensor<float>& grad_out) {
    return grad_out.view(_orig_shape);
}

// Sequential

Tensor<float> Sequential::forward(const Tensor<float>& x) {
    Tensor<float> out = x;
    for (auto& m : _layers) out = m->forward(out);
    return out;
}

Tensor<float> Sequential::backward(const Tensor<float>& grad_out) {
    Tensor<float> g = grad_out;
    for (auto it = _layers.rbegin(); it != _layers.rend(); ++it) {
        g = (*it)->backward(g);
    }
    return g;
}

std::vector<Parameter*> Sequential::parameters() {
    std::vector<Parameter*> ps;
    for (auto& m : _layers) {
        auto sub = m->parameters();
        ps.insert(ps.end(), sub.begin(), sub.end());
    }
    return ps;
}

void Sequential::to(Device d) {
    for (auto& m : _layers) m->to(d);
}

// CrossEntropyLoss

float CrossEntropyLoss::forward(const Tensor<float>& logits,
                                const Tensor<int>& labels) {
    if (logits.rank() != 2) {
        throw std::invalid_argument("CrossEntropyLoss expects logits of shape (N, C)");
    }
    if (labels.rank() != 1 || labels.shape()[0] != logits.shape()[0]) {
        throw std::invalid_argument("labels shape must match logits batch size");
    }

    _batch  = logits.shape()[0];
    Index C = logits.shape()[1];

    _softmax = logits.softmax();
    _labels  = labels;

    Tensor<float> host = _softmax.on_cuda() ? _softmax.to_cpu() : _softmax;
    double total = 0.0;
    for (Index n = 0; n < _batch; ++n) {
        int label = labels.data()[n];
        if (label < 0 || (Index)label >= C) {
            throw std::out_of_range("CrossEntropyLoss label out of range");
        }
        float p = host.data()[n*C + label];
        total -= std::log(std::max(p, 1e-12f));
    }
    return (float)(total / (double)_batch);
}

Tensor<float> CrossEntropyLoss::backward() {
    Index N = _batch;
    Index C = _softmax.shape()[1];

    Tensor<float> grad(_softmax.shape(), Device::CPU);
    Tensor<float> host = _softmax.on_cuda() ? _softmax.to_cpu() : _softmax;

    float inv_n = 1.0f / (float)N;
    for (Index n = 0; n < N; ++n) {
        for (Index c = 0; c < C; ++c) {
            grad.data()[n*C + c] = host.data()[n*C + c] * inv_n;
        }
        int label = _labels.data()[n];
        grad.data()[n*C + label] -= inv_n;
    }
    return _softmax.on_cuda() ? grad.to_cuda() : grad;
}

// MSELoss

float MSELoss::forward(const Tensor<float>& pred, const Tensor<float>& target) {
    _diff = pred - target;
    float s = (_diff * _diff).sum();
    return s / (float)_diff.size();
}

Tensor<float> MSELoss::backward() {
    return _diff * (2.0f / (float)_diff.size());
}

// SGD

SGD::SGD(std::vector<Parameter*> params, float lr)
        : _params(std::move(params)), _lr(lr) {}

void SGD::step() {
    for (auto* p : _params) {
        p->value = p->value - p->grad * _lr;
    }
}

void SGD::zero_grad() {
    for (auto* p : _params) p->zero_grad();
}

}
