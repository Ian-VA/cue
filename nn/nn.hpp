#pragma once

#include "../tensor/tensor.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace cue::nn {

struct Parameter {
    Tensor<float> value;
    Tensor<float> grad;

    Parameter() = default;

    explicit Parameter(Tensor<float> v)
            : value(std::move(v)),
              grad(Tensor<float>::zeros(value.shape(), value.device())) {}

    void zero_grad() {
        grad = Tensor<float>::zeros(value.shape(), value.device());
    }

    void to(Device d) {
        if (value.device() == d) return;
        value = value.to(d);
        grad  = grad.to(d);
    }
};

class Module {
    public:
        virtual ~Module() = default;
        virtual Tensor<float> forward(const Tensor<float>& x) = 0;
        virtual Tensor<float> backward(const Tensor<float>& grad_out) = 0;
        virtual std::vector<Parameter*> parameters() { return {}; }
        virtual void to(Device) {}
};

class Linear : public Module {
    public:
        Linear(Index in_features, Index out_features, Device d = Device::CPU);

        Tensor<float>           forward(const Tensor<float>& x) override;
        Tensor<float>           backward(const Tensor<float>& grad_out) override;
        std::vector<Parameter*> parameters() override;
        void                    to(Device d) override;

        Parameter& weight() { return _weight; }
        Parameter& bias()   { return _bias; }

    private:
        Parameter     _weight;
        Parameter     _bias;
        Tensor<float> _input;
};

class Conv2d : public Module {
    public:
        Conv2d(Index in_channels, Index out_channels, Index kernel_size,
               Index stride = 1, Index padding = 0, Device d = Device::CPU);

        Tensor<float>           forward(const Tensor<float>& x) override;
        Tensor<float>           backward(const Tensor<float>& grad_out) override;
        std::vector<Parameter*> parameters() override;
        void                    to(Device d) override;

        Parameter& weight() { return _weight; }
        Parameter& bias()   { return _bias; }

    private:
        Parameter     _weight;
        Parameter     _bias;
        Index         _stride;
        Index         _padding;
        Tensor<float> _input;
};

class ReLU : public Module {
    public:
        Tensor<float> forward(const Tensor<float>& x) override;
        Tensor<float> backward(const Tensor<float>& grad_out) override;
    private:
        Tensor<float> _input;
};

class MaxPool2d : public Module {
    public:
        MaxPool2d(Index kH, Index kW, Index stride = 0, Index padding = 0);
        Tensor<float> forward(const Tensor<float>& x) override;
        Tensor<float> backward(const Tensor<float>& grad_out) override;
    private:
        Index         _kH, _kW, _stride, _padding;
        Tensor<float> _input;
};

class Flatten : public Module {
    public:
        explicit Flatten(Index start_dim = 1);
        Tensor<float> forward(const Tensor<float>& x) override;
        Tensor<float> backward(const Tensor<float>& grad_out) override;
    private:
        Index              _start_dim;
        std::vector<Index> _orig_shape;
};

class Sequential : public Module {
    public:
        Sequential() = default;

        Sequential& add(std::shared_ptr<Module> m) {
            _layers.push_back(std::move(m));
            return *this;
        }

        template <typename T, typename... Args>
        Sequential& emplace(Args&&... args) {
            _layers.push_back(std::make_shared<T>(std::forward<Args>(args)...));
            return *this;
        }

        Tensor<float>           forward(const Tensor<float>& x) override;
        Tensor<float>           backward(const Tensor<float>& grad_out) override;
        std::vector<Parameter*> parameters() override;
        void                    to(Device d) override;

        size_t                  size()   const { return _layers.size(); }
        Module&                 at(size_t i)   { return *_layers[i]; }

    private:
        std::vector<std::shared_ptr<Module>> _layers;
};

class CrossEntropyLoss {
    public:
        float         forward(const Tensor<float>& logits, const Tensor<int>& labels);
        Tensor<float> backward();

    private:
        Tensor<float> _softmax;
        Tensor<int>   _labels;
        Index         _batch {0};
};

class MSELoss {
    public:
        float         forward(const Tensor<float>& pred, const Tensor<float>& target);
        Tensor<float> backward();

    private:
        Tensor<float> _diff;
};

class SGD {
    public:
        SGD(std::vector<Parameter*> params, float lr);

        void step();
        void zero_grad();

        float lr() const { return _lr; }
        void  set_lr(float lr) { _lr = lr; }

    private:
        std::vector<Parameter*> _params;
        float                   _lr;
};

}
