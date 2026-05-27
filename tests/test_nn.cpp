#include "test_util.hpp"

#include "../nn/nn.hpp"

#include <functional>

using cue::nn::Linear;
using cue::nn::Conv2d;
using cue::nn::ReLU;
using cue::nn::MaxPool2d;
using cue::nn::Flatten;
using cue::nn::Sequential;
using cue::nn::CrossEntropyLoss;
using cue::nn::MSELoss;
using cue::nn::SGD;

using ScalarFn = std::function<float(const Tensor<float>&)>;

static Tensor<float> numerical_grad(ScalarFn f, const Tensor<float>& x,
                                    float eps = 1e-3f) {
    Tensor<float> xc = x.clone();
    Tensor<float> grad(xc.shape(), Device::CPU);
    for (Index i = 0; i < xc.size(); ++i) {
        float orig = xc.data()[i];
        xc.data()[i] = orig + eps;
        float fp = f(xc);
        xc.data()[i] = orig - eps;
        float fm = f(xc);
        xc.data()[i] = orig;
        grad.data()[i] = (fp - fm) / (2.0f * eps);
    }
    return grad;
}


TEST(linear, forward_shape) {
    Linear layer(4, 3);
    auto x = Tensor<float>::randn({5, 4});
    auto y = layer.forward(x);
    REQUIRE_EQ(y.rank(), 2u);
    REQUIRE_EQ(y.shape()[0], 5u);
    REQUIRE_EQ(y.shape()[1], 3u);
}

TEST(linear, gradcheck_input) {
    Linear layer(3, 2);
    auto x = Tensor<float>::randn({4, 3});

    auto y    = layer.forward(x);
    auto gout = Tensor<float>::ones(y.shape());
    auto gx_a = layer.backward(gout);

    auto gx_n = numerical_grad(
        [&](const Tensor<float>& xin) { return layer.forward(xin).sum(); },
        x);

    expect_tensors_equal(gx_a, gx_n, 1e-2, "linear grad x");
}

TEST(linear, gradcheck_weight) {
    Linear layer(3, 2);
    auto x = Tensor<float>::randn({4, 3});

    auto y    = layer.forward(x);
    auto gout = Tensor<float>::ones(y.shape());
    (void)layer.backward(gout);
    auto gw_a = layer.weight().grad.clone();

    auto gw_n = numerical_grad(
        [&](const Tensor<float>& w) {
            auto saved = layer.weight().value;
            layer.weight().value = w;
            float r = layer.forward(x).sum();
            layer.weight().value = saved;
            return r;
        },
        layer.weight().value);

    expect_tensors_equal(gw_a, gw_n, 1e-2, "linear grad w");
}

TEST(linear, gradcheck_bias) {
    Linear layer(3, 2);
    auto x = Tensor<float>::randn({4, 3});

    auto y    = layer.forward(x);
    auto gout = Tensor<float>::ones(y.shape());
    (void)layer.backward(gout);
    auto gb_a = layer.bias().grad.clone();

    auto gb_n = numerical_grad(
        [&](const Tensor<float>& b) {
            auto saved = layer.bias().value;
            layer.bias().value = b;
            float r = layer.forward(x).sum();
            layer.bias().value = saved;
            return r;
        },
        layer.bias().value);

    expect_tensors_equal(gb_a, gb_n, 1e-2, "linear grad b");
}


TEST(relu, forward_backward) {
    ReLU layer;
    auto x = Tensor<float>::from_values({-2, -1, 0, 1, 2});
    auto y = layer.forward(x);
    expect_tensor_close(y, {0, 0, 0, 1, 2}, 1e-6, "relu fwd");

    auto gout = Tensor<float>::from_values({1, 1, 1, 1, 1});
    auto gin  = layer.backward(gout);
    expect_tensor_close(gin, {0, 0, 0, 1, 1}, 1e-6, "relu bwd");
}


TEST(conv2d, forward_shape) {
    Conv2d layer(1, 4, /*kernel=*/2, /*stride=*/1, /*padding=*/1);
    auto x = Tensor<float>::randn({3, 1, 2, 2});
    auto y = layer.forward(x);
    REQUIRE_EQ(y.shape()[0], 3u);
    REQUIRE_EQ(y.shape()[1], 4u);
    REQUIRE_EQ(y.shape()[2], 3u);
    REQUIRE_EQ(y.shape()[3], 3u);
}

TEST(conv2d, gradcheck_input) {
    Conv2d layer(2, 3, 2, 1, 0);
    auto x = Tensor<float>::randn({2, 2, 3, 3});

    auto y    = layer.forward(x);
    auto gout = Tensor<float>::ones(y.shape());
    auto gx_a = layer.backward(gout);

    auto gx_n = numerical_grad(
        [&](const Tensor<float>& xin) { return layer.forward(xin).sum(); },
        x, 2e-3f);

    expect_tensors_equal(gx_a, gx_n, 2e-2, "conv2d grad x");
}

TEST(conv2d, gradcheck_weight) {
    Conv2d layer(2, 3, 2, 1, 0);
    auto x = Tensor<float>::randn({2, 2, 3, 3});

    auto y    = layer.forward(x);
    auto gout = Tensor<float>::ones(y.shape());
    (void)layer.backward(gout);
    auto gw_a = layer.weight().grad.clone();

    auto gw_n = numerical_grad(
        [&](const Tensor<float>& w) {
            auto saved = layer.weight().value;
            layer.weight().value = w;
            float r = layer.forward(x).sum();
            layer.weight().value = saved;
            return r;
        },
        layer.weight().value, 2e-3f);

    expect_tensors_equal(gw_a, gw_n, 2e-2, "conv2d grad w");
}

TEST(conv2d, gradcheck_bias) {
    Conv2d layer(2, 3, 2, 1, 0);
    auto x = Tensor<float>::randn({2, 2, 3, 3});

    auto y    = layer.forward(x);
    auto gout = Tensor<float>::ones(y.shape());
    (void)layer.backward(gout);
    auto gb_a = layer.bias().grad.clone();

    auto gb_n = numerical_grad(
        [&](const Tensor<float>& b) {
            auto saved = layer.bias().value;
            layer.bias().value = b;
            float r = layer.forward(x).sum();
            layer.bias().value = saved;
            return r;
        },
        layer.bias().value, 2e-3f);

    expect_tensors_equal(gb_a, gb_n, 2e-2, "conv2d grad b");
}


TEST(maxpool, forward_backward) {
    MaxPool2d layer(2, 2);
    auto x = Tensor<float>::zeros({1, 1, 2, 2});
    x.data()[0] = 1; x.data()[1] = 3; x.data()[2] = 2; x.data()[3] = 4;
    auto y = layer.forward(x);
    REQUIRE_EQ(y.size(), 1u);
    REQUIRE_EQ(y.data()[0], 4.0f);

    auto gout = Tensor<float>::ones({1, 1, 1, 1}) * 5.0f;
    auto gin  = layer.backward(gout);
    expect_tensor_close(gin, {0, 0, 0, 5}, 1e-6, "maxpool grad");
}

TEST(maxpool, gradcheck) {
    MaxPool2d layer(2, 2);
    Tensor<float> x({2, 2, 4, 4}, Device::CPU);
    for (Index i = 0; i < x.size(); ++i) x.data()[i] = (float)i * 0.13f;

    auto y    = layer.forward(x);
    auto gout = Tensor<float>::ones(y.shape());
    auto gx_a = layer.backward(gout);

    auto gx_n = numerical_grad(
        [&](const Tensor<float>& xin) { return layer.forward(xin).sum(); },
        x, 1e-3f);

    expect_tensors_equal(gx_a, gx_n, 1e-2, "maxpool grad");
}


TEST(flatten, forward_backward) {
    Flatten layer(1);
    auto x = Tensor<float>::randn({3, 2, 2, 2});
    auto y = layer.forward(x);
    REQUIRE_EQ(y.rank(), 2u);
    REQUIRE_EQ(y.shape()[0], 3u);
    REQUIRE_EQ(y.shape()[1], 8u);

    auto gout = Tensor<float>::ones(y.shape());
    auto gin  = layer.backward(gout);
    REQUIRE(gin.shape() == x.shape());
}


TEST(sequential, forward_backward) {
    Sequential model;
    model.emplace<Linear>(4, 8);
    model.emplace<ReLU>();
    model.emplace<Linear>(8, 3);

    auto x = Tensor<float>::randn({5, 4});
    auto y = model.forward(x);
    REQUIRE_EQ(y.shape()[0], 5u);
    REQUIRE_EQ(y.shape()[1], 3u);

    auto gout = Tensor<float>::ones(y.shape());
    auto gx_a = model.backward(gout);

    auto gx_n = numerical_grad(
        [&](const Tensor<float>& xin) { return model.forward(xin).sum(); },
        x);

    expect_tensors_equal(gx_a, gx_n, 2e-2, "sequential grad x");
    REQUIRE_EQ(model.parameters().size(), 4u);
}


TEST(loss, mse) {
    MSELoss loss;
    auto pred   = Tensor<float>::from_values({1, 2, 3, 4});
    auto target = Tensor<float>::from_values({1, 3, 3, 6});
    float v = loss.forward(pred, target);
    // (0 + 1 + 0 + 4) / 4 = 1.25
    REQUIRE_CLOSE(v, 1.25f, 1e-6);

    auto grad = loss.backward();
    expect_tensor_close(grad, {0.0f, -0.5f, 0.0f, -1.0f}, 1e-6, "mse grad");
}

TEST(loss, mse_gradcheck) {
    MSELoss loss;
    auto pred   = Tensor<float>::randn({3, 4});
    auto target = Tensor<float>::randn({3, 4});

    (void)loss.forward(pred, target);
    auto g_a = loss.backward();

    auto g_n = numerical_grad(
        [&](const Tensor<float>& p) { return loss.forward(p, target); },
        pred);

    expect_tensors_equal(g_a, g_n, 1e-3, "mse grad");
}

TEST(loss, cross_entropy) {
    CrossEntropyLoss loss;
    auto logits = Tensor<float>::from_values({{1.0f, 2.0f, 3.0f},
                                              {1.0f, 2.0f, 3.0f}});
    Tensor<int> labels({2}, Device::CPU);
    labels.data()[0] = 2;
    labels.data()[1] = 0;

    float v = loss.forward(logits, labels);
    // softmax row = [0.09003, 0.24473, 0.66524]; -log(0.66524)~=0.4076,
    // -log(0.09003)~=2.4076; mean ~= 1.4076
    REQUIRE_CLOSE(v, 1.4076f, 1e-3);
}

TEST(loss, cross_entropy_gradcheck) {
    CrossEntropyLoss loss;
    auto logits = Tensor<float>::randn({4, 3});
    Tensor<int> labels({4}, Device::CPU);
    labels.data()[0] = 0;
    labels.data()[1] = 2;
    labels.data()[2] = 1;
    labels.data()[3] = 0;

    (void)loss.forward(logits, labels);
    auto g_a = loss.backward();

    auto g_n = numerical_grad(
        [&](const Tensor<float>& z) { return loss.forward(z, labels); },
        logits, 1e-3f);

    expect_tensors_equal(g_a, g_n, 2e-3, "ce grad");
}


TEST(sgd, reduces_loss) {
    Sequential model;
    model.emplace<Linear>(4, 8);
    model.emplace<ReLU>();
    model.emplace<Linear>(8, 3);

    auto x = Tensor<float>::randn({16, 4});
    Tensor<int> y({16}, Device::CPU);
    for (Index i = 0; i < 16; ++i) { // toy dataset
        int c = (int)(i % 3);
        y.data()[i] = c;
        x.data()[i*4 + c] += 3.0f;
    }

    CrossEntropyLoss loss;
    SGD opt(model.parameters(), 0.05f);

    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 100; ++step) {
        auto logits = model.forward(x);
        float l = loss.forward(logits, y);
        if (step == 0)   first = l;
        if (step == 99)  last  = l;
        opt.zero_grad();
        auto g = loss.backward();
        model.backward(g);
        opt.step();
    }
    REQUIRE(last < first);
    REQUIRE(last < first * 0.5f);
}


TEST(cuda, linear_matches_cpu) {
    Linear cpu_layer(4, 3, Device::CPU);
    Linear gpu_layer(4, 3, Device::CUDA);
    // Copy cpu weights to gpu layer so both start from the same parameters.
    gpu_layer.weight().value = cpu_layer.weight().value.to_cuda();
    gpu_layer.bias().value   = cpu_layer.bias().value.to_cuda();

    auto x_cpu = Tensor<float>::randn({8, 4});
    auto x_gpu = x_cpu.to_cuda();

    auto y_cpu = cpu_layer.forward(x_cpu);
    auto y_gpu = gpu_layer.forward(x_gpu);
    expect_tensors_equal(y_cpu, y_gpu.to_cpu(), 1e-4, "linear forward parity");

    auto gout_cpu = Tensor<float>::ones(y_cpu.shape());
    auto gout_gpu = gout_cpu.to_cuda();
    auto gx_cpu   = cpu_layer.backward(gout_cpu);
    auto gx_gpu   = gpu_layer.backward(gout_gpu);

    expect_tensors_equal(gx_cpu, gx_gpu.to_cpu(), 1e-4, "linear backward parity");
    expect_tensors_equal(cpu_layer.weight().grad,
                         gpu_layer.weight().grad.to_cpu(),
                         1e-4, "linear grad_w parity");
    expect_tensors_equal(cpu_layer.bias().grad,
                         gpu_layer.bias().grad.to_cpu(),
                         1e-4, "linear grad_b parity");
}

TEST(cuda, conv2d_matches_cpu) {
    Conv2d cpu_layer(2, 4, 3, 1, 1, Device::CPU);
    Conv2d gpu_layer(2, 4, 3, 1, 1, Device::CUDA);
    gpu_layer.weight().value = cpu_layer.weight().value.to_cuda();
    gpu_layer.bias().value   = cpu_layer.bias().value.to_cuda();

    auto x_cpu = Tensor<float>::randn({2, 2, 5, 5});
    auto x_gpu = x_cpu.to_cuda();

    auto y_cpu = cpu_layer.forward(x_cpu);
    auto y_gpu = gpu_layer.forward(x_gpu);
    expect_tensors_equal(y_cpu, y_gpu.to_cpu(), 1e-3, "conv2d forward parity");

    auto gout_cpu = Tensor<float>::ones(y_cpu.shape());
    auto gout_gpu = gout_cpu.to_cuda();
    auto gx_cpu   = cpu_layer.backward(gout_cpu);
    auto gx_gpu   = gpu_layer.backward(gout_gpu);

    expect_tensors_equal(gx_cpu, gx_gpu.to_cpu(), 1e-3, "conv2d grad_x parity");
    expect_tensors_equal(cpu_layer.weight().grad,
                         gpu_layer.weight().grad.to_cpu(),
                         1e-3, "conv2d grad_w parity");
    expect_tensors_equal(cpu_layer.bias().grad,
                         gpu_layer.bias().grad.to_cpu(),
                         1e-3, "conv2d grad_b parity");
}

TEST(cuda, maxpool_matches_cpu) {
    MaxPool2d cpu_layer(2, 2);
    MaxPool2d gpu_layer(2, 2);

    auto x_cpu = Tensor<float>::randn({2, 3, 6, 6});
    auto x_gpu = x_cpu.to_cuda();

    auto y_cpu = cpu_layer.forward(x_cpu);
    auto y_gpu = gpu_layer.forward(x_gpu);
    expect_tensors_equal(y_cpu, y_gpu.to_cpu(), 1e-6, "maxpool forward parity");

    auto gout_cpu = Tensor<float>::ones(y_cpu.shape());
    auto gout_gpu = gout_cpu.to_cuda();
    auto gx_cpu   = cpu_layer.backward(gout_cpu);
    auto gx_gpu   = gpu_layer.backward(gout_gpu);
    expect_tensors_equal(gx_cpu, gx_gpu.to_cpu(), 1e-5, "maxpool grad parity");
}


int main() { return run_all_tests(); }
