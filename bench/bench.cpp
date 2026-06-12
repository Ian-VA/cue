#include "../nn/nn.hpp"
#include "../tensor/tensor.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using T = Tensor<float>;
using clk = std::chrono::high_resolution_clock;

static std::vector<std::string> g_filters;

static bool want(const char *group) {
    if (g_filters.empty()) return true;
    for (const auto &f : g_filters)
        if (std::strstr(group, f.c_str())) return true;
    return false;
}

template <typename Fn>
static double ms(Fn fn, Device dev, int iters) {
    int warmup = dev == Device::CUDA ? 3 : 1;
    for (int i = 0; i < warmup; ++i) fn();
    if (dev == Device::CUDA) cue::cuda::device_synchronize();

    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) fn();
    if (dev == Device::CUDA) cue::cuda::device_synchronize();
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count() / iters;
}

static void header(const char *section) {
    printf("\n== %s ==\n", section);
    printf("%-26s %11s %11s %9s %14s\n", "op (ms/call)", "cpu", "gpu", "speedup",
           "gpu rate");
}

template <typename CpuFn, typename GpuFn>
static void row(const char *tag, CpuFn cpu, GpuFn gpu, int iters,
                double work = 0.0, const char *unit = "") {
    double c = ms(cpu, Device::CPU, iters);
    double g = ms(gpu, Device::CUDA, iters);
    char rate[32] = "-";
    if (work > 0.0)
        snprintf(rate, sizeof rate, "%.1f %s", work / (g * 1e-3) / 1e9, unit);
    printf("%-26s %11.4f %11.4f %8.1fx %14s\n", tag, c, g, c / g, rate);
}

static void bench_transfer() {
    if (!want("transfer")) return;
    printf("\n== host <-> device transfer ==\n");
    printf("%-26s %11s %14s\n", "bytes", "ms/call", "GB/s");
    for (Index n : {1u << 16, 1u << 20, 1u << 22}) {
        auto host = T::randn({n});
        auto dev = host.to_cuda();
        double bytes = (double)n * sizeof(float);
        double h2d = ms([&] { (void)host.to_cuda(); }, Device::CUDA, 20);
        double d2h = ms([&] { (void)dev.to_cpu(); }, Device::CUDA, 20);
        char tag[48];
        snprintf(tag, sizeof tag, "H2D %zu", (size_t)n);
        printf("%-26s %11.4f %14.1f\n", tag, h2d, bytes / (h2d * 1e-3) / 1e9);
        snprintf(tag, sizeof tag, "D2H %zu", (size_t)n);
        printf("%-26s %11.4f %14.1f\n", tag, d2h, bytes / (d2h * 1e-3) / 1e9);
    }
}

static void bench_elementwise() {
    if (!(want("elementwise") || want("activation"))) return;
    if (want("elementwise")) {
        header("element-wise (bandwidth bound)");
        for (Index n : {1u << 16, 1u << 20, 1u << 24}) {
            auto a = T::randn({n}), b = T::randn({n});
            auto ag = a.to_cuda(), bg = b.to_cuda();
            int it = n <= (1u << 20) ? 200 : 50;
            double rw = 3.0 * n * sizeof(float);  // 2 reads + 1 write
            char tag[48];
            snprintf(tag, sizeof tag, "add %zu", (size_t)n);
            row(tag, [&] { a.add(b); }, [&] { ag.add(bg); }, it, rw, "GB/s");
            snprintf(tag, sizeof tag, "mul %zu", (size_t)n);
            row(tag, [&] { a.mul(b); }, [&] { ag.mul(bg); }, it, rw, "GB/s");
        }
    }
    if (want("activation")) {
        header("activations (bandwidth bound)");
        Index n = 1u << 22;
        auto a = T::randn({n});
        auto ag = a.to_cuda();
        int it = 100;
        double rw = 2.0 * n * sizeof(float);  // 1 read + 1 write
        char tag[48];
        snprintf(tag, sizeof tag, "relu %zu", (size_t)n);
        row(tag, [&] { a.relu(); }, [&] { ag.relu(); }, it, rw, "GB/s");
        snprintf(tag, sizeof tag, "sigmoid %zu", (size_t)n);
        row(tag, [&] { a.sigmoid(); }, [&] { ag.sigmoid(); }, it, rw, "GB/s");
        snprintf(tag, sizeof tag, "exp %zu", (size_t)n);
        row(tag, [&] { a.exp(); }, [&] { ag.exp(); }, it, rw, "GB/s");
    }
}

// reduction (bandwidth bound: reads the whole tensor)
static void bench_reduction() {
    if (!want("reduction")) return;
    header("reduction (bandwidth bound)");
    for (Index n : {1u << 20, 1u << 24}) {
        auto a = T::randn({n});
        auto ag = a.to_cuda();
        double rd = (double)n * sizeof(float);
        char tag[48];
        snprintf(tag, sizeof tag, "sum %zu", (size_t)n);
        row(tag, [&] { volatile float s = a.sum(); (void)s; },
            [&] { volatile float s = ag.sum(); (void)s; }, 50, rd, "GB/s");
    }
}

// matmul & transpose
static void bench_linalg() {
    if (!(want("matmul") || want("transpose"))) return;
    if (want("matmul")) {
        header("matmul (compute bound)");
        for (Index n : {128u, 256u, 512u, 1024u}) {
            auto a = T::randn({n, n}), b = T::randn({n, n});
            auto ag = a.to_cuda(), bg = b.to_cuda();
            int it = n <= 128 ? 50 : n <= 256 ? 30 : n <= 512 ? 10 : 3;
            double flops = 2.0 * n * n * n;
            char tag[48];
            snprintf(tag, sizeof tag, "matmul %zux%zu", (size_t)n, (size_t)n);
            row(tag, [&] { a.matmul(b); }, [&] { ag.matmul(bg); }, it, flops,
                "GFLOP/s");
        }
    }
    if (want("transpose")) {
        header("transpose (bandwidth bound)");
        for (Index n : {512u, 1024u, 2048u}) {
            auto a = T::randn({n, n});
            auto ag = a.to_cuda();
            double rw = 2.0 * n * n * sizeof(float);
            char tag[48];
            snprintf(tag, sizeof tag, "transpose %zux%zu", (size_t)n, (size_t)n);
            row(tag, [&] { a.transpose(0, 1); }, [&] { ag.transpose(0, 1); }, 20,
                rw, "GB/s");
        }
    }
}

// softmax & broadcasting
static void bench_softmax() {
    if (!want("softmax")) return;
    header("softmax (row reduction)");
    for (Index rows : {256u, 1024u}) {
        Index cols = 1024;
        auto a = T::randn({rows, cols});
        auto ag = a.to_cuda();
        double rw = 2.0 * rows * cols * sizeof(float);
        char tag[48];
        snprintf(tag, sizeof tag, "softmax %zux%zu", (size_t)rows, (size_t)cols);
        row(tag, [&] { a.softmax(); }, [&] { ag.softmax(); }, 50, rw, "GB/s");
    }
}

static void bench_broadcast() {
    if (!want("broadcast")) return;
    header("broadcasting");
    auto a = T::randn({1024, 1024}), v = T::randn({1024});
    auto ag = a.to_cuda(), vg = v.to_cuda();
    double rw = 3.0 * 1024 * 1024 * sizeof(float);
    row("bcast 1024x1024 + row", [&] { a.add(v); }, [&] { ag.add(vg); }, 30, rw,
        "GB/s");
}

// convolution & pooling
static void bench_conv() {
    if (!want("conv")) return;
    header("conv2d forward / backward");
    struct Cfg { Index N, Cin, H, W, Cout, k; };
    for (Cfg c : {Cfg{8, 3, 32, 32, 16, 3}, Cfg{16, 16, 32, 32, 32, 3}}) {
        auto x = T::randn({c.N, c.Cin, c.H, c.W});
        auto w = T::randn({c.Cout, c.Cin, c.k, c.k});
        auto xg = x.to_cuda(), wg = w.to_cuda();
        Index oH = c.H - c.k + 1, oW = c.W - c.k + 1;
        double flops =
            2.0 * c.N * c.Cout * oH * oW * c.Cin * c.k * c.k;
        char tag[64];
        snprintf(tag, sizeof tag, "fwd N%zu C%zu->%zu", (size_t)c.N,
                 (size_t)c.Cin, (size_t)c.Cout);
        row(tag, [&] { x.conv2d(w, 1, 0); }, [&] { xg.conv2d(wg, 1, 0); }, 10,
            flops, "GFLOP/s");

        // backward passes use the forward output's gradient
        auto y = x.conv2d(w, 1, 0);
        auto yg = xg.conv2d(wg, 1, 0);
        snprintf(tag, sizeof tag, "bwd-input N%zu C%zu", (size_t)c.N,
                 (size_t)c.Cin);
        row(tag,
            [&] { y.conv2d_backward_input(w, x.shape(), 1, 0); },
            [&] { yg.conv2d_backward_input(wg, xg.shape(), 1, 0); }, 10, flops,
            "GFLOP/s");
        snprintf(tag, sizeof tag, "bwd-kernel N%zu C%zu", (size_t)c.N,
                 (size_t)c.Cin);
        row(tag,
            [&] { y.conv2d_backward_kernel(x, w.shape(), 1, 0); },
            [&] { yg.conv2d_backward_kernel(xg, wg.shape(), 1, 0); }, 10, flops,
            "GFLOP/s");
    }
}

static void bench_pool() {
    if (!want("pool")) return;
    header("max_pool2d forward / backward");
    auto x = T::randn({16, 16, 64, 64});
    auto xg = x.to_cuda();
    row("maxpool fwd 2x2", [&] { x.max_pool2d(2, 2, 2, 0); },
        [&] { xg.max_pool2d(2, 2, 2, 0); }, 20);
    auto y = x.max_pool2d(2, 2, 2, 0);
    auto yg = xg.max_pool2d(2, 2, 2, 0);
    row("maxpool bwd 2x2", [&] { y.max_pool2d_backward(x, 2, 2, 2, 0); },
        [&] { yg.max_pool2d_backward(xg, 2, 2, 2, 0); }, 20);
}

// end-to-end training step (forward + backward + optimizer step)
using namespace cue::nn;

static std::shared_ptr<Sequential> make_mlp(Device d) {
    auto m = std::make_shared<Sequential>();
    m->emplace<Linear>(1024, 1024, d).emplace<ReLU>()
        .emplace<Linear>(1024, 512, d).emplace<ReLU>()
        .emplace<Linear>(512, 10, d);
    return m;
}

static std::shared_ptr<Sequential> make_cnn(Device d) {
    auto m = std::make_shared<Sequential>();
    m->emplace<Conv2d>(3, 16, 3, 1, 1, d).emplace<ReLU>()
        .emplace<MaxPool2d>(2, 2)
        .emplace<Flatten>()
        .emplace<Linear>(16 * 16 * 16, 10, d);
    return m;
}

static double train_step_ms(std::shared_ptr<Sequential> model, const T &x,
                            const Tensor<int> &labels, Device d, int iters) {
    SGD opt(model->parameters(), 0.01f);
    auto step = [&] {
        CrossEntropyLoss loss;
        auto logits = model->forward(x);
        loss.forward(logits, labels);
        model->backward(loss.backward());
        opt.step();
        opt.zero_grad();
    };
    return ms(step, d, iters);
}

static Tensor<int> make_labels(Index n, int classes) {
    Tensor<int> labels({n}, Device::CPU);
    for (Index i = 0; i < n; ++i) labels.at({i}) = (int)(i % classes);
    return labels;
}

static void bench_train() {
    if (!want("train")) return;
    header("training step (fwd + bwd + sgd)");

    {  // MLP, batch 256
        Index N = 256;
        auto x = T::randn({N, 1024});
        auto xg = x.to_cuda();
        auto labels = make_labels(N, 10);
        auto cpu = make_mlp(Device::CPU);
        auto gpu = make_mlp(Device::CUDA);
        double c = train_step_ms(cpu, x, labels, Device::CPU, 10);
        double g = train_step_ms(gpu, xg, labels, Device::CUDA, 10);
        printf("%-26s %11.4f %11.4f %8.1fx %14s\n", "mlp N256", c, g, c / g, "-");
    }
    {  // CNN, batch 32, 3x32x32
        Index N = 32;
        auto x = T::randn({N, 3, 32, 32});
        auto xg = x.to_cuda();
        auto labels = make_labels(N, 10);
        auto cpu = make_cnn(Device::CPU);
        auto gpu = make_cnn(Device::CUDA);
        double c = train_step_ms(cpu, x, labels, Device::CPU, 5);
        double g = train_step_ms(gpu, xg, labels, Device::CUDA, 5);
        printf("%-26s %11.4f %11.4f %8.1fx %14s\n", "cnn N32 3x32x32", c, g,
               c / g, "-");
    }
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) g_filters.emplace_back(argv[i]);
    if (!g_filters.empty() && g_filters[0] == "list") {
        printf("groups: transfer elementwise activation reduction matmul "
               "transpose softmax broadcast conv pool train\n");
        return 0;
    }

    if (g_filters.size() == 1 && g_filters[0] == "all") g_filters.clear();

    printf("Starting performance analysis...\n");

    bench_transfer();
    bench_elementwise();
    bench_reduction();
    bench_linalg();
    bench_softmax();
    bench_broadcast();
    bench_conv();
    bench_pool();
    bench_train();
    return 0;
}
