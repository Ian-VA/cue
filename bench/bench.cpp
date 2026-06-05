// cpu vs gpu microbenchmarks gpu timings sync the device so async kernel
// launches are actually waited on the first few calls are discarded as warmup
#include "../tensor/tensor.hpp"

#include <chrono>
#include <cstdio>

using T   = Tensor<float>;
using clk = std::chrono::high_resolution_clock;

template <typename Fn>
static double ms(Fn fn, Device dev, int iters) {
    for (int i = 0; i < 3; ++i) fn();
    if (dev == Device::CUDA) cue::cuda::device_synchronize();
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) fn();
    if (dev == Device::CUDA) cue::cuda::device_synchronize();
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count() / iters;
}

template <typename CpuFn, typename GpuFn>
static void row(const char* tag, CpuFn cpu, GpuFn gpu, int iters) {
    double c = ms(cpu, Device::CPU, iters);
    double g = ms(gpu, Device::CUDA, iters);
    printf("%-22s %10.3f %10.3f %8.1fx\n", tag, c, g, c / g);
}

int main() {
    printf("%-22s %10s %10s %9s\n", "op (ms/call)", "cpu", "gpu", "speedup");

    for (Index n : {128u, 256u, 512u}) {
        auto a = T::randn({n, n}), b = T::randn({n, n});
        auto ag = a.to_cuda(), bg = b.to_cuda();
        int it = n <= 128 ? 50 : n <= 256 ? 20 : 5;
        char tag[64];
        snprintf(tag, sizeof tag, "matmul %zu", (size_t)n);
        row(tag, [&] { a.matmul(b); }, [&] { ag.matmul(bg); }, it);
        snprintf(tag, sizeof tag, "transpose %zu", (size_t)n);
        row(tag, [&] { a.transpose(0, 1); }, [&] { ag.transpose(0, 1); }, 50);
        snprintf(tag, sizeof tag, "add %zu", (size_t)n);
        row(tag, [&] { a.add(b); }, [&] { ag.add(bg); }, 50);
    }

    {  // broadcast (1024,1024) + (1024,)
        auto a = T::randn({1024, 1024}), v = T::randn({1024});
        auto ag = a.to_cuda(), vg = v.to_cuda();
        row("bcast 1024+row", [&] { a.add(v); }, [&] { ag.add(vg); }, 20);
    }
    {  // conv2d 1x3x64x64 with 8x3x3x3
        auto x = T::randn({1, 3, 64, 64}), w = T::randn({8, 3, 3, 3});
        auto xg = x.to_cuda(), wg = w.to_cuda();
        row("conv2d 64x64", [&] { x.conv2d(w, 1, 1); }, [&] { xg.conv2d(wg, 1, 1); }, 20);
    }
    return 0;
}
