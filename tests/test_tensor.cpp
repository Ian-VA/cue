#include "../tensor/tensor.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>

struct TestCase {
    std::string           name;
    std::function<void()> fn;
};

static std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct TestRegistrar {
    TestRegistrar(std::string name, std::function<void()> fn) {
        registry().push_back({std::move(name), std::move(fn)});
    }
};

#define TEST(group, name)                                                            \
    static void test_##group##_##name();                                             \
    static TestRegistrar test_reg_##group##_##name{#group "::" #name,                \
                                                   test_##group##_##name};          \
    static void test_##group##_##name()

#define FAIL(msg)                                                                    \
    do {                                                                             \
        std::ostringstream _os;                                                      \
        _os << msg << " at " << __FILE__ << ":" << __LINE__;                         \
        throw std::runtime_error(_os.str());                                         \
    } while (0)

#define REQUIRE(cond)                                                                \
    do {                                                                             \
        if (!(cond)) FAIL("REQUIRE failed: " #cond);                                 \
    } while (0)

#define REQUIRE_EQ(a, b)                                                             \
    do {                                                                             \
        auto _va = (a); auto _vb = (b);                                              \
        if (!(_va == _vb)) {                                                         \
            std::ostringstream _os;                                                  \
            _os << "REQUIRE_EQ failed: " #a " (" << _va << ") == " #b " (" << _vb    \
                << ")";                                                              \
            FAIL(_os.str());                                                         \
        }                                                                            \
    } while (0)

#define REQUIRE_CLOSE(a, b, tol)                                                     \
    do {                                                                             \
        double _da = (double)(a); double _db = (double)(b);                          \
        double _diff = std::abs(_da - _db);                                          \
        if (_diff > (tol)) {                                                         \
            std::ostringstream _os;                                                  \
            _os << "REQUIRE_CLOSE failed: " #a " (" << _da << ") ~= " #b " ("        \
                << _db << "), |diff| = " << _diff << " > " << (tol);                 \
            FAIL(_os.str());                                                         \
        }                                                                            \
    } while (0)

#define REQUIRE_THROWS(expr)                                                         \
    do {                                                                             \
        bool _threw = false;                                                         \
        try { (void)(expr); } catch (const std::exception&) { _threw = true; }       \
        if (!_threw) FAIL("REQUIRE_THROWS failed: " #expr " did not throw");         \
    } while (0)

// Compare two CPU tensors element-wise.
static void expect_tensor_close(const Tensor<float>& got,
                                const std::vector<float>& expected,
                                double tol,
                                const char* label) {
    if (got.size() != expected.size()) {
        std::ostringstream os;
        os << label << ": size mismatch — got " << got.size()
           << ", expected " << expected.size();
        throw std::runtime_error(os.str());
    }
    Tensor<float> host = got.on_cuda() ? got.to_cpu() : got;
    for (Index i = 0; i < expected.size(); ++i) {
        double diff = std::abs((double)host.data()[i] - (double)expected[i]);
        if (diff > tol) {
            std::ostringstream os;
            os << label << ": mismatch at index " << i << " — got "
               << host.data()[i] << ", expected " << expected[i]
               << ", |diff| = " << diff;
            throw std::runtime_error(os.str());
        }
    }
}

static void expect_tensors_equal(const Tensor<float>& a, const Tensor<float>& b,
                                 double tol, const char* label) {
    if (a.shape() != b.shape()) {
        std::ostringstream os;
        os << label << ": shape mismatch";
        throw std::runtime_error(os.str());
    }
    Tensor<float> ha = a.on_cuda() ? a.to_cpu() : a;
    Tensor<float> hb = b.on_cuda() ? b.to_cpu() : b;
    for (Index i = 0; i < ha.size(); ++i) {
        double diff = std::abs((double)ha.data()[i] - (double)hb.data()[i]);
        if (diff > tol) {
            std::ostringstream os;
            os << label << ": mismatch at index " << i << " — a=" << ha.data()[i]
               << ", b=" << hb.data()[i] << ", |diff|=" << diff;
            throw std::runtime_error(os.str());
        }
    }
}


TEST(construction, shape_ctor) {
    std::vector<Index> shape = {2, 3, 4};
    Tensor<float> t(shape);
    REQUIRE_EQ(t.rank(), 3u);
    REQUIRE_EQ(t.size(), 24u);
    REQUIRE_EQ(t.shape()[0], 2u);
    REQUIRE_EQ(t.shape()[1], 3u);
    REQUIRE_EQ(t.shape()[2], 4u);
}

TEST(construction, default_is_zero) {
    auto t = Tensor<float>::zeros({5});
    for (Index i = 0; i < 5; ++i) REQUIRE_EQ(t.data()[i], 0.0f);
}

TEST(construction, initializer_list_1d) {
    auto t = Tensor<float>::from_values({1, 2, 3, 4});
    REQUIRE_EQ(t.rank(), 1u);
    REQUIRE_EQ(t.size(), 4u);
    REQUIRE_EQ(t[{0}], 1.0f);
    REQUIRE_EQ(t[{3}], 4.0f);
}

TEST(construction, initializer_list_2d) {
    auto t = Tensor<float>::from_values({{1, 2, 3}, {4, 5, 6}});
    REQUIRE_EQ(t.rank(), 2u);
    REQUIRE_EQ(t.shape()[0], 2u);
    REQUIRE_EQ(t.shape()[1], 3u);
    REQUIRE_EQ((t[{0, 0}]), 1.0f);
    REQUIRE_EQ((t[{0, 2}]), 3.0f);
    REQUIRE_EQ((t[{1, 0}]), 4.0f);
    REQUIRE_EQ((t[{1, 2}]), 6.0f);
}

TEST(construction, initializer_list_2d_uneven_throws) {
    REQUIRE_THROWS((Tensor<float>{{1, 2, 3}, {4, 5}}));
}

TEST(construction, oob_throws) {
    auto t = Tensor<float>::from_values({{1, 2}, {3, 4}});
    REQUIRE_THROWS((t[{2, 0}]));
    REQUIRE_THROWS((t[{0}]));
}

TEST(construction, at_mutates) {
    auto t = Tensor<float>::zeros({2, 2});
    t.at({0, 1}) = 5.0f;
    REQUIRE_EQ((t[{0, 1}]), 5.0f);
}


TEST(factories, zeros) {
    auto t = Tensor<float>::zeros({3, 4});
    REQUIRE_EQ(t.size(), 12u);
    for (Index i = 0; i < 12; ++i) REQUIRE_EQ(t.data()[i], 0.0f);
}

TEST(factories, ones) {
    auto t = Tensor<float>::ones({2, 3});
    for (Index i = 0; i < 6; ++i) REQUIRE_EQ(t.data()[i], 1.0f);
}

TEST(factories, full) {
    auto t = Tensor<float>::full({4}, 7.5f);
    for (Index i = 0; i < 4; ++i) REQUIRE_EQ(t.data()[i], 7.5f);
}

TEST(factories, randn_shape_only) {
    auto t = Tensor<float>::randn({100});
    REQUIRE_EQ(t.size(), 100u);
    // Sanity: mean of 100 N(0,1) samples should be within a few sigma of 0.
    float s = 0.0f;
    for (Index i = 0; i < 100; ++i) s += t.data()[i];
    REQUIRE(std::abs(s / 100.0f) < 0.5f);
}


TEST(shape, reshape) {
    auto t = Tensor<float>::ones({2, 6});
    t.reshape({3, 4});
    REQUIRE_EQ(t.rank(), 2u);
    REQUIRE_EQ(t.shape()[0], 3u);
    REQUIRE_EQ(t.shape()[1], 4u);
    REQUIRE_EQ(t.size(), 12u);
}

TEST(shape, reshape_bad_size_throws) {
    auto t = Tensor<float>::ones({2, 6});
    REQUIRE_THROWS(t.reshape({3, 5}));
}

TEST(shape, view_shares_storage) {
    auto t = Tensor<float>::from_values({{1, 2}, {3, 4}});
    auto v = t.view({4});
    REQUIRE_EQ(v.rank(), 1u);
    // Mutating through one view should be visible through the other.
    v.at({0}) = 99.0f;
    REQUIRE_EQ((t[{0, 0}]), 99.0f);
}

TEST(shape, flatten) {
    auto t = Tensor<float>::ones({2, 3, 4, 5});
    auto flat = t.flatten();        // collapses dims [1, end)
    REQUIRE_EQ(flat.rank(), 2u);
    REQUIRE_EQ(flat.shape()[0], 2u);
    REQUIRE_EQ(flat.shape()[1], 60u);
}

TEST(shape, transpose_2d) {
    auto t = Tensor<float>::from_values({{1, 2, 3}, {4, 5, 6}});
    auto tt = t.transpose(0, 1);
    REQUIRE_EQ(tt.shape()[0], 3u);
    REQUIRE_EQ(tt.shape()[1], 2u);
    expect_tensor_close(tt, {1, 4, 2, 5, 3, 6}, 1e-6, "transpose_2d");
}


TEST(elementwise, add) {
    auto a = Tensor<float>::from_values({{1, 2}, {3, 4}});
    auto b = Tensor<float>::from_values({{5, 6}, {7, 8}});
    auto c = a + b;
    expect_tensor_close(c, {6, 8, 10, 12}, 1e-6, "add");
}

TEST(elementwise, sub) {
    auto a = Tensor<float>::from_values({{5, 6}, {7, 8}});
    auto b = Tensor<float>::from_values({{1, 2}, {3, 4}});
    auto c = a - b;
    expect_tensor_close(c, {4, 4, 4, 4}, 1e-6, "sub");
}

TEST(elementwise, mul) {
    auto a = Tensor<float>::from_values({{1, 2}, {3, 4}});
    auto b = Tensor<float>::from_values({{5, 6}, {7, 8}});
    auto c = a * b;
    expect_tensor_close(c, {5, 12, 21, 32}, 1e-6, "mul");
}

TEST(elementwise, div) {
    auto a = Tensor<float>::from_values({{10, 20}, {30, 40}});
    auto b = Tensor<float>::from_values({{2, 4}, {5, 8}});
    auto c = a / b;
    expect_tensor_close(c, {5, 5, 6, 5}, 1e-6, "div");
}

TEST(elementwise, shape_mismatch_throws) {
    auto a = Tensor<float>::from_values({1, 2, 3});
    auto b = Tensor<float>::from_values({1, 2});
    REQUIRE_THROWS(a + b);
}

TEST(elementwise, neg) {
    auto a = Tensor<float>::from_values({1, -2, 3});
    auto n = -a;
    expect_tensor_close(n, {-1, 2, -3}, 1e-6, "neg");
}

TEST(elementwise, inplace_ops) {
    auto a = Tensor<float>::from_values({1, 2, 3});
    auto b = Tensor<float>::from_values({10, 20, 30});
    a += b;
    expect_tensor_close(a, {11, 22, 33}, 1e-6, "+=");
    a -= b;
    expect_tensor_close(a, {1, 2, 3}, 1e-6, "-=");
}


TEST(scalar, add) {
    auto a = Tensor<float>::from_values({1, 2, 3});
    auto b = a + 10.0f;
    expect_tensor_close(b, {11, 12, 13}, 1e-6, "scalar add");
}

TEST(scalar, sub) {
    auto a = Tensor<float>::from_values({10, 20, 30});
    auto b = a - 5.0f;
    expect_tensor_close(b, {5, 15, 25}, 1e-6, "scalar sub");
}

TEST(scalar, mul) {
    auto a = Tensor<float>::from_values({1, 2, 3});
    auto b = a * 3.0f;
    expect_tensor_close(b, {3, 6, 9}, 1e-6, "scalar mul");
}


TEST(matmul, basic_2x2) {
    auto a = Tensor<float>::from_values({{1, 2}, {3, 4}});
    auto b = Tensor<float>::from_values({{5, 6}, {7, 8}});
    auto c = a.matmul(b);
    REQUIRE_EQ(c.shape()[0], 2u);
    REQUIRE_EQ(c.shape()[1], 2u);
    expect_tensor_close(c, {19, 22, 43, 50}, 1e-6, "matmul 2x2");
}

TEST(matmul, rectangular) {
    auto a = Tensor<float>::from_values({{1, 2, 3}, {4, 5, 6}});      // 2x3
    auto b = Tensor<float>::from_values({{7, 8}, {9, 10}, {11, 12}}); // 3x2
    auto c = a.matmul(b);                          // 2x2
    REQUIRE_EQ(c.shape()[0], 2u);
    REQUIRE_EQ(c.shape()[1], 2u);
    expect_tensor_close(c, {58, 64, 139, 154}, 1e-6, "matmul 2x3 * 3x2");
}

TEST(matmul, inner_mismatch_throws) {
    auto a = Tensor<float>::from_values({{1, 2}});
    auto b = Tensor<float>::from_values({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    REQUIRE_THROWS(a.matmul(b));
}


TEST(reduce, sum) {
    auto t = Tensor<float>::from_values({{1, 2, 3}, {4, 5, 6}});
    REQUIRE_CLOSE(t.sum(), 21.0f, 1e-6);
}

TEST(reduce, mean) {
    auto t = Tensor<float>::from_values({{1, 2, 3}, {4, 5, 6}});
    REQUIRE_CLOSE(t.mean(), 3.5f, 1e-6);
}


TEST(activation, relu) {
    auto t = Tensor<float>::from_values({-2, -1, 0, 1, 2});
    auto r = t.relu();
    expect_tensor_close(r, {0, 0, 0, 1, 2}, 1e-6, "relu");
}

TEST(activation, relu_backward) {
    auto grad = Tensor<float>::from_values({1, 1, 1, 1, 1});
    auto input = Tensor<float>::from_values({-2, -1, 0, 1, 2});
    auto out = grad.relu_backward(input);
    // mask is input > 0: positives pass, non-positives zeroed.
    expect_tensor_close(out, {0, 0, 0, 1, 1}, 1e-6, "relu_backward");
}

TEST(activation, sigmoid) {
    auto t = Tensor<float>::from_values({-10, 0, 10});
    auto s = t.sigmoid();
    REQUIRE_CLOSE(s.data()[0], 0.0f, 1e-4);
    REQUIRE_CLOSE(s.data()[1], 0.5f, 1e-6);
    REQUIRE_CLOSE(s.data()[2], 1.0f, 1e-4);
}

TEST(activation, tanh) {
    auto t = Tensor<float>::from_values({-10, 0, 10});
    auto r = t.tanh();
    REQUIRE_CLOSE(r.data()[0], -1.0f, 1e-4);
    REQUIRE_CLOSE(r.data()[1],  0.0f, 1e-6);
    REQUIRE_CLOSE(r.data()[2],  1.0f, 1e-4);
}

TEST(activation, exp_log) {
    auto t = Tensor<float>::from_values({0, 1, 2});
    auto e = t.exp();
    REQUIRE_CLOSE(e.data()[0], 1.0f,           1e-6);
    REQUIRE_CLOSE(e.data()[1], std::exp(1.0f), 1e-5);
    REQUIRE_CLOSE(e.data()[2], std::exp(2.0f), 1e-5);
    auto back = e.log();
    expect_tensor_close(back, {0, 1, 2}, 1e-5, "log(exp(x))");
}

TEST(activation, softmax_last_dim) {
    auto t = Tensor<float>::from_values({{1, 2, 3}, {1, 2, 3}});
    auto s = t.softmax();

    // Row sums should be 1
    for (Index r = 0; r < 2; ++r) {
        float sum = s.data()[r*3] + s.data()[r*3+1] + s.data()[r*3+2];
        REQUIRE_CLOSE(sum, 1.0f, 1e-5);
    }

    REQUIRE_CLOSE(s.data()[0], 0.09003f, 1e-4);
    REQUIRE_CLOSE(s.data()[1], 0.24473f, 1e-4);
    REQUIRE_CLOSE(s.data()[2], 0.66524f, 1e-4);
}


TEST(conv, conv2d_3x3_with_2x2) {
    // Input (1,1,3,3) = [[1,2,3],[4,5,6],[7,8,9]]
    auto in = Tensor<float>::zeros({1, 1, 3, 3});
    for (Index i = 0; i < 9; ++i) in.data()[i] = (float)(i + 1);
    // Kernel (1,1,2,2) = [[1,0],[0,-1]]
    auto ker = Tensor<float>::zeros({1, 1, 2, 2});
    ker.data()[0] =  1.0f;
    ker.data()[1] =  0.0f;
    ker.data()[2] =  0.0f;
    ker.data()[3] = -1.0f;

    auto out = in.conv2d(ker, /*stride=*/1, /*padding=*/0);
    REQUIRE_EQ(out.rank(), 4u);
    REQUIRE_EQ(out.shape()[0], 1u);
    REQUIRE_EQ(out.shape()[1], 1u);
    REQUIRE_EQ(out.shape()[2], 2u);
    REQUIRE_EQ(out.shape()[3], 2u);
    expect_tensor_close(out, {-4, -4, -4, -4}, 1e-5, "conv2d");
}

TEST(conv, conv2d_with_padding) {
    auto in = Tensor<float>::from_values({{1.0f, 2.0f}, {3.0f, 4.0f}});
    in = in.view({1, 1, 2, 2});
    auto ker = Tensor<float>::zeros({1, 1, 1, 1});
    ker.data()[0] = 1.0f;

    auto out = in.conv2d(ker, 1, 1);
    REQUIRE_EQ(out.shape()[2], 4u);
    REQUIRE_EQ(out.shape()[3], 4u);
    REQUIRE_EQ(out.data()[0], 0.0f);
    REQUIRE_EQ(out.data()[5], 1.0f);
    REQUIRE_EQ(out.data()[6], 2.0f);
}


TEST(pool, max_pool2d) {
    auto in = Tensor<float>::zeros({1, 1, 2, 2});
    in.data()[0] = 1.0f; in.data()[1] = 2.0f;
    in.data()[2] = 3.0f; in.data()[3] = 4.0f;
    auto out = in.max_pool2d(2, 2);
    REQUIRE_EQ(out.size(), 1u);
    REQUIRE_EQ(out.data()[0], 4.0f);
}

TEST(pool, avg_pool2d) {
    auto in = Tensor<float>::zeros({1, 1, 2, 2});
    in.data()[0] = 1.0f; in.data()[1] = 2.0f;
    in.data()[2] = 3.0f; in.data()[3] = 4.0f;
    auto out = in.avg_pool2d(2, 2);
    REQUIRE_CLOSE(out.data()[0], 2.5f, 1e-6);
}

TEST(pool, max_pool2d_stride) {
    auto in = Tensor<float>::zeros({1, 1, 4, 4});
    for (Index i = 0; i < 16; ++i) in.data()[i] = (float)i;
    auto out = in.max_pool2d(2, 2);
    REQUIRE_EQ(out.shape()[2], 2u);
    REQUIRE_EQ(out.shape()[3], 2u);
    expect_tensor_close(out, {5, 7, 13, 15}, 1e-6, "max_pool2d stride");
}

TEST(bias, nchw) {
    // 1x2x2x2 input, 2-channel bias
    auto in = Tensor<float>::zeros({1, 2, 2, 2});
    for (Index i = 0; i < 8; ++i) in.data()[i] = (float)i;
    auto bias = Tensor<float>::from_values({10.0f, 100.0f});
    auto out = in.add_bias(bias);
    expect_tensor_close(out, {10, 11, 12, 13, 104, 105, 106, 107},
                        1e-6, "add_bias_nchw");
}

TEST(bias, fc_2d) {
    auto in = Tensor<float>::from_values({{1, 2, 3}, {10, 20, 30}});
    auto bias = Tensor<float>::from_values({100, 200, 300});
    auto out = in.add_bias(bias);
    expect_tensor_close(out, {101, 202, 303, 110, 220, 330},
                        1e-6, "add_bias_2d");
}

TEST(cuda, roundtrip) {
    auto cpu = Tensor<float>::from_values({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto gpu     = cpu.to_cuda();
    REQUIRE(gpu.on_cuda());
    auto back    = gpu.to_cpu();
    REQUIRE(!back.on_cuda());
    expect_tensors_equal(back, cpu, 0.0, "roundtrip");
}

TEST(cuda, mixed_device_throws) {
    auto cpu = Tensor<float>::from_values({1, 2, 3});
    auto gpu = cpu.to_cuda();
    REQUIRE_THROWS(cpu + gpu);
}

TEST(cuda, indexing_on_cuda_throws) {
    auto gpu = Tensor<float>::ones({4}, Device::CUDA);
    REQUIRE_THROWS(gpu[{0}]);
}

static void check_elementwise(const std::vector<Index>& shape) {
    auto a_cpu = Tensor<float>::randn(shape);
    auto b_cpu = Tensor<float>::randn(shape);
    auto a_gpu = a_cpu.to_cuda();
    auto b_gpu = b_cpu.to_cuda();

    expect_tensors_equal(a_cpu + b_cpu, (a_gpu + b_gpu).to_cpu(), 1e-5, "add");
    expect_tensors_equal(a_cpu - b_cpu, (a_gpu - b_gpu).to_cpu(), 1e-5, "sub");
    expect_tensors_equal(a_cpu * b_cpu, (a_gpu * b_gpu).to_cpu(), 1e-5, "mul");
}

TEST(cuda, elementwise_small) { check_elementwise({16}); }
TEST(cuda, elementwise_large) { check_elementwise({1000}); }
TEST(cuda, elementwise_4d)    { check_elementwise({4, 8, 16, 16}); }

TEST(cuda, scalar_ops) {
    auto a_cpu = Tensor<float>::randn({256});
    auto a_gpu = a_cpu.to_cuda();
    expect_tensors_equal(a_cpu + 3.0f, (a_gpu + 3.0f).to_cpu(), 1e-5, "scalar +");
    expect_tensors_equal(a_cpu * 2.5f, (a_gpu * 2.5f).to_cpu(), 1e-5, "scalar *");
}

TEST(cuda, activations) {
    auto a_cpu = Tensor<float>::randn({1024});
    auto a_gpu = a_cpu.to_cuda();

    expect_tensors_equal(a_cpu.relu(),    a_gpu.relu().to_cpu(),    1e-5, "relu");
    expect_tensors_equal(a_cpu.sigmoid(), a_gpu.sigmoid().to_cpu(), 1e-4, "sigmoid");
    expect_tensors_equal(a_cpu.tanh(),    a_gpu.tanh().to_cpu(),    1e-5, "tanh");
}

TEST(cuda, relu_backward) {
    auto input_cpu = Tensor<float>::randn({512});
    auto grad_cpu  = Tensor<float>::randn({512});
    auto input_gpu = input_cpu.to_cuda();
    auto grad_gpu  = grad_cpu.to_cuda();

    expect_tensors_equal(grad_cpu.relu_backward(input_cpu),
                         grad_gpu.relu_backward(input_gpu).to_cpu(),
                         1e-5, "relu_backward");
}

TEST(cuda, softmax) {
    auto a_cpu = Tensor<float>::randn({8, 10});
    auto a_gpu = a_cpu.to_cuda();
    expect_tensors_equal(a_cpu.softmax(), a_gpu.softmax().to_cpu(), 1e-5, "softmax");
}

TEST(cuda, matmul_small) {
    auto a_cpu = Tensor<float>::randn({4, 5});
    auto b_cpu = Tensor<float>::randn({5, 3});
    auto a_gpu = a_cpu.to_cuda();
    auto b_gpu = b_cpu.to_cuda();
    expect_tensors_equal(a_cpu.matmul(b_cpu), a_gpu.matmul(b_gpu).to_cpu(),
                         1e-4, "matmul small");
}

TEST(cuda, matmul_large) {
    auto a_cpu = Tensor<float>::randn({64, 128});
    auto b_cpu = Tensor<float>::randn({128, 32});
    auto a_gpu = a_cpu.to_cuda();
    auto b_gpu = b_cpu.to_cuda();
    expect_tensors_equal(a_cpu.matmul(b_cpu), a_gpu.matmul(b_gpu).to_cpu(),
                         5e-3, "matmul large");
}

TEST(cuda, transpose_2d) {
    auto a_cpu = Tensor<float>::randn({8, 5});
    auto a_gpu = a_cpu.to_cuda();
    expect_tensors_equal(a_cpu.transpose(0, 1), a_gpu.transpose(0, 1).to_cpu(),
                         1e-6, "transpose 2d");
}

TEST(cuda, sum) {
    auto a_cpu = Tensor<float>::randn({4096});
    auto a_gpu = a_cpu.to_cuda();
    REQUIRE_CLOSE(a_cpu.sum(), a_gpu.sum(), 1e-2);
}

TEST(cuda, conv2d) {
    auto in_cpu  = Tensor<float>::randn({2, 3, 8, 8});
    auto ker_cpu = Tensor<float>::randn({4, 3, 3, 3});
    auto in_gpu  = in_cpu.to_cuda();
    auto ker_gpu = ker_cpu.to_cuda();
    expect_tensors_equal(in_cpu.conv2d(ker_cpu, 1, 1),
                         in_gpu.conv2d(ker_gpu, 1, 1).to_cpu(),
                         1e-3, "conv2d");
}

TEST(cuda, max_pool2d) {
    auto in_cpu = Tensor<float>::randn({2, 4, 8, 8});
    auto in_gpu = in_cpu.to_cuda();
    expect_tensors_equal(in_cpu.max_pool2d(2, 2),
                         in_gpu.max_pool2d(2, 2).to_cpu(),
                         1e-6, "max_pool2d");
}

TEST(cuda, avg_pool2d) {
    auto in_cpu = Tensor<float>::randn({2, 4, 8, 8});
    auto in_gpu = in_cpu.to_cuda();
    expect_tensors_equal(in_cpu.avg_pool2d(2, 2),
                         in_gpu.avg_pool2d(2, 2).to_cpu(),
                         1e-6, "avg_pool2d");
}

TEST(cuda, bias_nchw) {
    auto in_cpu   = Tensor<float>::randn({2, 4, 5, 5});
    auto bias_cpu = Tensor<float>::randn({4});
    auto in_gpu   = in_cpu.to_cuda();
    auto bias_gpu = bias_cpu.to_cuda();
    expect_tensors_equal(in_cpu.add_bias(bias_cpu),
                         in_gpu.add_bias(bias_gpu).to_cpu(),
                         1e-5, "bias nchw");
}


TEST(integration, cnn_forward_gpu) {
    Index N = 2, Cin = 3, H = 16, W = 16;
    auto input = Tensor<float>::randn({N, Cin, H, W}).to_cuda();

    auto conv1_w = Tensor<float>::randn({8, 3, 3, 3}, 0.0f, 0.1f).to_cuda();
    auto conv1_b = Tensor<float>::randn({8},          0.0f, 0.01f).to_cuda();
    auto conv2_w = Tensor<float>::randn({16, 8, 3, 3}, 0.0f, 0.1f).to_cuda();
    auto conv2_b = Tensor<float>::randn({16},          0.0f, 0.01f).to_cuda();
    auto fc_w    = Tensor<float>::randn({16 * 4 * 4, 10}, 0.0f, 0.1f).to_cuda();
    auto fc_b    = Tensor<float>::randn({10},             0.0f, 0.01f).to_cuda();

    auto x = input.conv2d(conv1_w, 1, 1).add_bias(conv1_b).relu().max_pool2d(2, 2);
    x = x.conv2d(conv2_w, 1, 1).add_bias(conv2_b).relu().max_pool2d(2, 2);
    x = x.flatten().matmul(fc_w).add_bias(fc_b).softmax();

    REQUIRE_EQ(x.rank(), 2u);
    REQUIRE_EQ(x.shape()[0], N);
    REQUIRE_EQ(x.shape()[1], 10u);

    auto host = x.to_cpu();
    for (Index i = 0; i < host.size(); ++i) {
        REQUIRE(std::isfinite(host.data()[i]));
    }
    // Softmax rows sum to 1
    for (Index n = 0; n < N; ++n) {
        float s = 0.0f;
        for (Index c = 0; c < 10; ++c) s += host.data()[n*10 + c];
        REQUIRE_CLOSE(s, 1.0f, 1e-4);
    }
}

int main() {
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failures;
    for (const auto& t : registry()) {
        try {
            t.fn();
            std::printf("  PASS  %s\n", t.name.c_str());
            ++passed;
        } catch (const std::exception& e) {
            std::printf("  FAIL  %s\n        %s\n", t.name.c_str(), e.what());
            failures.push_back(t.name + " — " + e.what());
            ++failed;
        }
        std::fflush(stdout);
    }
    std::printf("\n%d passed, %d failed\n", passed, failed);
    if (failed > 0) {
        std::printf("\nFailing tests:\n");
        for (const auto& f : failures) std::printf("  - %s\n", f.c_str());
    }
    return failed > 0 ? 1 : 0;
}
