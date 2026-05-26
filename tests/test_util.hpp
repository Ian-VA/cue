#pragma once

#include "../tensor/tensor.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct TestCase {
    std::string           name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
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

inline void expect_tensor_close(const Tensor<float>& got,
                                const std::vector<float>& expected,
                                double tol,
                                const char* label) {
    if (got.size() != expected.size()) {
        std::ostringstream os;
        os << label << ": size mismatch, got " << got.size()
           << ", expected " << expected.size();
        throw std::runtime_error(os.str());
    }
    Tensor<float> host = got.on_cuda() ? got.to_cpu() : got;
    for (Index i = 0; i < expected.size(); ++i) {
        double diff = std::abs((double)host.data()[i] - (double)expected[i]);
        if (diff > tol) {
            std::ostringstream os;
            os << label << ": mismatch at index " << i << ", got "
               << host.data()[i] << ", expected " << expected[i]
               << ", |diff| = " << diff;
            throw std::runtime_error(os.str());
        }
    }
}

inline void expect_tensors_equal(const Tensor<float>& a, const Tensor<float>& b,
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
            os << label << ": mismatch at index " << i << ", a=" << ha.data()[i]
               << ", b=" << hb.data()[i] << ", |diff|=" << diff;
            throw std::runtime_error(os.str());
        }
    }
}

inline int run_all_tests() {
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
            failures.push_back(t.name + ": " + e.what());
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
