#include "test_util.hpp"

#include "../data/dataset.hpp"
#include "../nn/nn.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using cue::data::Dataset;
using cue::data::DataLoader;
using cue::data::from_csv;
using cue::data::read_csv;
using cue::nn::Conv2d;
using cue::nn::CrossEntropyLoss;
using cue::nn::Flatten;
using cue::nn::Linear;
using cue::nn::MaxPool2d;
using cue::nn::ReLU;
using cue::nn::SGD;
using cue::nn::Sequential;

#ifndef CUE_DATA_DIR
#define CUE_DATA_DIR "."
#endif

namespace {

constexpr const char* IRIS_URL =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";

std::string ensure_iris() {
    std::filesystem::path path = std::filesystem::path(CUE_DATA_DIR) / "iris.csv";
    if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 0) {
        return path.string();
    }
    std::filesystem::create_directories(path.parent_path());
    std::string cmd = "curl -fsSL --max-time 15 \"" + std::string(IRIS_URL) +
                      "\" -o \"" + path.string() + "\"";
    int rc = std::system(cmd.c_str());
    if (rc != 0 || !std::filesystem::exists(path) ||
        std::filesystem::file_size(path) == 0) {
        std::filesystem::remove(path);
        return "";
    }
    return path.string();
}

float accuracy(const Tensor<float>& logits, const Tensor<int>& labels) {
    Tensor<float> host = logits.on_cuda() ? logits.to_cpu() : logits;
    Index N = host.shape()[0];
    Index C = host.shape()[1];
    Index correct = 0;
    for (Index n = 0; n < N; ++n) {
        Index best = 0;
        float bv   = host.data()[n*C];
        for (Index c = 1; c < C; ++c) {
            float v = host.data()[n*C + c];
            if (v > bv) { bv = v; best = c; }
        }
        if ((int)best == labels.data()[n]) ++correct;
    }
    return (float)correct / (float)N;
}

Sequential build_cnn(Device device) {
    Sequential m;
    m.emplace<Conv2d>(1, 8, 2, 1, 1, device);
    m.emplace<ReLU>();
    m.emplace<MaxPool2d>(2, 2, 1, 0);
    m.emplace<Flatten>();
    m.emplace<Linear>(8 * 2 * 2, 16, device);
    m.emplace<ReLU>();
    m.emplace<Linear>(16, 3, device);
    return m;
}

float train_one_epoch(Sequential& model, DataLoader& dl,
                      CrossEntropyLoss& loss, SGD& opt) {
    Tensor<float> x; Tensor<int> y;
    dl.reset();
    float total = 0.0f;
    Index batches = 0;
    while (dl.next(x, y)) {
        auto logits = model.forward(x);
        float l = loss.forward(logits, y);
        opt.zero_grad();
        auto g = loss.backward();
        model.backward(g);
        opt.step();
        total += l;
        ++batches;
    }
    return total / (float)batches;
}

float evaluate(Sequential& model, const Dataset& ds, Device device) {
    std::vector<Index> idx(ds.size());
    for (Index i = 0; i < idx.size(); ++i) idx[i] = i;
    auto x = ds.gather_features(idx, device);
    auto y = ds.gather_labels(idx);
    auto logits = model.forward(x);
    return accuracy(logits, y);
}

Dataset load_iris_dataset(const std::string& path) {
    auto table = read_csv(path, ',', /*has_header=*/false);
    return from_csv(table, {1, 2, 2});
}

void run_training(Device device, const std::string& tag) {
    auto path = ensure_iris();
    if (path.empty()) {
        std::printf("        [skip %s] could not fetch %s\n",
                    tag.c_str(), IRIS_URL);
        return;
    }

    auto ds = load_iris_dataset(path);
    REQUIRE(ds.size() >= 90);
    REQUIRE_EQ(ds.num_classes(), 3u);
    REQUIRE_EQ(ds.num_features(), 4u);
    ds.normalise();

    DataLoader dl(ds, /*batch=*/15, /*shuffle=*/true, /*seed=*/42, device);

    auto             model = build_cnn(device);
    CrossEntropyLoss loss;
    SGD              opt(model.parameters(), 0.05f);

    float last_loss = 0.0f;
    for (int epoch = 0; epoch < 60; ++epoch) {
        last_loss = train_one_epoch(model, dl, loss, opt);
    }
    float final_acc = evaluate(model, ds, device);
    std::printf("        [%s] final_acc=%.2f last_loss=%.4f\n",
                tag.c_str(), final_acc, last_loss);
    REQUIRE(final_acc > 0.90f);
}

}


TEST(train_iris, cpu) {
    run_training(Device::CPU, "cpu");
}

TEST(train_iris, gpu) {
    run_training(Device::CUDA, "gpu");
}


int main() { return run_all_tests(); }
