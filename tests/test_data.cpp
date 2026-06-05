#include "test_util.hpp"

#include "../data/dataset.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

using cue::data::Dataset;
using cue::data::DataLoader;
using cue::data::from_csv;
using cue::data::from_tensors;
using cue::data::read_csv;

static std::string write_tmp_csv(const std::string& body, const std::string& tag) {
    auto path = std::filesystem::temp_directory_path() /
                ("cue_" + tag + ".csv");
    std::ofstream out(path);
    out << body;
    return path.string();
}

static Dataset make_synthetic(Index per_class = 30, Index classes = 3,
                              Index features = 4, uint32_t seed = 42) {
    std::vector<float>       feats;
    std::vector<int>         labs;
    std::vector<std::string> names;
    std::mt19937             rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.5f);

    feats.reserve(per_class * classes * features);
    labs.reserve(per_class * classes);

    for (Index c = 0; c < classes; ++c) {
        names.push_back("class_" + std::to_string(c));
        for (Index i = 0; i < per_class; ++i) {
            for (Index j = 0; j < features; ++j) {
                feats.push_back((float)c + noise(rng));
            }
            labs.push_back((int)c);
        }
    }
    return Dataset(std::move(feats), std::move(labs),
                   {features}, std::move(names));
}


TEST(csv, parses_header_and_rows) {
    auto path = write_tmp_csv(
        "a,b,c\n"
        "1.0,2.0,foo\n"
        "3.5,4.5,bar\n",
        "header");

    auto table = read_csv(path);
    REQUIRE_EQ(table.header.size(), 3u);
    REQUIRE_EQ(table.header[0], std::string("a"));
    REQUIRE_EQ(table.rows.size(), 2u);
    REQUIRE_EQ(table.rows[1][2], std::string("bar"));
}

TEST(csv, no_header) {
    auto path = write_tmp_csv("1,2,x\n3,4,y\n", "noheader");
    auto table = read_csv(path, ',', /*has_header=*/false);
    REQUIRE_EQ(table.header.size(), 0u);
    REQUIRE_EQ(table.rows.size(), 2u);
}

TEST(csv, missing_file_throws) {
    REQUIRE_THROWS(read_csv("/nope/this/file/should/not/exist.csv"));
}


TEST(dataset, from_csv_label_mapping) {
    auto path = write_tmp_csv(
        "f1,f2,label\n"
        "1.0,2.0,a\n"
        "3.0,4.0,b\n"
        "5.0,6.0,a\n",
        "labels");

    auto table = read_csv(path);
    auto ds    = from_csv(table, {2});
    REQUIRE_EQ(ds.size(), 3u);
    REQUIRE_EQ(ds.num_classes(), 2u);
    REQUIRE_EQ(ds.class_names()[0], std::string("a"));
    REQUIRE_EQ(ds.class_names()[1], std::string("b"));

    auto [feats, labs] = ds.batch({0, 1, 2});
    REQUIRE_EQ(feats.shape()[0], 3u);
    REQUIRE_EQ(feats.shape()[1], 2u);
    REQUIRE_EQ(labs.data()[0], 0);
    REQUIRE_EQ(labs.data()[1], 1);
    REQUIRE_EQ(labs.data()[2], 0);
    REQUIRE_EQ(feats.data()[0], 1.0f);
    REQUIRE_EQ(feats.data()[5], 6.0f);
}

TEST(dataset, reshape_via_feature_shape) {
    auto path = write_tmp_csv(
        "f1,f2,f3,f4,label\n"
        "1,2,3,4,a\n",
        "shape");
    auto ds = from_csv(read_csv(path), {1, 2, 2});

    auto [x, y] = ds.batch({0});
    REQUIRE_EQ(x.rank(), 4u);
    REQUIRE_EQ(x.shape()[0], 1u);
    REQUIRE_EQ(x.shape()[1], 1u);
    REQUIRE_EQ(x.shape()[2], 2u);
    REQUIRE_EQ(x.shape()[3], 2u);
    REQUIRE_EQ(x.data()[0], 1.0f);
    REQUIRE_EQ(x.data()[3], 4.0f);
}

TEST(dataset, normalise) {
    std::vector<float> feats = {0.0f, 10.0f, 4.0f, 20.0f};
    std::vector<int>   labs  = {0, 1};
    Dataset ds(std::move(feats), std::move(labs), {2});

    auto [mean, stddev] = ds.normalise();
    REQUIRE_CLOSE(mean[0], 2.0f,  1e-6);
    REQUIRE_CLOSE(mean[1], 15.0f, 1e-6);

    auto [x, _] = ds.batch({0, 1});
    // each normalised value should have magnitude 1 for single pair samples
    for (Index i = 0; i < x.size(); ++i) {
        REQUIRE_CLOSE(std::abs(x.data()[i]), 1.0f, 1e-5);
    }
}


TEST(dataset, from_stacked_tensors) {
    // (n, 2, 2) feature tensor and 1-d label tensor built purely from tensors
    auto features = Tensor<float>::from_values({
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
    });
    features.reshape({3, 2, 2});

    Tensor<int> labels(std::vector<Index>{3});
    labels.data()[0] = 0;
    labels.data()[1] = 1;
    labels.data()[2] = 0;

    auto ds = from_tensors(features, labels);
    REQUIRE_EQ(ds.size(), 3u);
    REQUIRE_EQ(ds.num_features(), 4u);
    REQUIRE_EQ(ds.feature_shape().size(), 2u);
    REQUIRE_EQ(ds.feature_shape()[0], 2u);
    REQUIRE_EQ(ds.feature_shape()[1], 2u);
    REQUIRE_EQ(ds.num_classes(), 2u);

    auto [x, y] = ds.batch({1});
    REQUIRE_EQ(x.rank(), 3u);
    REQUIRE_EQ(x.data()[0], 5.0f);
    REQUIRE_EQ(x.data()[3], 8.0f);
    REQUIRE_EQ(y.data()[0], 1);
}

TEST(dataset, from_tensor_collection) {
    // one tensor per sample stacked along a new leading axis
    std::vector<Tensor<float>> samples = {
        Tensor<float>::from_values({1, 2, 3, 4}),
        Tensor<float>::from_values({5, 6, 7, 8}),
    };
    std::vector<int> labels = {2, 5};

    auto ds = from_tensors(samples, labels, {"a", "b", "c", "d", "e", "f"});
    REQUIRE_EQ(ds.size(), 2u);
    REQUIRE_EQ(ds.num_features(), 4u);
    REQUIRE_EQ(ds.num_classes(), 6u);

    auto [x, y] = ds.batch({0, 1});
    REQUIRE_EQ(x.shape()[0], 2u);
    REQUIRE_EQ(x.shape()[1], 4u);
    REQUIRE_EQ(x.data()[4], 5.0f);
    REQUIRE_EQ(y.data()[1], 5);
}

TEST(dataset, from_tensors_size_mismatch_throws) {
    std::vector<Tensor<float>> samples = {
        Tensor<float>::from_values({1, 2}),
    };
    std::vector<int> labels = {0, 1};
    REQUIRE_THROWS(from_tensors(samples, labels));
}

TEST(dataset, to_cuda_batches_match_cpu) {
    auto cpu_ds = make_synthetic();

    auto gpu_ds = make_synthetic();
    gpu_ds.to_cuda();
    REQUIRE_EQ((int)gpu_ds.device(), (int)Device::CUDA);

    std::vector<Index> idx = {5, 0, 17, 3};

    auto [xc, yc] = cpu_ds.batch(idx);
    auto [xg, yg] = gpu_ds.batch(idx, Device::CUDA);
    REQUIRE_EQ((int)xg.device(), (int)Device::CUDA);

    auto xg_host = xg.to_cpu();
    for (Index i = 0; i < xc.size(); ++i) {
        REQUIRE_CLOSE(xc.data()[i], xg_host.data()[i], 1e-6);
    }
    for (Index i = 0; i < yc.size(); ++i) {
        REQUIRE_EQ(yc.data()[i], yg.data()[i]);
    }

    // a gpu resident dataset can still hand back cpu batches
    auto [xg_cpu, _] = gpu_ds.batch(idx, Device::CPU);
    REQUIRE_EQ((int)xg_cpu.device(), (int)Device::CPU);
    for (Index i = 0; i < xc.size(); ++i) {
        REQUIRE_CLOSE(xc.data()[i], xg_cpu.data()[i], 1e-6);
    }
}

TEST(dataset, normalise_on_cuda) {
    auto ds = make_synthetic();
    ds.to_cuda();
    ds.normalise();
    ds.to_cpu();

    // each normalised feature column should have zero mean
    Index N = ds.size();
    std::vector<Index> all(N);
    for (Index i = 0; i < N; ++i) all[i] = i;
    auto [x, _] = ds.batch(all);

    for (Index j = 0; j < ds.num_features(); ++j) {
        float s = 0.0f;
        for (Index n = 0; n < N; ++n) s += x.data()[n * ds.num_features() + j];
        REQUIRE_CLOSE(s / (float)N, 0.0f, 1e-4);
    }
}

TEST(dataset, csv_roundtrip) {
    auto ds = make_synthetic();
    auto path = std::filesystem::temp_directory_path() / "cue_roundtrip.csv";
    std::ofstream out(path);
    out << "f1,f2,f3,f4,class\n";
    for (Index i = 0; i < ds.size(); ++i) {
        auto [x, y] = ds.batch({i});
        for (Index j = 0; j < 4; ++j) {
            out << x.data()[j];
            if (j < 3) out << ",";
        }
        out << "," << ds.class_names()[(Index)y.data()[0]] << "\n";
    }
    out.close();

    auto loaded = from_csv(read_csv(path.string()), {4});
    REQUIRE_EQ(loaded.size(), ds.size());
    REQUIRE_EQ(loaded.num_classes(), ds.num_classes());
}


TEST(dataloader, iterates_all_samples) {
    auto ds = make_synthetic();
    DataLoader dl(ds, /*batch=*/8, /*shuffle=*/true, /*seed=*/7);
    REQUIRE_EQ(dl.num_batches(), (Index)((ds.size() + 7) / 8));

    Tensor<float> x; Tensor<int> y;
    Index seen = 0;
    Index batches = 0;
    while (dl.next(x, y)) {
        seen += y.size();
        ++batches;
        REQUIRE(y.size() <= 8u);
        REQUIRE_EQ(x.shape()[0], y.size());
    }
    REQUIRE_EQ(seen, ds.size());
    REQUIRE_EQ(batches, dl.num_batches());
}

TEST(dataloader, shuffle_order_changes) {
    auto ds = make_synthetic();
    DataLoader a(ds, 16, true, 1);
    DataLoader b(ds, 16, true, 2);

    Tensor<float> xa; Tensor<int> ya;
    Tensor<float> xb; Tensor<int> yb;
    a.next(xa, ya);
    b.next(xb, yb);

    bool any_diff = false;
    for (Index i = 0; i < ya.size(); ++i) {
        if (ya.data()[i] != yb.data()[i]) { any_diff = true; break; }
    }
    REQUIRE(any_diff);
}

TEST(dataloader, reset_restarts) {
    auto ds = make_synthetic();
    DataLoader dl(ds, 32, false);

    Tensor<float> x; Tensor<int> y;
    while (dl.next(x, y)) {}
    REQUIRE(!dl.next(x, y));
    dl.reset();
    REQUIRE(dl.next(x, y));
}


int main() { return run_all_tests(); }
