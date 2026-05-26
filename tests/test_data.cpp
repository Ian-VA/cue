#include "test_util.hpp"

#include "../data/dataset.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

using cue::data::Dataset;
using cue::data::DataLoader;
using cue::data::from_csv;
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
    // Each normalised value should have magnitude 1 (single-pair samples).
    for (Index i = 0; i < x.size(); ++i) {
        REQUIRE_CLOSE(std::abs(x.data()[i]), 1.0f, 1e-5);
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
