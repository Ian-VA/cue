#include "dataset.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace cue::data {

namespace {

void trim(std::string& s) {
    auto issp = [](unsigned char c) { return std::isspace(c); };
    while (!s.empty() && issp(s.front())) s.erase(s.begin());
    while (!s.empty() && issp(s.back()))  s.pop_back();
}

std::vector<std::string> split(const std::string& line, char delim) {
    std::vector<std::string> out;
    std::string              cell;
    for (char c : line) {
        if (c == delim) {
            out.push_back(cell);
            cell.clear();
        } else {
            cell.push_back(c);
        }
    }
    out.push_back(cell);
    for (auto& s : out) trim(s);
    return out;
}

bool parse_float(const std::string& s, float& out) {
    if (s.empty()) return false;
    try {
        size_t idx = 0;
        out = std::stof(s, &idx);
        while (idx < s.size() && std::isspace((unsigned char)s[idx])) ++idx;
        return idx == s.size();
    } catch (...) {
        return false;
    }
}

Index shape_product(const std::vector<Index>& shape) {
    Index p = 1;
    for (Index d : shape) p *= d;
    return p;
}

}

CsvTable read_csv(const std::string& path, char delim, bool has_header) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open csv: " + path);
    }

    CsvTable    table;
    std::string line;
    bool        first = true;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto cells = split(line, delim);
        if (first && has_header) {
            table.header = std::move(cells);
            first        = false;
            continue;
        }
        table.rows.push_back(std::move(cells));
        first = false;
    }
    return table;
}

// Dataset

Dataset::Dataset(std::vector<float>       features,
                 std::vector<int>         labels,
                 std::vector<Index>       feature_shape,
                 std::vector<std::string> class_names)
        : _features(std::move(features)),
          _labels(std::move(labels)),
          _feature_shape(std::move(feature_shape)),
          _class_names(std::move(class_names)) {
    _per_sample = shape_product(_feature_shape);
    if (_per_sample == 0) {
        throw std::invalid_argument("feature_shape must be non-empty");
    }
    if (_features.size() != _per_sample * _labels.size()) {
        throw std::invalid_argument("features and labels size mismatch");
    }
}

Index Dataset::num_classes() const {
    if (!_class_names.empty()) return _class_names.size();
    int max_label = -1;
    for (int l : _labels) if (l > max_label) max_label = l;
    return (Index)(max_label + 1);
}

Tensor<float> Dataset::gather_features(const std::vector<Index>& indices,
                                       Device device) const {
    std::vector<Index> out_shape = {indices.size()};
    out_shape.insert(out_shape.end(), _feature_shape.begin(), _feature_shape.end());

    Tensor<float> cpu(out_shape, Device::CPU);
    for (Index i = 0; i < indices.size(); ++i) {
        Index src = indices[i] * _per_sample;
        if (src + _per_sample > _features.size()) {
            throw std::out_of_range("gather index past end of features");
        }
        std::copy_n(_features.data() + src, _per_sample,
                    cpu.data() + i * _per_sample);
    }
    return device == Device::CPU ? cpu : cpu.to_cuda();
}

Tensor<int> Dataset::gather_labels(const std::vector<Index>& indices) const {
    Tensor<int> out({indices.size()}, Device::CPU);
    for (Index i = 0; i < indices.size(); ++i) {
        if (indices[i] >= _labels.size()) {
            throw std::out_of_range("gather index past end of labels");
        }
        out.data()[i] = _labels[indices[i]];
    }
    return out;
}

std::pair<Tensor<float>, Tensor<int>>
Dataset::batch(const std::vector<Index>& indices, Device device) const {
    return {gather_features(indices, device), gather_labels(indices)};
}

std::pair<std::vector<float>, std::vector<float>> Dataset::normalise() {
    Index N = _labels.size();
    std::vector<float> mean(_per_sample, 0.0f);
    std::vector<float> var(_per_sample,  0.0f);

    for (Index n = 0; n < N; ++n) {
        for (Index j = 0; j < _per_sample; ++j) {
            mean[j] += _features[n * _per_sample + j];
        }
    }
    for (auto& m : mean) m /= (float)N;

    for (Index n = 0; n < N; ++n) {
        for (Index j = 0; j < _per_sample; ++j) {
            float d = _features[n * _per_sample + j] - mean[j];
            var[j] += d * d;
        }
    }
    std::vector<float> stddev(_per_sample);
    for (Index j = 0; j < _per_sample; ++j) {
        stddev[j] = std::sqrt(var[j] / (float)N);
        if (stddev[j] < 1e-8f) stddev[j] = 1.0f;
    }

    apply_normalisation(mean, stddev);
    return {mean, stddev};
}

void Dataset::apply_normalisation(const std::vector<float>& mean,
                                  const std::vector<float>& stddev) {
    if (mean.size() != _per_sample || stddev.size() != _per_sample) {
        throw std::invalid_argument("normalisation stats size mismatch");
    }
    Index N = _labels.size();
    for (Index n = 0; n < N; ++n) {
        for (Index j = 0; j < _per_sample; ++j) {
            _features[n * _per_sample + j] =
                (_features[n * _per_sample + j] - mean[j]) / stddev[j];
        }
    }
}

// CSV → Dataset

Dataset from_csv(const CsvTable& table, const std::vector<Index>& feature_shape) {
    if (table.rows.empty()) {
        throw std::invalid_argument("empty csv table");
    }
    Index per_sample = shape_product(feature_shape);
    Index cols       = table.rows[0].size();
    if (cols < 2) {
        throw std::invalid_argument("csv needs at least one feature and a label");
    }
    if (cols - 1 != per_sample) {
        throw std::invalid_argument("feature_shape does not match csv column count");
    }

    std::vector<float>                       features;
    std::vector<int>                         labels;
    std::vector<std::string>                 class_names;
    std::unordered_map<std::string, int>     name_to_id;

    features.reserve(table.rows.size() * per_sample);
    labels.reserve(table.rows.size());

    for (const auto& row : table.rows) {
        if (row.size() != cols) {
            throw std::invalid_argument("inconsistent csv row width");
        }
        for (Index j = 0; j < per_sample; ++j) {
            float v;
            if (!parse_float(row[j], v)) {
                throw std::invalid_argument("non-numeric feature value: " + row[j]);
            }
            features.push_back(v);
        }
        const auto& label_str = row.back();
        auto it = name_to_id.find(label_str);
        int id;
        if (it == name_to_id.end()) {
            id = (int)class_names.size();
            class_names.push_back(label_str);
            name_to_id.emplace(label_str, id);
        } else {
            id = it->second;
        }
        labels.push_back(id);
    }

    return Dataset(std::move(features), std::move(labels),
                   feature_shape, std::move(class_names));
}

// DataLoader

DataLoader::DataLoader(const Dataset& dataset, Index batch_size, bool shuffle,
                       uint64_t seed, Device device)
        : _dataset(&dataset),
          _batch(batch_size),
          _shuffle(shuffle),
          _device(device),
          _rng(seed == 0 ? std::random_device{}() : seed) {
    if (_batch == 0) throw std::invalid_argument("batch size must be > 0");
    _order.resize(dataset.size());
    std::iota(_order.begin(), _order.end(), (Index)0);
    reset();
}

void DataLoader::reset() {
    if (_shuffle) {
        std::shuffle(_order.begin(), _order.end(), _rng);
    }
    _cursor = 0;
}

Index DataLoader::num_batches() const {
    return (_dataset->size() + _batch - 1) / _batch;
}

bool DataLoader::next(Tensor<float>& features, Tensor<int>& labels) {
    if (_cursor >= _order.size()) return false;
    Index end = std::min(_cursor + _batch, _order.size());
    std::vector<Index> idx(_order.begin() + _cursor, _order.begin() + end);
    _cursor = end;

    features = _dataset->gather_features(idx, _device);
    labels   = _dataset->gather_labels(idx);
    return true;
}

}
