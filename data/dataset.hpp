#pragma once

#include "../tensor/tensor.hpp"

#include <random>
#include <string>
#include <utility>
#include <vector>

namespace cue::data {

struct CsvTable {
    std::vector<std::string>              header;
    std::vector<std::vector<std::string>> rows;
};

CsvTable read_csv(const std::string& path,
                  char delim = ',', bool has_header = true);

// In-memory dataset of (features, label) pairs. Features are a flat row-major buffer while labels are integer class ids
class Dataset {
    public:
        Dataset(std::vector<float>        features,
                std::vector<int>          labels,
                std::vector<Index>        feature_shape,
                std::vector<std::string>  class_names = {});

        Index size()         const { return _labels.size(); }
        Index num_features() const { return _per_sample; }
        Index num_classes()  const;

        const std::vector<Index>&       feature_shape() const { return _feature_shape; }
        const std::vector<std::string>& class_names()   const { return _class_names; }

        // Build a contiguous (n, *feature_shape) tensor from selected indices.
        Tensor<float> gather_features(const std::vector<Index>& indices,
                                      Device device = Device::CPU) const;
        Tensor<int>   gather_labels(const std::vector<Index>& indices) const;

        std::pair<Tensor<float>, Tensor<int>>
            batch(const std::vector<Index>& indices,
                  Device device = Device::CPU) const;

        // Z-score normalise each feature column using statistics computed from this dataset. Returns (mean, std)
        std::pair<std::vector<float>, std::vector<float>> normalise();
        void apply_normalisation(const std::vector<float>& mean,
                                 const std::vector<float>& stddev);

    private:
        std::vector<float>       _features;
        std::vector<int>         _labels;
        std::vector<Index>       _feature_shape;
        std::vector<std::string> _class_names;
        Index                    _per_sample {0};
};

// Build a Dataset from a CSV table. The last column is treated as the class
// label while remaining columns are parsed as floats. Class strings are mapped to
// integer ids in first-seen order.
Dataset from_csv(const CsvTable& table,
                 const std::vector<Index>& feature_shape);

class DataLoader {
    public:
        DataLoader(const Dataset& dataset,
                   Index batch_size,
                   bool  shuffle = true,
                   uint64_t seed = 0,
                   Device device = Device::CPU);

        void reset();
        bool next(Tensor<float>& features, Tensor<int>& labels);

        Index batch_size() const { return _batch; }
        Index num_batches() const;

    private:
        const Dataset*     _dataset;
        Index              _batch;
        bool               _shuffle;
        Device             _device;
        std::vector<Index> _order;
        Index              _cursor {0};
        std::mt19937       _rng;
};

}
