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

// in-memory dataset of feature label pairs the feature buffer is held as a
// single (n, *feature_shape) tensor and can live on the cpu or the gpu labels
// are integer class ids kept on the host
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

        // where the feature buffer currently lives
        Device device() const { return _features.device(); }

        // move the whole feature buffer to a device returns *this for chaining
        Dataset& to(Device device);
        Dataset& to_cuda() { return to(Device::CUDA); }
        Dataset& to_cpu()  { return to(Device::CPU); }

        // build a contiguous (n, *feature_shape) tensor from selected indices
        Tensor<float> gather_features(const std::vector<Index>& indices,
                                      Device device = Device::CPU) const;
        Tensor<int>   gather_labels(const std::vector<Index>& indices) const;

        std::pair<Tensor<float>, Tensor<int>>
            batch(const std::vector<Index>& indices,
                  Device device = Device::CPU) const;

        // z-score normalise each feature column returns mean and std
        std::pair<std::vector<float>, std::vector<float>> normalise();
        void apply_normalisation(const std::vector<float>& mean,
                                 const std::vector<float>& stddev);

    private:
        Tensor<float>            _features;
        std::vector<int>         _labels;
        std::vector<Index>       _feature_shape;
        std::vector<std::string> _class_names;
        Index                    _per_sample {0};
};

// build a dataset from a csv table the last column is the class label the rest
// are parsed as floats class strings get integer ids in first seen order
Dataset from_csv(const CsvTable& table,
                 const std::vector<Index>& feature_shape);

// build a dataset from a stacked (n, *feature_shape) tensor and a length n label
// tensor the leading dim is the sample axis
Dataset from_tensors(const Tensor<float>& features,
                     const Tensor<int>&   labels,
                     std::vector<std::string> class_names = {});

// build a dataset from one tensor per sample stacked along a new leading axis
Dataset from_tensors(const std::vector<Tensor<float>>& samples,
                     const std::vector<int>&           labels,
                     std::vector<std::string>          class_names = {});

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
