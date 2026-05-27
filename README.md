# cue

Cue is a small ML library written in CUDA C++.

### Running cue

Build the library and tests:

```sh
cmake -S . -B build
cmake --build build -j
```

Run all tests:

```sh
ctest --test-dir build --output-on-failure
```

Run an individual test executable:

```sh
./build/test_tensor
./build/test_nn
./build/test_data
./build/test_train_iris
```
