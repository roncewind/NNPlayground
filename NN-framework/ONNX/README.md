# Using ONNX

## Setup ONNX on Linux

- Download the ONNX runtime .tgz file from: https://github.com/microsoft/onnxruntime/releases
- Un-tar: `tar -zxvf ~/Downloads/onnxruntime-linux-x64-<version>.tgz`
    - This will create an `onnxruntime-linux-x64-<version>` directory
- inside the `onnxruntime-linux-x64-<version>` directory are
    - an `include` directory
    - a `lib` directory
    - a number of other files
- Copy the contents of the `include` directory to `/usr/local/include/onnxruntime`.
    You may need to create the `onnxruntime` subdirectory if it doesn't exist.

```
sudo mkdir /usr/local/include/onnxruntime
sudo cp -r include/* /usr/local/include/onnxruntime
```

- Copy the shared object files from the `lib` directory to `/usr/local/lib`.

```
sudo cp lib/libonnxruntime.so /usr/local/lib
```

Note: one could copy the versioned .so and then create symlinks if necessary.

```
sudo cp lib/libonnxruntime.so.<major>.<minor>.<patch> /usr/local/lib
sudo ln -sf /usr/local/lib/libonnxruntime.so.<major>.<minor>.<patch> /usr/local/lib/libonnxruntime.so.<major>
sudo ln -sf /usr/local/lib/libonnxruntime.so.<major> /usr/local/lib/libonnxruntime.so
```

for example:

```
sudo cp lib/libonnxruntime.so.1.22.0 /usr/local/lib
sudo ln -sf /usr/local/lib/libonnxruntime.so.1.22.0 /usr/local/lib/libonnxruntime.so.1
sudo ln -sf /usr/local/lib/libonnxruntime.so.1 /usr/local/lib/libonnxruntime.so
```

- Update the linker bindings and cache with `ldconfig`.

```
sudo ldconf
```

- Make sure your `LD_LIBRARY_PATH` includes `/usr/local/lib`

```
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

## Files:

- `predict.cpp`: will make prediction based on the model file you give it against a test CSV file. Usage:

```
./predict --model ../data/best_model.onnx --labels ../data/class_labels.txt --input-size 33 --output-size 11 --top-k 3 --test-csv ../data/test_features.csv
```

- `save_model.cpp`: generated an `embedded_model.h` file which is a version of the model that can be embedded in a C++ program
- `test_embedded_model.cpp`: tests that the model created by `save_model` can be loaded by the ONNX runtime.
- `predict_embedded.cpp`: just like `predict` but will make prediction based on the model saved in the `embedded_model.h` file. Usage:

```
./predict_embedded --labels ../data/class_labels.txt --input-size 33 --output-size 11 --top-k 3 --test-csv ../data/test_features.csv
```

Note: if you diff the output of `predict` and `predict_embedded` the only difference should be one line about which model it's using.

## TODO:

🔧 Nice-to-Haves:
- ✅  📉 Add a softmax function to get probability scores
- ✅  🔝 Print or log top-1 / top-3 predicted class labels
-     📉 Measure inference time per sample
- ✅  🔁 Loop over a batch of CSV inputs
-     🚀 Use ONNX Runtime with GPU (CUDA EP) for acceleration???
-     📦 Work on CLI options for the embedded examples: output file name, input filename, others???
