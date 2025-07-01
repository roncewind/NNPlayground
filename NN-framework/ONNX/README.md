# Using ONNX

## Linux

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

- Update the library path with `ldconfig`

```
sudo ldconf
```

- Make sure your `LD_LIBRARY_PATH` includes `/usr/local/lib`

```
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```