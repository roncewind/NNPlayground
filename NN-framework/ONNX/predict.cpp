// This code is a simple ONNX Runtime C++ example to load a pre-trained ONNX model and perform inference.
// It initializes the ONNX Runtime environment, creates a session with the model, prepares an input tensor, and runs inference to get the output.
// The output is expected to be a vector of logits or softmax scores for 11 classes, which are printed to the console.
// The input tensor is initialized with dummy values (0.5) for demonstration purposes, but in a real application, you would replace these with actual data.
// The code assumes that the ONNX model is named `best_model.onnx` and is located in the `data` directory relative to the executable's path.
// Make sure to have ONNX Runtime installed and properly linked in your C++ project to compile and run this code successfully.

// Compile with:
// g++ -std=c++17 -I/path/to/onnxruntime/include -L/path/to/onnxruntime/lib -lonnxruntime predict.cpp -o predict
// Make sure to replace /path/to/onnxruntime with the actual path to your ONNX Runtime installation.
// Make sure ONNX Runtime's shared libraries are in your LD_LIBRARY_PATH or system path.
// You can run the compiled binary with:
// ./predict
// Ensure that the best_model.onnx file is in the data directory or provide the correct path to it in the code.
// The input tensor is initialized with dummy values (0.5) for demonstration purposes. Replace these with actual data as needed.
// The output will show the raw logits or softmax scores for each of the 11 classes.

#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main()
{
    // Initialize environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_inference");

    // Create session
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "../data/best_model.onnx", session_options);

    // Get input info
    Ort::AllocatorWithDefaultOptions allocator;
    const char *input_name = session.GetInputName(0, allocator);
    const std::array<int64_t, 2> input_shape{1, 33}; // batch=1, 33 features

    // Create input tensor with 33 dummy values
    std::vector<float> input_values(33, 0.5f); // Replace with real data

    size_t input_tensor_size = 33;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

    // Run inference
    const char *output_names[] = {"output"};
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        &input_name,
        &input_tensor,
        1,
        output_names,
        1);

    // Get output
    float *output = output_tensors.front().GetTensorMutableData<float>();

    std::cout << "Model output (raw logits or softmax scores):" << std::endl;
    for (int i = 0; i < 11; ++i)
    {
        std::cout << "Class " << i << ": " << output[i] << std::endl;
    }

    return 0;
}
