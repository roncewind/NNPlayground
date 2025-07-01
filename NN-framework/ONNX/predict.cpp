// This code is a simple ONNX Runtime C++ example to load a pre-trained ONNX model and perform inference.
// It initializes the ONNX Runtime environment, creates a session with the model, prepares an input tensor, and runs inference to get the output.
// The output is expected to be a vector of logits or softmax scores for 11 classes, which are printed to the console.
// The input tensor is initialized with dummy values (0.5) for demonstration purposes, but in a real application, you would replace these with actual data.
// The code assumes that the ONNX model is named `best_model.onnx` and is located in the `data` directory relative to the executable's path.
// Make sure to have ONNX Runtime installed and properly linked in your C++ project to compile and run this code successfully.

// Compile with:
// g++ predict.cpp -o predict -I/path/to/onnxruntime/include -L/path/to/onnxruntime/lib -lonnxruntime -std=c++17
// Make sure to replace /path/to/onnxruntime with the actual path to your ONNX Runtime installation.
// Make sure ONNX Runtime's shared libraries are in your LD_LIBRARY_PATH or system path.
// You can run the compiled binary with:
// ./predict
// Ensure that the best_model.onnx file is in the data directory or provide the correct path to it in the code.
// The input tensor is initialized with dummy values (0.5) for demonstration purposes. Replace these with actual data as needed.
// The output will show the raw logits or softmax scores for each of the 11 classes.

#include <iostream>
#include <vector>
#include <onnxruntime/onnxruntime_cxx_api.h>

int main()
{
    // Initialize environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_inference");

    // Create session
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "../data/best_model.onnx", session_options);

    // Get allocator and input/output names
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input name
    // const char* input_name = session.GetInputName(0, allocator);
    // const char* output_name = session.GetOutputName(0, allocator);
    std::vector<std::string> input_names_str = session.GetInputNames();
    std::vector<std::string> output_names_str = session.GetOutputNames();

    // Convert to const char* arrays
    std::vector<const char*> input_names;
    for (const auto& s : input_names_str) input_names.push_back(s.c_str());

    std::vector<const char*> output_names;
    for (const auto& s : output_names_str) output_names.push_back(s.c_str());

    // batch=1, 33 features
    const std::array<int64_t, 2> input_shape{1, 33};

    // Create input tensor with 33 dummy values
    std::vector<float> input_values(33, 0.5f); // TODO: Replace with real data

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_values.data(), input_values.size(), input_shape.data(), input_shape.size()
    );

    // Run inference
    // auto output_tensors = session.Run(
    //     Ort::RunOptions{nullptr},
    //     &input_names,
    //     &input_tensor,
    //     1,
    //     &output_name,
    //     1
    // );
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1
    );

    // Get output
    float *output = output_tensors.front().GetTensorMutableData<float>();

    std::cout << "Model output (class scores):" << std::endl;
    for (int i = 0; i < 11; ++i)
    {
        std::cout << "Class " << i << ": " << output[i] << std::endl;
    }

    return 0;
}



// int main() {
//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_inference");

//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);

//     Ort::Session session(env, "model.onnx", session_options);

//     // Get allocator and input/output names
//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input name
//     std::vector<const char*> input_names = session.GetInputNames(allocator);
//     std::vector<const char*> output_names = session.GetOutputNames(allocator);

//     const std::array<int64_t, 2> input_shape{1, 33}; // batch size 1, 33 features
//     std::vector<float> input_values(33, 0.5f); // Dummy input values

//     Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_values.data(), input_values.size(), input_shape.data(), input_shape.size()
//     );

//     // Run inference
//     auto output_tensors = session.Run(
//         Ort::RunOptions{nullptr},
//         input_names.data(),
//         &input_tensor,
//         1,
//         output_names.data(),
//         1
//     );

//     // Extract output
//     float* output = output_tensors.front().GetTensorMutableData<float>();

//     std::cout << "Model output (class scores):" << std::endl;
//     for (int i = 0; i < 11; ++i) {
//         std::cout << "Class " << i << ": " << output[i] << std::endl;
//     }

//     return 0;
// }