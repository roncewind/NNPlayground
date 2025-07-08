// This code is an attempt to embed an ONNX model into C++ such that a model file is  not needed.
// It initializes the ONNX Runtime environment, creates a session with the model, prepares an input tensor, and runs inference to get the output.
// The output is expected to be a vector of logits or softmax scores for 11 classes, which are printed to the console.
// The input tensor is initialized with dummy values (0.5) for demonstration purposes, but in a real application, you would replace these with actual data.
// The code assumes that the ONNX model is named `best_model.onnx` and is located in the `data` directory relative to the executable's path.
// Make sure to have ONNX Runtime installed and properly linked in your C++ project to compile and run this code successfully.

// Compile with:
// g++ test_embedded_model.cpp embedded_model.cpp -o test_embedded_model -I/path/to/onnxruntime/include -L/path/to/onnxruntime/lib -lonnxruntime -std=c++17
// g++ test_embedded_model.cpp embedded_model.cpp -o test_embedded_model -I/usr/local/include/onnxruntime/ -L/usr/local/lib -lonnxruntime -std=c++17
// Make sure to replace /path/to/onnxruntime with the actual path to your ONNX Runtime installation.
// Make sure ONNX Runtime's shared libraries are in your LD_LIBRARY_PATH or system path.
// You can run the compiled binary with:
// ./predict
// Ensure that the best_model.onnx file is in the data directory or provide the correct path to it in the code.
// The input tensor is initialized with dummy values (0.5) for demonstration purposes. Replace these with actual data as needed.
// The output will show the raw logits or softmax scores for each of the 11 classes.

#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>

#include <vector>
#include "embedded_model.h"

// ----------------------------------------------------------------------------
int main() {
    // 2. Create an ONNX Runtime Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_RUNTIME_EXAMPLE");

    // 3. Create an ONNX Runtime Session from Memory
    Ort::SessionOptions session_options;
    // Add any desired session options
    // session_options.SetIntraOpNumThreads(1);

    try {
        Ort::Session session(env, embedded_model_data, embedded_model_data_size, session_options);

        // Model successfully loaded from the C++ variable (byte array)
        // Now you can proceed with inference using this session
        // ... (e.g., prepare input tensors, run session, process output)

        std::cout << "ONNX model loaded successfully from C++ variable." << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}