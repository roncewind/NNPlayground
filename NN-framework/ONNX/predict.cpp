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
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>

// ============================================================================
// Minimal flag parser
std::unordered_map<std::string, std::string> parse_flags(int argc, char *argv[])
{
    std::unordered_map<std::string, std::string> flags;
    for (int i = 1; i < argc - 1; ++i)
    {
        if (argv[i][0] == '-' && argv[i + 1][0] != '-')
        {
            flags[argv[i]] = argv[i + 1];
            ++i;
        }
    }
    return flags;
}

// ============================================================================
std::vector<std::pair<int, float>> top_k(const std::vector<float> &probs, int k)
{
    std::vector<std::pair<int, float>> indexed_probs;
    for (size_t i = 0; i < probs.size(); ++i)
    {
        indexed_probs.emplace_back(i, probs[i]);
    }

    // Partial sort for top-k
    std::partial_sort(
        indexed_probs.begin(),
        indexed_probs.begin() + k,
        indexed_probs.end(),
        [](const auto &a, const auto &b)
        { return a.second > b.second; });

    indexed_probs.resize(k);
    return indexed_probs;
}

// ============================================================================
std::vector<float> softmax(const float *logits, size_t size)
{
    std::vector<float> probs(size);
    float max_logit = *std::max_element(logits, logits + size);

    // Compute exponentials (for numerical stability)
    float sum_exp = 0.0f;
    for (size_t i = 0; i < size; ++i)
    {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }

    // Normalize
    for (size_t i = 0; i < size; ++i)
    {
        probs[i] /= sum_exp;
    }

    return probs;
}

// ============================================================================
std::vector<std::string> load_class_labels(const std::string &filename)
{
    std::vector<std::string> labels;
    std::ifstream infile(filename);
    std::string line;
    while (std::getline(infile, line))
    {
        if (!line.empty())
            labels.push_back(line);
    }
    return labels;
}

// ============================================================================
int main()
{
    auto flags = parse_flags(argc, argv);

    std::string model_path = flags.count("--model") ? flags["--model"] : "model.onnx";
    std::string labels_path = flags.count("--labels") ? flags["--labels"] : "class_labels.txt";
    int input_size = flags.count("--input-size") ? std::stoi(flags["--input-size"]) : 33;
    int output_size = flags.count("--output-size") ? std::stoi(flags["--output-size"]) : 11;
    int k = flags.count("--top-k") ? std::stoi(flags["--top-k"]) : 3;

    std::cout << "📦 Model: " << model_path << "\n";
    std::cout << "📄 Labels: " << labels_path << "\n";
    std::cout << "📐 Input size: " << input_size << ", Output size: " << output_size << "\n";

    // Initialize environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_inference");

    // Create session
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path.c_str(), session_options);

    // Get allocator and input/output names
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input name
    // const char* input_name = session.GetInputName(0, allocator);
    // const char* output_name = session.GetOutputName(0, allocator);
    std::vector<std::string> input_names_str = session.GetInputNames();
    std::vector<std::string> output_names_str = session.GetOutputNames();

    // Convert to const char* arrays
    std::vector<const char *> input_names;
    for (const auto &s : input_names_str)
        input_names.push_back(s.c_str());

    std::vector<const char *> output_names;
    for (const auto &s : output_names_str)
        output_names.push_back(s.c_str());

    // batch=1, 33 features
    const std::array<int64_t, 2> input_shape{1, input_size};

    // Create input tensor with 33 dummy values
    std::vector<float> input_values(input_size, 0.5f); // TODO: Replace with real data

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_values.data(), input_values.size(), input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1);

    // Get output
    float *output = output_tensors.front().GetTensorMutableData<float>();

    // Load labels
    std::vector<std::string> class_labels = load_class_labels(labels_path);
    if (class_labels.size() != static_cast<size_t>(output_size))
    {
        std::cerr << "❌ Mismatch: class label count != output size\n";
        return 1;
    }

    std::cout << "\n✅ Model output (class scores):" << std::endl;
    for (int i = 0; i < output_size; ++i)
    {
        std::cout << class_labels[i] << " (class " << i << "): " << output[i] << std::endl;
    }

    // Apply softmax to get probabilities
    std::vector<float> probs = softmax(output, output_size);

    // Print top-3 predictions
    auto top_preds = top_k(probs, k);

    std::cout << "\n🔝 Top " << k << " predictions:" << std::endl;
    for (const auto &[class_idx, prob] : top_preds)
    {
        std::cout << class_labels[class_idx]
                  << " (class " << class_idx << "): "
                  << prob * 100 << "% confidence" << std::endl;
    }

    return 0;
}
