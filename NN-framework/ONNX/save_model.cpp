// This code is an attempt to embed an ONNX model into C++ such that a model
// file is  not needed by creating a header file with the model.
// By default, the code assumes that the ONNX model is named `best_model.onnx`
// and is located in the `data` directory relative to the executable's path.
// However, you can run the program with the `--model <path to .onnx file>` option
// to point to whatever model you'd like.
// Make sure to have ONNX Runtime installed and properly linked in your C++
// project to compile and run this code successfully.

// Compile with:
// g++ save_model.cpp -o save_model -I/path/to/onnxruntime/include -L/path/to/onnxruntime/lib -lonnxruntime -std=c++17
// Make sure to replace /path/to/onnxruntime with the actual path to your ONNX Runtime installation.
// EG. g++ save_model.cpp -o save_model -I/usr/local/include/onnxruntime/ -L/usr/local/lib -lonnxruntime -std=c++17
// Make sure ONNX Runtime's shared libraries are in your LD_LIBRARY_PATH or system path.

// You can run test the embedded model with:
// ./test_embedded_model
//  and
// ./predict_embedded
// Note: make sure to include your header(.h) file if you've created with a different
// output name than the default.

// How to run:
// With all the defaults:
// ./save_model
// With custom inputs
// ./save_model --model <path to onnx file> --output <path to custom header file>

#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>

#include <vector>
#include <fstream> // For reading model into a vector (if needed for testing)
#include <ios>     // For std::ios::binary and std::ios::ate

#include <iomanip> // Required for std::setw and std::setfill
#include <ostream> // Required for std::endl
#include <sstream> // Required for std::hex, std::uppercase, std::setfill, etc.
#include <filesystem> // Required for std::filesystem::path
#include <algorithm> // Required for std::transform and std::replace

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

// ----------------------------------------------------------------------------
std::string toHName(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    // FIXME:  compile time error with replace??
    // std::replace(h_name.begin(), h_name.end(), ".", "_");
    // Let's try a lambda expression instead.
    std::transform(s.begin(), s.end(), s.begin(),
                   [](char ch) { return (ch == '.' ? '_' : ch);  });
    return s;
}

// ----------------------------------------------------------------------------
void writeModelToFile(const std::vector<char>& data, const std::string& filename) {
    std::filesystem::path path(filename);
    std::string h_name = toHName(path.filename().string());

    std::ofstream outputFile(filename);

    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    outputFile << "#ifndef " << h_name << std::endl;
    outputFile << "#define " << h_name << std::endl;
    outputFile << "#include <cstddef>" << std::endl;
    outputFile << "extern const unsigned char embedded_model_data[] = {" << std::endl;
    // Set the output stream to hexadecimal mode and ensure two digits are always printed
    outputFile << std::hex << std::uppercase << std::setfill('0');

    std::string prefix = "0x";
    for (char c : data)
    {
        // Cast to unsigned char to avoid sign extension issues when printing hex
        outputFile << prefix << std::setw(2) << static_cast<int>(static_cast<unsigned char>(c));
        prefix = ", 0x";
    }

    outputFile << "};" << std::endl;
    outputFile << "extern const size_t embedded_model_data_size = sizeof(embedded_model_data);" << std::endl;
    outputFile << "#endif // " << h_name << std::endl;

    outputFile.close();
    std::cout << "Model successfully written to " << filename << std::endl;
}

// ----------------------------------------------------------------------------
// Function to read a file into a byte vector (for demonstration)
std::vector<char> read_file_to_vector(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
        return buffer;
    }
    return {};
}

// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    auto flags = parse_flags(argc, argv);

    std::string model_path = flags.count("--model") ? flags["--model"] : "../data/best_model.onnx";
    std::string output_path = flags.count("--output") ? flags["--output"] : "embedded_model.h";

    // 1. Prepare the model data (e.g., loaded from a file into a vector)
    std::vector<char> onnx_model_bytes = read_file_to_vector(model_path);

    writeModelToFile(onnx_model_bytes, output_path);

    // 2. Create an ONNX Runtime Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_RUNTIME_EXAMPLE");

    // 3. Create an ONNX Runtime Session from Memory
    Ort::SessionOptions session_options;
    // Add any desired session options
    // session_options.SetIntraOpNumThreads(1);

    try {
        Ort::Session session(env, onnx_model_bytes.data(), onnx_model_bytes.size(), session_options);

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