import onnx

onnx_model = onnx.load("data/best_model.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX model is valid.")
