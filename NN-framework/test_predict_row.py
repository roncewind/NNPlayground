# This script tests the model with a single row of data, simulating a prediction scenario.
# python test_predict_row.py
import pandas as pd
import torch
from config import MODEL_PATH, NUM_INPUTS
from dataset import feature_extractor
from model import NeuralNet
from utils import load_class_labels

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
labels = load_class_labels()

# Prepare a new test input (manual or random)
new_row = pd.Series({f"f{i}": 0.5 for i in range(NUM_INPUTS)})  # constant feature row
new_row["label"] = 0  # dummy label, not used in prediction
print(f"New row: {new_row.to_dict()}")
features = feature_extractor(new_row)
input_tensor = torch.tensor(features, dtype=torch.float32).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    pred_idx = output.argmax(0).item()

true_label = new_row["label"] if "label" in new_row else None
msg = f"Predicted = {pred_idx} ({labels[pred_idx]})"
if true_label is not None:
    msg += f" | Actual = {true_label} ({labels[int(true_label)]})"
print(msg)
