# predict.py
# python predict.py --data data/test_samples.csv
import argparse

import pandas as pd
import torch
from config import MODEL_PATH
from dataset import feature_extractor
from model import NeuralNet
from torch.nn.functional import softmax
from utils import load_class_labels


def predict(csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    data = pd.read_csv(csv_path)
    labels = load_class_labels()

    with torch.no_grad():
        for i, row in data.iterrows():
            vec = feature_extractor(row)
            x = torch.tensor([vec], dtype=torch.float32).to(device)

            out = model(x)  # raw logits

            probs = softmax(out, dim=1).cpu()  # convert to probabilities
            prob_vals = probs[0].numpy()
            pred_idx = int(probs.argmax(dim=1))
            # pred_idx = out.argmax(1).item()

            print(f"\nRow {i} prediction:")
            for j, p in enumerate(prob_vals):
                print(f"  {labels[j]}: {p:.4f}")
            print(f">>> Predicted: {labels[pred_idx]} ({prob_vals[pred_idx]:.4f})")
            true_label = row["label"] if "label" in row else None
            msg = f"Row {i}: Predicted = {pred_idx} ({labels[pred_idx]})"
            if true_label is not None:
                msg += f" | Actual = {true_label} ({labels[int(true_label)]})"
            print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file to predict")
    args = parser.parse_args()
    predict(args.data)
