# Train with 64 hidden units (default)
# python train_model.py --data your_training_data.csv

# Train with 128 hidden units
# python train_model.py --data my_data.csv --hidden-size 128

# Train with custom hyperparameters
# python train_model.py \
#   --data my_data.csv \
#   --hidden-size 128 \
#   --num-layers 3 \
#   --batch-size 64 \
#   --learning-rate 0.0005

import argparse
import csv
import json
import os
from datetime import datetime

import torch
from config import (
    BATCH_SIZE,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MODEL_CONFIG_PATH,
    MODEL_PATH,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_INPUTS,
    NUM_LAYERS,
    PATIENCE,
)
from dataset import CSVDataset, feature_extractor
from model import NeuralNet
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from utils import (
    plot_confusion_matrix,
    print_classification_report,
    save_per_class_metrics,
    save_top_misclassifications,
)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
def log_run_results(config, best_val_acc, stopped_epoch):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"run_{timestamp}.csv")

    fields = {
        "timestamp": timestamp,
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "best_val_accuracy": round(best_val_acc, 4),
        "stopped_epoch": stopped_epoch,
    }

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields.keys())
        writer.writeheader()
        writer.writerow(fields)

    print(f"📄 Logged run to {log_path}")


# -----------------------------------------------------------------------------
# Train the model
# def train(csv_path, hidden_size=HIDDEN_SIZE):
def train(
    csv_path,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CSVDataset(csv_path, feature_extractor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # model = NeuralNet(hidden_size=hidden_size).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model = NeuralNet(hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0
    # epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        correct_train, total_train = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            pred = out.argmax(1)
            correct_train += (pred == y).sum().item()
            total_train += y.size(0)

        model.eval()
        correct_val, total_val = 0, 0
        all_preds, all_trues = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                pred = out.argmax(1)
                all_preds.extend(pred.cpu().tolist())
                all_trues.extend(y.cpu().tolist())
                correct_val += (pred == y).sum().item()
                total_val += y.size(0)

        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        print(
            f"Epoch {epoch + 1:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

        print(
            f"Best validation accuracy so far: {best_val_acc:.4f} (epoch {best_epoch})"
        )
    save_per_class_metrics(all_trues, all_preds)
    plot_confusion_matrix(all_trues, all_preds, normalized=True)
    plot_confusion_matrix(all_trues, all_preds, normalized=False)
    save_top_misclassifications(all_trues, all_preds)
    print_classification_report(all_trues, all_preds)
    log_run_results(
        config={
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        best_val_acc=best_val_acc,
        stopped_epoch=best_epoch,
    )
    # save the model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model state at epoch {epoch + 1}")
    # Save the best model configuration to a JSON file
    save_model_config(
        {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "best_val_accuracy": round(best_val_acc, 4),
            "early_stopped_epoch": best_epoch,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "input_dim": NUM_INPUTS,
            "output_classes": NUM_CLASSES,
        }
    )


# -----------------------------------------------------------------------------
# Save the best model configuration to a JSON file
def save_model_config(config_dict, path=MODEL_CONFIG_PATH):
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"📝 Saved model config to {path}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training CSV file")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=HIDDEN_SIZE,
        help="Hidden layer size (default: 64)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=NUM_LAYERS,
        help="Number of hidden layers (default: 2)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate (default: 0.001)",
    )
    args = parser.parse_args()

    train(
        csv_path=args.data,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data", required=True, help="Path to training CSV file")
#     parser.add_argument(
#         "--hidden-size", type=int, default=64, help="Hidden layer size (default: 64)"
#     )
#     args = parser.parse_args()

#     train(csv_path=args.data, hidden_size=args.hidden_size)
