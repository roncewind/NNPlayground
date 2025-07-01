# utils.py
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch.onnx
from config import (
    BATCH_SIZE,
    CLASSIFICATION_REPORT_TXT,
    CONF_MATRIX_IMG,
    HIDDEN_SIZE,
    LABELS_PATH,
    METRICS_CSV,
    MODEL_PATH,
    NUM_CLASSES,
    NUM_INPUTS,
    NUM_LAYERS,
    TOP_MISCLASSIFICATIONS,
)
from model import NeuralNet
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


# -----------------------------------------------------------------------------
def load_class_labels():
    with open(LABELS_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]


# -----------------------------------------------------------------------------
def save_per_class_metrics(true_labels, pred_labels):
    labels = load_class_labels()
    from collections import defaultdict

    correct = defaultdict(int)
    total = defaultdict(int)

    for t, p in zip(true_labels, pred_labels):
        total[t] += 1
        if t == p:
            correct[t] += 1

    metrics = []
    for idx in range(NUM_CLASSES):
        acc = 100 * correct[idx] / total[idx] if total[idx] > 0 else 0.0
        metrics.append(
            {
                "Class Index": idx,
                "Label": labels[idx],
                "Correct": correct[idx],
                "Total": total[idx],
                "Accuracy (%)": round(acc, 2),
            }
        )

    df = pd.DataFrame(metrics)
    df.to_csv(METRICS_CSV, index=False)
    print(f"✅ Saved per-class metrics to {METRICS_CSV}")


# -----------------------------------------------------------------------------
def plot_confusion_matrix(true_labels, pred_labels, normalized=True):
    labels = load_class_labels()
    normalize_option = "true" if normalized else None

    # Compute confusion matrix
    cm = confusion_matrix(
        true_labels,
        pred_labels,
        labels=list(range(len(labels))),
        normalize=normalize_option,
    )

    # Save as CSV
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    suffix = "_normalized" if normalized else ""
    base, _ = os.path.splitext(CONF_MATRIX_IMG)
    csv_path = base + suffix + ".csv"
    df_cm.to_csv(csv_path)
    print(f"📄 Saved confusion matrix CSV to {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format=".2f" if normalized else "d",
    )
    plt.title("Normalized Confusion Matrix" if normalized else "Confusion Matrix")
    plt.tight_layout()
    img_path = base + suffix + ".png"
    plt.savefig(img_path)
    # plt.show()
    print(f"✅ Saved image to {img_path}")


# -----------------------------------------------------------------------------
def save_top_misclassifications(
    true_labels, pred_labels, top_n=10, output_path=TOP_MISCLASSIFICATIONS
):
    labels = load_class_labels()
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(labels))))

    mis_list = []
    for i in range(len(labels)):
        true_class_total = cm[i].sum()
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                mis_list.append(
                    {
                        "True Class": labels[i],
                        "Predicted As": labels[j],
                        "Count": int(cm[i, j]),
                        "Total True": int(true_class_total),
                        "% of True": round((cm[i, j] / true_class_total) * 100, 2),
                    }
                )

    # Sort by percentage, then by count
    top_mis = sorted(mis_list, key=lambda x: (-x["% of True"], -x["Count"]))[:top_n]
    df = pd.DataFrame(top_mis)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n📊 Top {top_n} Misclassifications (also saved to {output_path}):")
    print(df.to_string(index=False))


# -----------------------------------------------------------------------------
def print_classification_report(true_labels, pred_labels):
    from utils import load_class_labels

    print("\n📊 Classification Report:")
    labels = load_class_labels()
    report = classification_report(
        true_labels, pred_labels, target_names=labels, digits=3
    )
    print(report)

    with open(CLASSIFICATION_REPORT_TXT, "w") as f:
        f.write(str(report))
    print(f"✅ Saved classification report to {CLASSIFICATION_REPORT_TXT}")


# -----------------------------------------------------------------------------
# python -c "from utils import export_to_onnx; export_to_onnx('data/best_model')"
def export_to_onnx(
    onnx_path=MODEL_PATH,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_size=BATCH_SIZE,
    model_path=MODEL_PATH,
    num_inputs=NUM_INPUTS,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(hidden_size=hidden_size, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Dummy input for ONNX export
    dummy_input = torch.randn(batch_size, num_inputs, device=device)

    # Export to ONNX
    onnx_path = os.path.splitext(onnx_path)[0] + ".onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print(f"✅ Exported model to ONNX format at {onnx_path}")
