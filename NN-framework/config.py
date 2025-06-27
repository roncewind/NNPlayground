# config.py
NUM_INPUTS = 33
NUM_CLASSES = 11
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 5
LEARNING_RATE = 0.001
NUM_LAYERS = 2
HIDDEN_SIZE = 64

MODEL_PATH = "data/best_model.pth"
MODEL_CONFIG_PATH = "data/best_model_config.json"
LABELS_PATH = "data/class_labels.txt"
METRICS_CSV = "data/per_class_metrics.csv"
CONF_MATRIX_IMG = "data/confusion_matrix.png"
CLASSIFICATION_REPORT_TXT = "data/classification_report.txt"
TOP_MISCLASSIFICATIONS = "data/top_misclassifications.csv"
