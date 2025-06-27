# NNPlayground
Neural Network mucking about.

## NN-framework

This is an attempt to create a framework for creating NNs for various purposes.

### Getting started:

Create a virtual environment and activate it:

```
python3 -m venv venv
sounce venv/bin/activate
```

### How to train:

Note: see the sample_data.csv entry in [Output files](Output files:) for a python one-liner to create a random sample file.

- Train with 64 hidden units (default)

```
python train_model.py --data data/sample_data.csv
```

- Train with 128 hidden units

```
python train_model.py --data data/sample_data.csv --hidden-size 128
```

- Train with custom hyperparameters

```
python train_model.py \
  --data data/sample_data.csv \
  --hidden-size 128 \
  --num-layers 3 \
  --batch-size 64 \
  --learning-rate 0.0005
  ```

### How to predict:

Note: see the test_samples.csv entry in [Output files](Output files:) for a python one-liner to create a test sample file from the training set.


- Predict a few samples from the training set:

```
python predict.py --data data/test_samples.csv
```

- Predict a single random input sample:

```
python test_predict_row.py
```

### Input files:

- `config.py`
    Externalizes many of the hyperparameters for a Neural Network and various
    files that get generated.

- `data/class_labels.txt` is a list of the labels for the output classifications.
    The NN classifies into numeric indices which are translated to the labels
    specified in this file. EG index 0 = first label in the list, index 1 = second label, etc.

### Output files:

- `data/best_model.pth` and `data/best_model_config.json` are both input and output files
    They are created when the model is trained and the model is used when predictions are made.
    the JSON file documents the hyperparameters used to create the model during training.

- `data/confusion_matrix*` various confusion matrix outputs. These give an idea of how
    well the model performs.

- `data/sample_data.csv` and `test_sample.csv` dummy random sample data created to test
    the framework.
    - the sample data is created with `python -c "from dataset import generate_random_csv; generate_random_csv('data/sample_data.csv', 1500)"`
    - test sample is created with `python -c "import pandas as pd; df = pd.read_csv('sample_data.csv'); df.tail(5).to_csv('data/test_samples.csv', index=False)"`

- `data/classification_report.txt` records a summary of how well the model performs it includes: the precision, recall, F1-score, and support.

- `data/per_class_metrics.txt` records the accuracy of each classification.

- `data/top_misclassifications.txt` records a deeper look at the top 10 misclassifications.

## tinkering

These are just playground files, tinkering with NNs in various ways.


