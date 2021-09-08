import os
import argparse
import json
from glob import glob
import numpy as np
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from data_preprocessing import read_params
from model_training import training_model

def evaluate_model(config_path):

    config = read_params(config_path=config_path)

    # Reading Model Path and test set
    test_set = training_model(config_path=config_path)
    with open("reports/metrics/scores.json", "r") as f:
        data = json.load(f)
    model_path = data['model_scores'][-1]['model_path']

    y_true = []
    y_pred = []
    test_length = len(test_set)

    # Loading Model

    model = load_model(model_path)

    # getting labels of both train and test
    for i in range(test_length):
        x, y = test_set[i]
        y_true.extend([np.argmax(i) for i in y])
        y_pred.extend([np.argmax(i) for i in model.predict(x)])
    
    # saving Model metrics
    metric_file = config["reports"]["metric_path"]

    with open(metric_file, "r") as f:
        data = json.load(f)

    model_metric = {
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "precision": precision_score(y_true, y_pred, average=None),
        "recall": recall_score(y_true, y_pred, average=None),
        "f1_score": f1_score(y_true, y_pred, average=None)
    }

    data['model_metric'].append(model_metric)
    with open(metric_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    evaluate_model(config_path=parsed_args.config)
