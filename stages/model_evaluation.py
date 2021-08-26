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


def evaluate_model(config_path):

    config = read_params(config_path=config_path)

    # Reading Model Path

    with open("reports/metrics/scores.json", "r") as f:
        data = json.load(f)
    model_path = data['model_scores'][-1]['model_path']

    test_path = config['load_data']['test_path']
    paths = ['NORMAL/*', 'PNEUMONIA/*']
    true_outputs = []
    predicted_outputs = []

    model = load_model(model_path)

    # Prediction

    for i in paths:
        path = os.path.join(test_path, i)
        images = glob(path)
        for j in range(len(images)):

            if 'NORMAL' in images[j]:
                # Not affected
                true_outputs.append(1)
            else:
                # Affected with pneumonia
                true_outputs.append(0)

            img = image.load_img(images[j], target_size=(224, 224))
            x = image.img_to_array(img)
            y = np.expand_dims(x, axis=0)

            img_data = preprocess_input(y)
            classes = model.predict(img_data)
            result = int(classes[0][0])

            predicted_outputs.append(result)

    # Classification report, confusion metrics and f1 score

    metric_file = config["reports"]["metric_path"]

    with open(metric_file, "r") as f:
        data = json.load(f)

    model_metric = {
        "confusion_matrix": confusion_matrix(true_outputs, predicted_outputs).tolist(),
        "precision": precision_score(true_outputs, predicted_outputs),
        "recall": recall_score(true_outputs, predicted_outputs),
        "f1_score": f1_score(true_outputs, predicted_outputs)
    }

    data['model_metric'].append(model_metric)
    with open(metric_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    evaluate_model(config_path=parsed_args.config)
