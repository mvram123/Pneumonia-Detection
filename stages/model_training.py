import os
import json
import time
from datetime import datetime
import argparse
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from data_preprocessing import read_params, pre_processing


class TimeHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def training_model(config_path):

    config = read_params(config_path=config_path)

    input_shape = config['estimators']['VGG16']['params']['input_shape']
    weights = config['estimators']['VGG16']['params']['weights']
    activation = config['estimators']['VGG16']['params']['activation']

    loss = config['estimators']['VGG16']['params']['loss']
    optimizer = config['estimators']['VGG16']['params']['optimizer']
    metrics = config['estimators']['VGG16']['params']['metrics']
    epochs = config['estimators']['VGG16']['params']['epochs']

    model_summary_path = config['reports']['model_summary_path']

    model_dir = config['model_dir']

    output = config['base']['output']

    # Downloading VGG16 Model
    vgg = VGG16(input_shape=input_shape,
                weights=weights,
                include_top=False)

    # Keeping Original Weights
    for layer in vgg.layers:
        layer.trainable = False

    x = Flatten()(vgg.output)
    prediction = Dense(output, activation=activation)(x)
    model = Model(inputs=vgg.input, outputs=prediction)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[metrics]
    )

    # Current datetime
    dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    file_name = 'model_sum_' + dt_string + '.txt'

    # Saving Model Summary in a file
    with open(os.path.join(model_summary_path, file_name), 'w') as f:
        model.summary(print_fn=lambda a: f.write(a + '\n'))

    training_set, test_set = pre_processing(config_path=config_path)

    time_callback = TimeHistory()
    history = model.fit(training_set,
                        validation_data=test_set,
                        epochs=epochs,
                        callbacks=[time_callback],
                        steps_per_epoch=len(training_set),
                        validation_steps=len(test_set))

    # Time Taken
    time_taken_for_execution = sum(time_callback.times)

    # Saving Model
    model_name = 'model.h5'
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)

    # Saving Metrics and Params

    model_loss = history.history['loss'][epochs - 1]
    model_accuracy = history.history['accuracy'][epochs - 1]

    scores_file = config["reports"]["scores_path"]
    params_file = config["reports"]["params_path"]

    # Writing Scores file

    with open(scores_file, "r") as f:
        data = json.load(f)

    model_scores = {
        "time_stamp": dt_string,
        "model_name": model_name,
        "model_path": model_path,
        "time_taken": time_taken_for_execution,
        "loss": model_loss,
        "accuracy": model_accuracy
    }

    data['model_scores'].append(model_scores)

    with open(scores_file, "w") as f:

        json.dump(data, f, indent=4)

    # Writing in Params file

    with open(params_file, "r") as f:
        data_params = json.load(f)

    model_params = {
        "time_stamp": dt_string,
        "model_name": model_name,
        "input_shape": input_shape,
        "weights": weights,
        "activation": activation,
        "loss": loss,
        "optimizer": optimizer,
        "metrics": metrics,
        "epochs": epochs,
    }

    data_params['model_params'].append(model_params)

    with open(params_file, "w") as f:
        json.dump(data_params, f, indent=4)
    
    return test_set


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    training_model(config_path=parsed_args.config)
