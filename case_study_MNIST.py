from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import idx2numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, top_k_accuracy_score
import click
import json
import os
import ast

"""
Bachelor Thesis: Case study using the MNIST data set
    RQ1: Collect sklearn performance measurements

Description:
    This script generates an estimator for each allowed/predefined configuration of an sklearn classifier. It will be called during the evaluation pipeline of the VaRA Tool-Suite, providing each hyperparameter configuration for a specified sklearn estimator. It trains and predicts each model, in order to finally collect all performance measurements in an output file.

Example usage (How to):
    Type in the following command in the terminal to run the script: (inside the root directory of the repo) (Note: The estimator class and params must be adjusted according to your needs)
python case_study_MNIST.py --estimator sklearn.linear_model.SGDClassifier --config_id 90 --params '["average", "early_stopping", "fit_intercept", "loss", "huber", "n_jobs", "-1", "penalty", "None", "warm_start"]' --output "scripts/io/mnist/cs_output/cs00_measurements.json"
python case_study_MNIST.py --estimator sklearn.linear_model.SGDClassifier --config_id 563 --params '["early_stopping", "fit_intercept", "loss", "epsilon_insensitive", "n_jobs", "-1", "penalty", "None"]' --output "scripts/io/mnist/cs_output/cs00_measurements.json"
"""

def load_mnist_data():
    ### PREPARATION/PREPROCESSING ###

    # Load MNIST
    X_train = './data/MNIST/train-images.idx3-ubyte'
    y_train = './data/MNIST/train-labels.idx1-ubyte'
    X_test = './data/MNIST/t10k-images.idx3-ubyte'
    y_test = './data/MNIST/t10k-labels.idx1-ubyte'

    # Converting IDX files to numpy arrays
    X_train = idx2numpy.convert_from_file(X_train)
    y_train = idx2numpy.convert_from_file(y_train)
    X_test = idx2numpy.convert_from_file(X_test)
    y_test = idx2numpy.convert_from_file(y_test)

    # Flatten the images - Reshape the 3d array (60000, 28, 28) into a 2d array (60000, 28 x 28)
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    # Normalize the data - 255 since it's a 8-bit gray-scale, resulting in [0;1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test

def model_training(model, X_train, y_train, X_test, y_test):
    ### TRAIN AND PREDICT ###
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ### METRICS ###
    # NOTE: Discuss which metrics to use in the end.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted') # weighted ?
    recall = recall_score(y_test, y_pred, average='weighted') # weighted ?
    f1 = f1_score(y_test, y_pred, average='weighted') # weighted ?
    #class_report = classification_report(y_test, y_pred, output_dict=True) # precision, recall, f1-score, support
    #top_k_accuracy = top_k_accuracy_score(y_test, y_pred, k=5) # k=5 ?
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        # 'classification_report': class_report,
        # 'top_k_accuracy': top_k_accuracy
    }

    ### LOGGING ###
    print(f"Estimator {model.get_params()}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    #print(f"Classification Report: {class_report}")
    #print(f"Top-K Accuracy: {top_k_accuracy}")
    print("\n######################################################\n")

    return model, metrics

@click.command()
@click.option('--estimator', required=True, help='Full class path, e.g. sklearn.linear_model.SGDClassifier')
@click.option('--config_id', required=True, type=int, help='Configuration ID for this run')
@click.option('--params', required=True, help='Parameter list as a Python literal, e.g. \'["loss", "hinge", "n_jobs", "-1", "penalty", "l1"]\'')
@click.option('--output', required=True, help='Path to output file for results')
def main(estimator, config_id, params, output):
    ### MODEL TRAINING (for each hyperparameter config) ###

    # Dynamically import estimator class
    module_name, class_name = estimator.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    EstClass = getattr(module, class_name)

    # Parse params
    param_list = ast.literal_eval(params)
    # Get default params to infer types
    default = EstClass()
    default_params = default.get_params()
    est_kwargs = {}
    seen_keys = set()
    i = 0
    while i < len(param_list):
        key = param_list[i]
        if key not in default_params:
            i += 1
            continue
        default_val = default_params[key]
        seen_keys.add(key)
        # Handle bools as flags (present = True)
        if isinstance(default_val, bool):
            est_kwargs[key] = True
            i += 1
        else:
            if i + 1 < len(param_list):
                val = param_list[i + 1]
                # Convert string input to correct type
                if val == 'None':
                    est_kwargs[key] = None
                elif val == 'True':
                    est_kwargs[key] = True
                elif val == 'False':
                    est_kwargs[key] = False
                elif isinstance(default_val, int) or isinstance(default_val, type(None)): # NOTE: default=None and value space: float, might be a problem.
                    try:
                        est_kwargs[key] = int(val)
                    except ValueError:
                        est_kwargs[key] = val
                elif isinstance(default_val, float):
                    try:
                        est_kwargs[key] = float(val)
                    except ValueError:
                        est_kwargs[key] = val
                else:
                    est_kwargs[key] = val
                i += 2
            else:
                i += 1
    # Set all bools not seen to False
    for k, v in default_params.items():
        if isinstance(v, bool) and k not in seen_keys:
            est_kwargs[k] = False

    estimator_instance = EstClass(**est_kwargs)

    X_train, y_train, X_test, y_test = load_mnist_data()
    performance = model_training(estimator_instance, X_train, y_train, X_test, y_test)

    # Save the final model and its performance to the output file (-> NOTE: same filepath as stated in sklearn project from my oot TS repo)
    # NOTE: Might exclude estimator and params later. See what is really needed in the end.
    result = {
        "estimator": estimator,
        "params": performance[0].get_params(),
        "performance": performance[1]
    }
    
    # Load existing results if file exists
    if os.path.exists(output):
        with open(output, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[str(config_id)] = result

    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()