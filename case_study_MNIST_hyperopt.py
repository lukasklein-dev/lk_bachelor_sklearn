# Experiment Main Skript Part 2 (RQ2): Hyperopt-sklearn optimizes models (with restricted search space)

from hpsklearn import HyperoptEstimator, k_neighbors_classifier, sgd_classifier
from sklearn.metrics import accuracy_score
from hyperopt import hp, tpe
from scripts.io.mnist.hyperopt_spaces import *
from case_study_MNIST import load_mnist_data
    
# %%% Models %%%

sgd_search_space = {
    'loss': hp.choice('loss', sgd_space['loss']),
    'penalty': hp.choice('penalty', sgd_space['penalty']),
    'fit_intercept': hp.choice('fit_intercept', sgd_space['fit_intercept']),
    'shuffle': hp.choice('shuffle', sgd_space['shuffle']),
    'early_stopping': hp.choice('early_stopping', sgd_space['early_stopping']),
    'warm_start': hp.choice('warm_start', sgd_space['warm_start']),
    'average': hp.choice('average', sgd_space['average']),
    'n_jobs': hp.choice('n_jobs', sgd_space['n_jobs']),
}

# TODO: Add all other classifier search spaces

# %%% Training %%%
search_space = sgd_search_space # TODO: Enter the search space of the classifier you want to use 

estim = HyperoptEstimator(
    algo=tpe.suggest,
    max_evals=3,
    #trial_timeout=300,
    classifier=sgd_classifier('clf', **search_space),
    preprocessing=[]
)

X_train, y_train, X_test, y_test = load_mnist_data()

estim.fit(X_train, y_train)
y_pred = estim.predict(X_test)

# LOGGING
best_model = estim.best_model()
print("Best Model:", best_model)
print("Hyperopt-Sklearn Accuracy Score:", accuracy_score(y_test, y_pred))

# TODO: Inwieweit macht es überhaupt Sinn hyperopt für diesen (sehr stark) limitierten search space zu verwenden?