# lk_bachelor_sklearn

This repository contains all scripts related to the experiment of my bachelor's thesis and are needed for evaluating scikit-learn classifiers on the MNIST dataset.

In particular, there is the main script `case_study_MNIST.py` which will be called per SLURM job, used for collecting the perforance measurements (RQ1). And there is `case_study_MNIST_hyperopt.py` needed for evaluating RQ2. It includes code for hyperparameter optimization using hyperopt-sklearn.