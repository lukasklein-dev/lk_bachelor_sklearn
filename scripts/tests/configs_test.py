import unittest
import sys
import os

"""
Enter in the terminal to run and test the script:
PYTHONPATH=. python -m unittest scripts.tests.configs_test
"""

# Projekt-Root zum sys.path hinzufügen, damit "scripts" importiert werden kann
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import scripts.yml_sklearn_estim_gen as estim_gen

class TestYmlSklearnEstimGen(unittest.TestCase):
    def test_generate_estimators(self):
        # Minimal YAML config für SGDClassifier
        yaml_path = "scripts/io/mnist/configurations/sgd-configs.yml"
        estimator_name = "sklearn.linear_model.SGDClassifier"
        estimators = estim_gen.generate_estimators(yaml_path, estimator_name)
        self.assertIsInstance(estimators, list)
        self.assertGreater(len(estimators), 0)
        # Teste, ob das erste Element ein Tupel (name, estimator) ist
        name, est = estimators[0]
        self.assertIsInstance(name, (str, int))
        self.assertTrue(hasattr(est, "fit"))

if __name__ == "__main__":
    unittest.main()