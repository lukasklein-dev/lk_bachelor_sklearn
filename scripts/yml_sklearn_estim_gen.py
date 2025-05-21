"""
%%%%% Sklearn Estimator Generator %%%%%
This script generates an estimator for each allowed/predefined configuration of an sklearn classifier. It reads a YAML file containing hyperparameter configurations for a specified sklearn estimator and generates Python code that creates instances of those estimators with the specified parameters.

Usage (How to):
Type in the following command in the terminal to run the script: (inside the root directory of the repo)
TODO: In the command; adjust the (1) estimator class and the paths to the (2) input YAML file (configs) and the (3) output Python file.
python scripts/yml_sklearn_estim_gen.py --estimator sklearn.linear_model.SGDClassifier --configs scripts/io/mnist/configurations/sgd-configs.yml --output scripts/io/mnist/generated_sklearn_estimators.py
"""

import argparse
import yaml
import ast


def load_configs(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def get_estimator_class(full_name):
    module_name, class_name = full_name.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name), class_name


def parse_config_list(raw_list, default_params):
    """
    Parse the raw list of hyperparameters and values, ensuring proper types.
    """
    params = {}
    seen_keys = set()
    i = 0
    while i < len(raw_list):
        key = raw_list[i]
        if key not in default_params:
            i += 1
            continue

        default_val = default_params[key]
        seen_keys.add(key)

        if isinstance(default_val, bool):
            # Boolean flags: only presence matters -> Skip bool here. They will be set in fill_binary_features()
            i += 1
        else:
            # Non-boolean: expect a value next
            if i + 1 < len(raw_list):
                raw_val = raw_list[i + 1]
                try:
                    # Try to convert the value to None or to a number (int or float)
                    if raw_val == 'None':
                        params[key] = None
                    elif '.' in raw_val or 'e' in raw_val.lower():
                        params[key] = float(raw_val)  # Convert to float if it contains a decimal point or scientific notation
                    else:
                        params[key] = int(raw_val)  # Convert to int otherwise
                except (ValueError, TypeError):
                    # If conversion fails, fall back to the default type
                    try:
                        params[key] = type(default_val)(raw_val)
                    except Exception:
                        # If casting fails, keep the raw value
                        params[key] = raw_val
                i += 2
            else:
                i += 1
    return params, seen_keys


def fill_binary_features(params, default_params, seen_keys):
    for key, default_val in default_params.items():
        if isinstance(default_val, bool):
            # true iff explicitly listed; false otherwise
            params[key] = (key in seen_keys)
    return params


def generate_estimators(yaml_path, estimator_name):
    raw = load_configs(yaml_path)
    EstClass, _ = get_estimator_class(estimator_name)

    default = EstClass()
    default_params = default.get_params()

    estimators = []

    for variant, list_str in raw.items():
        # Expect a list or a Python-literal string representing a list
        if isinstance(list_str, list):
            raw_list = list_str
        elif isinstance(list_str, str):
            # Only Python-literal format (e.g. "['a', 'b']")
            raw_list = ast.literal_eval(list_str)
        else:
            raise TypeError(f"Config for variant '{variant}' must be list or list-string, got {type(list_str)}")

        # Confirm we have a list
        if not isinstance(raw_list, list):
            raise ValueError(f"Parsed config for variant '{variant}' is not a list: {raw_list}")

        params, seen_keys = parse_config_list(raw_list, default_params)
        params = fill_binary_features(params, default_params, seen_keys)

        est = EstClass(**params)
        estimators.append((variant, est))

    return estimators


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sklearn estimators from YAML configs"
    )
    parser.add_argument(
        "--estimator", required=True,
        help="Full class path, e.g. sklearn.linear_model.SGDClassifier"
    )
    parser.add_argument(
        "--configs", required=True,
        help="Path to YAML file with estimator configs"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file to write the generated code to"
    )
    args = parser.parse_args()

    all_ests = generate_estimators(args.configs, args.estimator)

    # LOGGING
    if args.output:
        _, class_name = get_estimator_class(args.estimator) # Needed for naming the estimators
        #with open(args.output, 'w') as f:
        #    f.write(f"# Generated Python code for sklearn estimators of {class_name}:\n")
        #    #f.write(f"from sklearn.utils import {class_name}\n\n")
        #    f.write(f"from {args.estimator.rsplit('.', 1)[0]} import {class_name}\n") # import the classifier
        #    f.write("estims = []\n\n") # initialize estims list
        #    for name, est in all_ests:
        #        f.write(f"estim_{class_name}_{name} = estims.append({est})\n")
        with open(args.output, 'w') as f:
            f.write(f"# Generated Python code for sklearn estimators of {class_name}:\n")
            #f.write(f"from sklearn.utils import {class_name}\n\n")
            f.write(f"from {args.estimator.rsplit('.', 1)[0]} import {class_name}\n") # import the classifier
            f.write("estims = []\n\n") # initialize estims list
            for name, est in all_ests:
                f.write(f"estims.append(({name},{est}))\n")
        print(f"Generated {len(all_ests)} estimators.")
        print(f"Generated estimators written to {args.output}")
    else:
        for name, est in all_ests:
            print(f"[{name}] params:", est.get_params())
        print(f"\nGenerated {len(all_ests)} estimators.")
        print("No output file specified. Thus, printed to console.")
        print("To save to a file, use the --output argument.")