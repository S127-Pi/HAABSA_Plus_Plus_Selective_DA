import tensorflow.compat.v1 as tf
import lcrModelAlt_hierarchical_v4

# from OntologyReasoner import OntReasoner
from loadData import *

#import parameter configuration and data paths
from config import *

#import modules
import random
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import numpy as np
import sys
import pickle
import os
import traceback
from bson import json_util
import json
import glob

train_size, test_size, train_polarity_vector, test_polarity_vector = loadHyperData(FLAGS, True)
remaining_size = 248
accuracyOnt = 0.87


# Define variabel spaces for hyperopt to run over
eval_num = 0
best_loss = None
best_hyperparams = None
lcrspace = [
                hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
                hp.quniform('keep_prob', 0.45, 0.75, 0.1),
                hp.choice('momentum', [0.85, 0.9, 0.95]),
                hp.choice('l2', [0.0001, 0.001]),
            ]

# Define objectives for hyperopt
def lcr_objective(hyperparams):
    global eval_num
    global best_loss
    global best_hyperparams

    eval_num += 1
    (learning_rate, keep_prob, momentum, l2) = hyperparams
    print(hyperparams)

    l, pred1, fw1, bw1, tl1, tr1, _, _ = lcrModel.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, accuracyOnt, test_size, remaining_size, learning_rate, keep_prob, momentum, l2)
    tf.reset_default_graph()

    # Save training results to disks with unique filenames

    print(eval_num, l, hyperparams)

    if best_loss is None or -l < best_loss:
        best_loss = -l
        best_hyperparams = hyperparams

    result = {
            'loss':   -l,
            'status': STATUS_OK,
            'space': hyperparams,
        }

    save_json_result(str(l), result)

    return result

def lcr_inv_objective(hyperparams):
    global eval_num
    global best_loss
    global best_hyperparams

    eval_num += 1
    (learning_rate, keep_prob, momentum, l2) = hyperparams
    print(hyperparams)

    l, pred1, fw1, bw1, tl1, tr1 = lcrModelInverse.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, accuracyOnt, test_size, remaining_size, learning_rate, keep_prob, momentum, l2)
    tf.reset_default_graph()

    # Save training results to disks with unique filenames

    print(eval_num, l, hyperparams)

    if best_loss is None or -l < best_loss:
        best_loss = -l
        best_hyperparams = hyperparams

    result = {
            'loss':   -l,
            'status': STATUS_OK,
            'space': hyperparams,
        }

    save_json_result(str(l), result)

    return result

def lcr_alt_objective(hyperparams):
    global eval_num
    global best_loss
    global best_hyperparams

    eval_num += 1
    (learning_rate, keep_prob, momentum, l2) = hyperparams
    print(hyperparams)

    l, pred1, fw1, bw1, tl1, tr1 = lcrModelAlt_hierarchical_v4.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, accuracyOnt, test_size, remaining_size, learning_rate, keep_prob, momentum, l2, tuning=True)
    tf.reset_default_graph()

    # Save training results to disks with unique filenames

    print(eval_num, l, hyperparams)
    result = {
        'loss':   -l,
        'status': STATUS_OK,
        'space': hyperparams,
    }
    if best_loss is None or -l < best_loss:
        best_loss = -l
        best_hyperparams = hyperparams

    # result = {
    #         'loss':   -l,
    #         'status': STATUS_OK,
    #         'space': hyperparams,
    #     }
        save_json_result(str(l), result)

    return result

# Run a hyperopt trial
def run_a_trial():
    max_evals = nb_evals = 10

    print("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        # Insert the method opbjective funtion
        lcr_alt_objective,
        # Define the methods hyperparameter space
        space     = lcrspace,
        algo      = tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open("results.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")
    print(best_hyperparams)

def print_json(result):
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))

    with open(F"results/best_hyperparameter/best_parameter_{FLAGS.da_type}_{FLAGS.year}.txt", 'w') as file:
        file.write(json.dumps(
            result,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        ))
    

def save_json_result(model_name, result):
    """Save json to a directory and a filename."""
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists("results/"):
        os.makedirs("results/")
    with open(os.path.join("results/", result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result(best_result_name):
    """Load json from a path (directory + filename)."""
    result_path = os.path.join("results/", best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
        )

def load_best_hyperspace():
    results = [
        f for f in list(sorted(os.listdir("results/"))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name)["space"]

def plot_best_model():
    """Plot the best model found yet."""
    space_best_model = load_best_hyperspace()
    if space_best_model is None:
        print("No best model to plot. Continuing...")
        return

    print("Best hyperspace yet:")
    print_json(space_best_model)

def delete_result_files():
    """Delete all .txt.json files in the 'results' folder."""
    results_path = 'results'
    txt_files = glob.glob(os.path.join(results_path, '*.txt.json'))
    
    for txt_file in txt_files:
        try:
            os.remove(txt_file)
        except Exception as e:
            print(f"Error deleting file {txt_file}: {e}")


while True:
    print("Optimizing New Model")
    try:
        run_a_trial()
    except Exception as err:
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
    plot_best_model()
    delete_result_files()
    break