
import sys
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#Define constants:
DATA_PATH = '../datasets/processed/datasetJazzyDescriptors.csv'
OUTPUT_FILE_PATH = '../outliers/outlier_info_from_jazzy.csv'
TARGET_VAR = 'logS'

#Define RFR hyperparametres:
N_ESTIMATORS = 2250
MAX_FEATURES = 0.45

#Define outlier detector parameters:
ACCEPTED_MAX_ERROR = 3
INITIAL_ERROR_VALUE = 15
NONE_RUN_LIMIT = 20
ERROR_DROP_RATE = 0.1


def prepare_data(data, target_var):
    """
    Prepares the train and test sets by separating features and target variable.

    Args:
        data (pd.DataFrame): The dataset containing features and target.
        target_var (str): The name of the target variable column.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Features array.
            - y (np.ndarray): Target variable array.
    """
    features_to_drop = [target_var, 'molindx']
    X = data.drop(columns=features_to_drop).to_numpy()
    y = data[target_var].to_numpy()
    return X, y

def log_results(outfile, step, idx, prediction_error, predicted_value, true_value, mae):
    """
    Logs the results to the output file.
    
    Args:
        outfile (file object): The file object to write results to.
        step (int): Current iteration or step number.
        idx (int): The molecular index of the data point.
        prediction_error (float): The error of the prediction.
        predicted_value (float): The predicted value.
        true_value (float): The actual value.
        mae (float): The mean absolute error of the model.
    """
    outfile.write(f'{step},{idx},{prediction_error:.4f},{predicted_value:.4f},{true_value:.4f},{mae:.4f}\n')
    outfile.flush()

def load_dataset(data_path):
    """
    Loads a dataset from the specified path with error handling.
    
    Args:
        file_path (str): The full path to the file.
    """
    dataset_path = Path(data_path)
    
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found at path: {dataset_path.resolve()}")
    
    try:
        return pd.read_csv(dataset_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file at {dataset_path.resolve()} is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Dataset file at {dataset_path.resolve()} contains parsing errors: {e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred while loading the dataset: {e}")
    
def ensure_directory_exists(file_path):
    """
    Ensures the directory for the given file path exists.
    
    Args:
        file_path (str): The full path to the file.
    """
    file_path = Path(file_path)
    directory = file_path.parent
    
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory created at: {directory.resolve()}")
    else:
        print(f"Directory already exists: {directory.resolve()}")

def main():
    try:
        data = load_dataset(DATA_PATH)
    except Exception as e:
        print(e)
        sys.exit(1)

    ensure_directory_exists(OUTPUT_FILE_PATH)
    with open(OUTPUT_FILE_PATH, 'w') as outfile:
        outfile.write('step,molindx,prediction_error,predicted_value,true_value,mae\n')

        error_value = INITIAL_ERROR_VALUE
        step_number = 0
        number_of_nones = 0

        while error_value > ACCEPTED_MAX_ERROR:
            step_number += 1
            print(f"Running step {step_number} ...")

            rnd_num = random.randint(0, 1_000_000)

            rfr = RandomForestRegressor(
                random_state=rnd_num,
                n_estimators=N_ESTIMATORS,
                max_features=MAX_FEATURES
            )

            train_set, test_set = train_test_split(
                data, shuffle=True, random_state=rnd_num, test_size=0.15
            )

            X_train, y_train = prepare_data(train_set, TARGET_VAR)
            X_test, y_test = prepare_data(test_set, TARGET_VAR)

            rfr.fit(X_train, y_train)

            y_pred = rfr.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

            diff = np.abs(y_test - y_pred)
            max_diff_idx = diff.argmax()
            fidx = test_set['molindx'].iloc[max_diff_idx]

            if diff[max_diff_idx] >= error_value:
                log_results(outfile, step_number, fidx, diff[max_diff_idx], y_pred[max_diff_idx], y_test[max_diff_idx], mae)

                data = data.drop(data[data['molindx'] == fidx].index)
                number_of_nones = 0
            else:
                number_of_nones += 1

            if number_of_nones >= NONE_RUN_LIMIT:
                error_value *= (1 - ERROR_DROP_RATE)
                print(f"Error decreases ... {error_value}")
                number_of_nones = 0

if __name__ == "__main__":
    main()