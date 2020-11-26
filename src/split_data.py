# author: Kevin Shahnazari
# date: 2020-11-25

"""
This script Splits the raw cleaned data to train and test splits
based on the user input and saves them into two separate csv files

Usage: clean_data.py --input_file_path=<input_file_path>  --saving_path_train=<saving_path_train>  --saving_path_test=<saving_path_test>  --test_size=<test_size>

Options:
--input_file_path=<file_path>   Path to the cleaned input data file
--saving_path_train=<saving_path_train>  Path the training data file must be saved as csv file
--saving_path_test=<saving_path_test>  Path the testing data file must be saved as csv file
--test_size=<saving_path_test>  The proportion of test data to all the data. must be between 0 and 1.
"""

import pandas as pd
from docopt import docopt
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)


def main(input_file_path, saving_path_train, saving_path_test, test_size):
    """
    The main function of script
    which splits the cleaned data to train and test
    portions for the predictive model

    Args:
        input_file_path (string): the file path to cleaned data file
        saving_path_train (string): the file path the script will
        save the train data to the csv file.
        saving_path_test (string): the file path the script will
        save the test data to the csv file.
        test_size (float) : the test portion of the data. must be
        between 0 and 1.

    Returns:
    0 if main was successful
    -1 if main failed.
    """
    # read in data
    try:
        df = pd.read_csv(input_file_path)
    except Exception as e:
        print(f"The script failed to open the cleaned data file with the error {e}")
        return -1

    # Check test size is valid
    try:
        test_size = float(test_size)
        if test_size < 0 or test_size > 1:
            print("The test_size argument must be between 0 and 1")
            return -1
    except:
        print("The test_size argument must be a numeric number")
        return -1

    # Split dataframe
    try:
        train_data, test_data = train_test_split(
            df, test_size=test_size, random_state=123
        )
    except Exception as e:
        print(f"The script failed to split data with error {e}")
        return -1

    # Save data

    try:
        # save train portion
        train_data.to_csv(saving_path_train, index_label=False, index=False)
        # save test portion
        test_data.to_csv(saving_path_test, index_label=False, index=False)
    except Exception as e:
        print(f"The script failed to save the save train or test with the error {e}")
        return -1
    return 0


if __name__ == "__main__":
    main(
        opt["--input_file_path"],
        opt["--saving_path_train"],
        opt["--saving_path_test"],
        opt["--test_size"],
    )
