# author: Kevin Shahnazari
# date: 2020-11-20

"""This script reads the diabetes data from a given path and saves the clean
data in the specified path

Usage: clean_data.py --file_path=<file_path>  --saving_path=<saving_path>

Options:
--file_path=<file_path>   Path to the raw data file
--saving_path=<saving_path>  Path the data file must be saved
"""

import pandas as pd
from docopt import docopt

opt = docopt(__doc__)


def main(file_path, saving_path):
    """
    The main function of the script
    Gets the filepath to the raw data file of diabetes
    and cleans the data and saves the dataframe to another
    csv in the given saving path

    Args:
        file_path (string): file path to the raw data file
        saving_path (string): saving path for the cleaned data

    Returns:
        0 if main was sucessful
        -1 if main failed
    """
    try:
        # read in data
        diabetes = pd.read_csv(file_path)
    except Exception as e:
        print(f"The script failed to read the file with the error {e}")
        return -1

    try:
        # Drop first unnamed column
        diabetes = diabetes.iloc[:, 1:]

        # Clean column names
        diabetes.columns = diabetes.columns.str.replace("\s+", "_")
        diabetes.columns = diabetes.columns.str.replace("\s+", "_").str.lower()

        diabetes.to_csv(saving_path, index_label=False, index=False)

    except Exception as e:
        print(f"The script failed to save the clean data with the error {e}")
        return -1

    return 0


if __name__ == "__main__":
    main(opt["--file_path"], opt["--saving_path"])
