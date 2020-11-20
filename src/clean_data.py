# author: Kevin Shahnazari
# date: 2020-11-20

"""This script reads the diabetes data from a given path and saves the clean
data in the specified path

Usage: python clean_data.py --file_path=<file_path>  --saving_path=<saving_path>

Options:
--file_path=<file_path>   Path to the data file
--saving_path=<saving_path>  Path the data file must be saved
"""

import pandas as pd
from docopt import docopt

opt = docopt(__doc__)


def main(file_path, saving_path):
    # read in data
    diabetes = pd.read_csv(file_path)

    # Drop first unnamed column
    diabetes = diabetes.iloc[:, 1:]

    # Clean column names
    diabetes.columns = diabetes.columns.str.replace("\s+", "_")
    diabetes.columns = diabetes.columns.str.replace("\s+", "_").str.lower()

    diabetes.to_csv(saving_path, index_label=False)


if __name__ == "__main__":
    main(opt["--file_path"], opt["--saving_path"])
