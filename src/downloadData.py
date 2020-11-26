# author: Sicheng Marc Sun
# date: 2020-11-19

"""This script downloads data from a given URL

Usage: downloadData.py --file_path=<file_path>  --saving_path=<saving_path> 

Options:
--file_path=<file_path>   Path to the data file
--saving_path=<saving_path> Path to save the downloaded file
"""


import pandas as pd
from docopt import docopt

opt = docopt(__doc__)


def main(file_path, saving_path):
    """
    The main function of script
    which downloads the file from file_path
    and saves it into the saving path.

    Args:
        file_path (string): the file path the script needs
        to download.
        saving_path (string): the file path the script will
        save the downloaded csv file to.

    Returns:
    0 if main was successful
    -1 if main failed.
    """
    # read in data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"The script failed to download the file with the error {e}")
        return -1

    # save the data
    try:
        df.to_csv(saving_path)
    except Exception as e:
        print(f"The script failed to save the file with the error {e}")
        return -1

    return 0


if __name__ == "__main__":
    main(opt["--file_path"], opt["--saving_path"])
