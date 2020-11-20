# author: Sicheng Marc Sun
# date: 2020-11-19

'''This script downloads data from a given URL

Usage: downloadData.py --file_path=<file_path>  --saving_path=<saving_path> 

Options:
--file_path=<file_path>   Path to the data file
--saving_path=<saving_path> Path to save the downloaded file
'''


import pandas as pd
from docopt import docopt

opt = docopt(__doc__)

def main(file_path, saving_path):
  # read in data
  print(1)
  df = pd.read_csv(file_path)
  print(2)
  df.to_csv(saving_path)
  print(3)
# standard error function


if __name__ == "__main__":
    main(opt["--file_path"],opt["--saving_path"])
