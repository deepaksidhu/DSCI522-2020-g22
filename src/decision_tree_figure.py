# author: Heidi Ye, Gurdeepak Sidhu
# date: 2020-11-27

"""This script reads the model model scores and creates a plot for each model based on the hyperparameter values and accuracy.

Usage: report_figures.py --model=<model> --save_figures=<save_figures> 

Options:
 --model=<model>                - Path to the model scores
--save_figures=<save_figures>   - Path to saving the figures
"""

import pandas as pd
import pickle
import string
import matplotlib.pyplot as plt
from sklearn import tree




tree_object = pickle.load(open("../results/models/decisiontreeclassifier", 'rb'))

tree.plot_tree(tree_object)

plt.savefig('decisiontree.png')