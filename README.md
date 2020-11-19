# DSCI522-2020-g22
DSCI 522 Group 22 Repo

The main objective of the exploratory data analysis (EDA) is to understand the dataset, its limitations and to determine if there are any special considerations to make based on the research question. Preliminary EDA would consist of running the `.info()` and `.describe()` functions to understand if there are any missing values or values that don't make sense (ie. negative ages). This will help determine if preprocessing steps such as imputation, scaling one-hot encoding or even dropping certain features are necessary. 

The second part of the EDA process is to perform data visualizations on the raw data. Part of this can be done through Pandas Profiling. The visualizations provided for each feature can help determine if there is class imbalance in the dataset. If so, the data cleaning process would require changes to the data or training methods. This is also a good opportunity to check if the numeric features are normally distributed or skewed in one direction. Again, this will give a better picture of how complete the data is and how representative the examples are of a general population. 

The final results will consist of:
 * a decision tree diagram outlining the split of the features
 * a table summarizing the weighted importance of each feature
 * a table summarizing the accuracy of the model and hyperparameters that were used
