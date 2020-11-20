# DSCI522-2020-g22
DSCI 522 Group 22 Repo

# Diabetes Risk Predictor

    - contributors: Heidi, Kevin, Marc, Deepak

Data analysis collaboration project for DSCI 522 (Data Science Workflows); a course in the Master of Data Science program at the University of British Columbia, for the school year 2020-2021.

## Project Proposal

For this data analysis project we are interested to investigate the question: based on medical screening questions, which features are important in assessing early stage risk of Diabetes for individuals? It is estimated that 463 M(million) of people worldwide have diabetes, and this number is expected to rise to 578 M in the year 2045 (Saeedi, Petershon et al., 2019). The statistics reported about the cases of diabetes is quite alarming, and most of these cases are undiagnosed. As a result, addressing the research question is crucial, since having a thorough knowledge of the presensce of which symptoms are likely to contribute to diabetes based on some screening question, will provide a precursor and direction for individuals to seek medical treatment. Some sub questions that may follow from this topic include: 1) Are traits that are commonly associated with Diabetes (ie. obesity) truely associated with the disease? 2) If our test scores are low, are there any obvious categories that are missing from the screen test?

The dataset used for the analysis in the project is based on medical screening questions from the patients of Sylhet Diabetes Hospital in Bangladesh collected by M, IslamEmail, Rahatara Ferdousi, Sadikur Rahman and Yasmin Bushra. The dataset was sourced from the UCI Machine Learning Repository and can be found in [here](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.) specifically [this file](https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv).

Each observation in the data set is a patient, and attributes corresponds to the medical screening questions asked such as age, sex, obesity and others. It includes the diagnosis for the patient ( positive( Diabetic) or negative( Not Diabetic)), and the diagnosis was conducted and approved by a certified doctor.

In order to answer the predictive question, we plan to build and compare two predictive classification models in terms of the accuracy of predictions and feature importance. The two models in considerations are, decision tree and logistic regression. Before carrying out classification on the data set, we will partition the data into a training and test split ( split 80%:20%) and perform preliminary exploratory data analysis on the training set.

The main objective of the exploratory data analysis (EDA) is to understand the dataset, its limitations and to determine if there are any special considerations to make based on the research question. Preliminary EDA would consist of running the `.info()` and `.describe()` functions to understand if there are any missing values or values that don't make sense (ie. negative ages). This will help determine if preprocessing steps such as imputation, scaling one-hot encoding or even dropping certain features are necessary. 

The second part of the EDA process is to perform data visualizations on the raw data. Part of this can be done through Pandas Profiling. The visualizations provided for each feature can help determine if there is class imbalance in the dataset. If so, the data cleaning process would require changes to the data or training methods. This is also a good opportunity to check if the numeric features are normally distributed or skewed in one direction. Again, this will give a better picture of how complete the data is and how representative the examples are of a general population. 

After perfoming the EDA, and preprocessing steps we will create a pipeline and carry out cross validation with our two models on the training set, with corresponding hyperparameter optimization for max_depth parameter for decision tree and paramter C for logistic regression. We will choose the corresponding hyperparameters with the highest validation accuracy, and use this to refit the model on the training set, and then for prediction on the test set. The accuracy score of both models on the test set will be compared to determine which model performs better in predicting diabetes diagnosis.

The final results will consist of:
 * a decision tree diagram outlining the split of the features
 * a table summarizing the weighted importance of each feature
 * a table summarizing the accuracy of the model and hyperparameters that were used
