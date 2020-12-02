Exploratory data analysis of the Diabetes Risk dataset
================
Gurdeepak Sidhu <br>
November 28, 2020

Summary of the data set
=======================

The dataset used for the analysis in the project is based on medical
screening questions from the patients of Sylhet Diabetes Hospital in
Bangladesh collected by M, IslamEmail, Rahatara Ferdousi, Sadikur Rahman
and Yasmin Bushra (M and Bushra 2019). The dataset was sourced from the
UCI Machine Learning Repository (Dua and Graff 2017) specifically [this
file](https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv).

Each observation in the data set is a patient, and attributes
corresponds to the medical screening questions asked such as age, sex,
obesity and others. It includes the diagnosis for the patient
(positive(Diabetic) or negative(Not Diabetic)), and the diagnosis was
conducted and approved by a certified doctor. There are 520 observations
in the data set, and 17 features. There are no observations with missing
values in the data set. The number of observations in each classes are
shown in the table below.

| Positive Diabetes | Negative Diabetes |
|------------------:|------------------:|
|               320 |               200 |

Table 1. Counts of observation for each class.

Partition the data set into training and test sets
==================================================

Before proceeding further, we will split the data such that 80% of
observations are in the training and 20% of observations are in the test
set. Below we list the counts of observations for each class:

| Data partition | Positive Diabetes | Negative Diabetes |
|:---------------|------------------:|------------------:|
| Training       |               254 |               162 |
| Test           |                66 |                38 |

Table 2. Counts of observation for each class for each data partition.

There seems to be an issue of class imbalance especially on the training
set, but it is not that alarming, we have decided to immediately start
our modeling plan with not addressing the class imbalance problem. But
however, if during initial model building phase, there are indicators
that the model makes a lot more mistakes on positive cases) then we will
employ techniques that deals with class imbalance such as SMOTE,
undersampling, and oversampling, in the hope that we will get better
predictive models.

Exploratory analysis on the training data set
=============================================

To look at whether each of the predictors might be useful to predict the
diabetes class, we plotted the barplots of each predictor based on the
target class from the training data set and colored the distribution by
class (Negative: blue and positive: orange). Since the dataset has
mostly categorical features and one numeric feature (age), we created a
separate histogram for the `age` feature. Based on the histogram of the
categorical predictors it seems the variables such as partial paresis,
polydipsia, sudden weight loss, and polyuria seems to be significant in
predicting diabetes cases. We will pay a close attention to this
variables in the predictive models trained.

Density plot for the `age` features in the Diabetes data set.

<div class="figure">

<img src="diabetes_eda_files/figure-gfm/Age class distributions-1.png" alt="Figure 1. Distribution of training set age predictor for the Positive and Negative diabetes cases" width="70%" height="70%" />
<p class="caption">
Figure 1. Distribution of training set age predictor for the Positive
and Negative diabetes cases
</p>

</div>

Histogram for the categorical features(predictors) in the Diabetes data
set.

<div class="figure">

<img src="diabetes_eda_files/figure-gfm/categorical predictor distributions-1.png" alt="Figure 2. Distribution of training set predictors for the Positive and Negative diabetes cases" width="70%" />
<p class="caption">
Figure 2. Distribution of training set predictors for the Positive and
Negative diabetes cases
</p>

</div>

References
==========

<div id="refs" class="references hanging-indent">

<div id="ref-Dua2019">

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.”
University of California, Irvine, School of Information; Computer
Sciences. <http://archive.ics.uci.edu/ml>.

</div>

<div id="ref-Islametal">

M, Rahatara Ferdousi, IslamEmail, and Yasmin Bushra. 2019. “Likelihood
prediction of diabetes at early stage using data mining techniques.” In
*Computer Vision and Machine Intelligence in Medical Image Analysis*,
edited by Debanjan Konar Mousumi Gupta and Siddhartha Bhattacharyya, 1st
ed., 113–25. International Society for Optics; Photonics; Springer.

</div>

</div>
