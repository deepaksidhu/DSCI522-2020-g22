Predicting diabetes from patient’s health information
================
Sicheng Marc Sun, Heidi Ye </br>
11/28/2020

-   [Summary](#summary)
-   [Introduction](#introduction)
-   [Methods](#methods)
-   [Data](#data)
-   [Analysis](#analysis)
-   [Exploratory Data Analysis](#exploratory-data-analysis)
-   [Result & Discussion](#result-discussion)
    -   [Decision Tree Hyperparameter
        Tuning](#decision-tree-hyperparameter-tuning)
    -   [Naive Bayes Hyperparameter
        Tuning](#naive-bayes-hyperparameter-tuning)
    -   [Logistic Regression Hyperparameter
        Tuning](#logistic-regression-hyperparameter-tuning)
    -   [Conclusion](#conclusion)
    -   [Future directions:](#future-directions)
-   [References](#references)

Summary
=======

Here we apply three different classification models, decision tree,
Gaussian Naïve Bayes and logistic regression to predict whether a
patient has diabetes given features such as age, gender and any other
existing conditions. The objective is to find the model that scores the
highest f1 score for our target of having diabetes.

Our analysis shows that all three models performed well on an unseen
test data set. The logistic regression model has the highest f1 score
(defined by :
$f1 = \\frac{2 \\cdot precision \\cdot recall}{precision+ recall}$)
which is 0.97. The corresponding precision and recall are also 0.97. The
two other models also performed well but have slightly lower scores than
the logistic regression model. Based on these results, we can conclude
that the logistic regression model is the optimal model for this
classification problem.

Introduction
============

There are currently over 425 million people living with diabetes. With
this number on the rise, and many cases going undiagnosed, it is
increasingly crucial to be able to predict a diagnosis at an early stage
for intervention.

A machine learning model that predicts whether a person has diabetes
enables doctors to quickly identify and inform potential candidates that
need to begin blood sugar testing. A simplistic and interpretable model
can serve as a simple at home self-diagnosing method or help guide
doctors on common traits that predict diabetes.

Methods
=======

This Python programming language (Van Rossum and Drake 2009) and and the
following packages were used to perform this analysis: SKLearn
(Pedregosa et al. 2011) and Pandas (team 2020). The visualizations were
done in the R programming language (R Core Team 2019) with the following
packages: Tidyverse (Wickham 2017), Knitr (Xie 2014), Carat(Jed Wing et
al. 2019), Reticulate (Ushey, Allaire, and Tang 2020). Docopt (Keleshev
2014) in both R and Python were used.

Data
====

The dataset used for the analysis in the project is based on medical
screening questions from the patients of Sylhet Diabetes Hospital in
Bangladesh collected by M, IslamEmail, Rahatara Ferdousi, Sadikur Rahman
and Yasmin Bushra. The dataset was sourced from the UCI Machine Learning
Repository and can be found in here specifically this file.

The data used for the project is collected by Sylhet Diabetes Hospital,
by using direct questionnaires. The data set was sourced from the UCI
Machine Learning Repository (Dua and Graff 2017) and can be found here.
Each row of the dataset contains answers to common medical screening
questions, and the last column indicates whether the patient has
diabetes. This is the response variable and the target we intend to
predict on.

Analysis
========

The code used to perform the analysis and create this report can be
found in the repository
[here.](https://github.com/UBC-MDS/DSCI522-2020-g22)

Exploratory Data Analysis
=========================

We begin by splitting our data into 80% training and 20% testing
respectively. We perform our exploratory data analysis using just the
training portion.

Initially, we can see that we have 16 features and 520 observations.
With the exception of age, which is a numeric feature, the remaining are
binary and categorical in nature. Since there is no missing data, the
main transformations that are required is one hot encoding for the
categorical features and scaling for the numeric features.

With 320 observations in the positive class (has diabetes) and 200
observations in the negative class, there doesn’t appear to be any
severe class imbalance in the data. Our EDA also indicates that there
are no major class imbalance issues on a feature by feature basis. The
dataset also does not appear to have any features that seem
inappropriate to train with. In general, our dataset came fairly clean
and prepared for training without too much additional preprocessing.

Result & Discussion
===================

We use our 80% split of the training data to train three models:
decision tree, Naive Bayes and logistic regression. These models were
selected mainly for its simplicity in interpretation. In addition, we
were interested in the difference in scoring between a probabilistic and
linear model and if one approach would fit the data better than another.
The objective is the find the most interpretable model that can
accurately predict for diabetes in this dataset.

Initially, the scoring metric used for this analysis was recall, since
protecting against false positives is particularly important in
predicting disease. However, since many of the features in this dataset
are binary, scoring based on recall was consistently overfitting and
returning perfect model scores across all models. From these preliminary
results, we shifted our scoring method to the f1 score which better
balanced recall and precision.

The optimal hyperparameters for each model and their corresponding
training and validation scores can be seen in the plots below.

Decision Tree Hyperparameter Tuning
-----------------------------------

The figures below show the top four performing decision tree models
tuned for the maximum tree depth. The plots show the fixed (optimized)
tree depth, while varying for the second hyperparameter, sample leaves.
The blue lines indicate the mean train score and the red lines indicate
the mean validation score.

<div class="figure" style="text-align: center">

<img src="../results/figures/decision_tree.png" alt="Figure 3: Decision Tree hyperparameter optimization for maximum depth and mimimum leaf values" width="80%" height="60%" />
<p class="caption">
Figure 3: Decision Tree hyperparameter optimization for maximum depth
and mimimum leaf values
</p>

</div>

We can see that the optimal hyperparameter is a maximum depth of the
tree is 7 and a minimum of 1 leaves since it returns the highest
validation score of 0.96. We can also see that the other three
hyperparameter combinations return very similar scores with slightly
slower mean fit times.

Our initial hypothesis was that the decision tree model would be one of
the most interpretable models with easy to visualize decision splits.
However, with the optimal model having a depth of 7, it’s likely a
little to complicated for day to day use.

Naive Bayes Hyperparameter Tuning
---------------------------------

Below, we have the figure results of the hyperparameter tuning of
variable smoothing of the Naive Bayes model. Again, the blue line
indicates the mean train score and the red line indicates the mean
validation score.

<div class="figure" style="text-align: center">

<img src="../results/figures/gaussian_hyperparameter.png" alt="Figure 4: Naive bayes hyperparameter optimization for variable smoothing" width="80%" height="10%" />
<p class="caption">
Figure 4: Naive bayes hyperparameter optimization for variable smoothing
</p>

</div>

The optimal hyperparameter with the Naive Bayes model is when the
variable smoothing hyperparameter is set to 10^{-7}. It has a mean
validation score of 0.903. Similar to decision trees, the next four
highest ranking models (not shown above) seem to perform comparably.
This indicates that this model may not too sensitive to the tuning of
this hyperparameter.

Logistic Regression Hyperparameter Tuning
-----------------------------------------

We have the figure results of the hyperparameter tuning of C (controls
model complexity) and Solver of the Logistic Regression model below.

<div class="figure" style="text-align: center">

<img src="../results/figures/logistic_reg.png" alt="Figure 5: Logisitic regression hyperparameter optimization for variable C and Solver" width="80%" height="60%" />
<p class="caption">
Figure 5: Logisitic regression hyperparameter optimization for variable
C and Solver
</p>

</div>

The optimal hyperparameter is a regularization variable of 10 using the
liblinear solver. It has a mean validation score of 0.93. Similar to the
two models above, the next four best ranking hyperparameter combinations
have a very comparable score but again, our optimal model as the fastest
fit time.

Conclusion
----------

We can summarize each of our hyperparameter tuned models using the f1
score below:

| Model name          | F1 score | Recall score | Precision score | Accuracy |
|:--------------------|---------:|-------------:|----------------:|---------:|
| Logistic Regression |   0.9697 |       0.9697 |          0.9697 |   0.9615 |
| Decision Tree       |   0.9231 |       0.9091 |          0.9375 |   0.9038 |
| Gaussian NB         |   0.9104 |       0.9242 |          0.8971 |   0.8846 |

Table 1

We can see that the logistic regression performs the best with a mean f1
score of approximately 0.97. The two other models perform well with f1
scores of 0.92 and 0.91 for the decision tree and Naive Bayes
respectively. In this case, the linear model did fit better than the
probabilitic model.

We can conclude that based on the f1 score, the logistic regression is
the optimal model out of the three selected models for the predicting a
diabetes diagnosis.

Future directions:
------------------

Although the analysis above indicates that the logistic regression is
the best model for this dataset, there are a few improvements that can
still be made.

-   Can we optimize fit and score time through feature selection without
    compromising on model accuracy?

-   Can we make soft predictions instead of hard predictions so that
    patients have an understanding of their likelihood being diagnosed?

-   Can we perform further analysis to understand the error rate in our
    training model?

-   Can we decrease the threshold for predicting positive classes to
    improve recall scores?

References
==========

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.”
University of California, Irvine, School of Information; Computer
Sciences.
<a href="http://archive.ics.uci.edu/ml" class="uri">http://archive.ics.uci.edu/ml</a>.

M, Rahatara Ferdousi, IslamEmail, and Yasmin Bushra. 2019. “Likelihood
prediction of diabetes at early stage using data mining techniques.” In
Computer Vision and Machine Intelligence in Medical Image Analysis,
edited by Debanjan Konar Mousumi Gupta and Siddhartha Bhattacharyya, 1st
ed., 113–25. International Society for Optics; Photonics; Springer.

<div id="refs" class="references hanging-indent">

<div id="ref-caret">

Jed Wing, Max Kuhn. Contributions from, Steve Weston, Andre Williams,
Chris Keefer, Allan Engelhardt, Tony Cooper, Zachary Mayer, et al. 2019.
*Caret: Classification and Regression Training*.
<https://CRAN.R-project.org/package=caret>.

</div>

<div id="ref-docopt">

Keleshev, Vladimir. 2014. *Docopt* (version 0.6.2).

</div>

<div id="ref-scikit-learn">

Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O.
Grisel, M. Blondel, et al. 2011. “Scikit-Learn: Machine Learning in
Python.” *Journal of Machine Learning Research* 12: 2825–30.

</div>

<div id="ref-R">

R Core Team. 2019. *R: A Language and Environment for Statistical
Computing*. Vienna, Austria: R Foundation for Statistical Computing.
<https://www.R-project.org/>.

</div>

<div id="ref-reback2020pandas">

team, The pandas development. 2020. *Pandas-Dev/Pandas: Pandas* (version
1.1.1). Zenodo. <https://doi.org/10.5281/zenodo.3993412>.

</div>

<div id="ref-reticulate">

Ushey, Kevin, JJ Allaire, and Yuan Tang. 2020. *Reticulate: Interface to
’Python’*. <https://CRAN.R-project.org/package=reticulate>.

</div>

<div id="ref-Python">

Van Rossum, Guido, and Fred L. Drake. 2009. *Python 3 Reference Manual*.
Scotts Valley, CA: CreateSpace.

</div>

<div id="ref-tidyverse">

Wickham, Hadley. 2017. *Tidyverse: Easily Install and Load the
’Tidyverse’*. <https://CRAN.R-project.org/package=tidyverse>.

</div>

<div id="ref-knitr">

Xie, Yihui. 2014. “Knitr: A Comprehensive Tool for Reproducible Research
in R.” In *Implementing Reproducible Computational Research*, edited by
Victoria Stodden, Friedrich Leisch, and Roger D. Peng. Chapman;
Hall/CRC. <http://www.crcpress.com/product/isbn/9781466561595>.

</div>

</div>
