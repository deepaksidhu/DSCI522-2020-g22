
Diabetes Predictor
==================

-   contributors: Heidi Ye, Gurdeepak Sidhu, Marc Sun, Kevin Shahnazari

Demo of a data analysis project for DSCI 522 (Data Science workflows); a
course in the Master of Data Science program at the University of
British Columbia.

About
-----

We attempt to build three classification models, namely logistic
regression, decision tree, and Naive Bayes model which can use the
predictors from the diabetes dataset to predict for our positive class
of having Diabetes. We compare the model predictive ability based on the
f1 score. The logistic regression model performs the best with a mean f1
score of approximately 0.97. The two other models perform well with f1
scores of 0.92 and 0.91 for the decision tree and Naive Bayes
respectively. The precision score is very satisfying, it shows that the
model has the ability to eliminate most non-diabetes cases, which helps
to save a lot of time in real life situations.

The dataset used for the analysis in the project is based on medical
screening questions from the patients of Sylhet Diabetes Hospital in
Bangladesh collected by M, IslamEmail, Rahatara Ferdousi, Sadikur Rahman
and Yasmin Bushra (M and Bushra 2019). The dataset was sourced from the
UCI Machine Learning Repository (Dua and Graff 2017) specifically [this
file](https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv).
There are 520 observations in the data set, and 17 features. There are
no observations with missing values in the data set. The number of
observations in each classes are shown in the table below.

<table>
<caption>
Table 1. Counts of observation for each class.
</caption>
<thead>
<tr>
<th style="text-align:right;">
Positive Diabetes
</th>
<th style="text-align:right;">
Negative Diabetes
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
320
</td>
<td style="text-align:right;">
200
</td>
</tr>
</tbody>
</table>

Report
------

The final report can be found
[here](http://htmlpreview.github.io/?https://raw.githubusercontent.com/UBC-MDS/DSCI522-2020-g22/main/docs/diabetes_predict_report.html).

Usage
-----

There are two suggested ways to run this analysis:

#### 1. Using Docker (recommended)

*note - the instructions in this section depends on running this in a
unix shell either Git Bash or terminal*

To run this analysis using Docker, clone/download this repository, use
the command line to navigate to the root of this project on your
computer, and then type the following (filling in
PATH\_ON\_YOUR\_COMPUTER with the absolute path to the root of this
project on your computer).

    docker run --rm -it -v $PWD:/home/dsci522-2020-g22 marcsun314/dsci522-2020-g22 make -C dsci522-2020-g22 all

To reset the repo to a clean state, with no intermediate or results
files, run the following command at the command line/terminal from the
root directory of this project:

    docker run --rm -it -v  $PWD:/home/dsci522-2020-g22 marcsun314/dsci522-2020-g22 make -C dsci522-2020-g22 clean


#### 2. Without using Docker

1.  Make sure you’ve installed all of the dependencies listed in the
    Dependencies section below.
2.  Download or clone this repository.
3.  Open a terminal session and navigate to the root of the project
    directory.
4.  Run the analysis with the following command:

`make all`

To reset the repo to a clean state, with no intermediate or results
files, run the following command at the command line/terminal from the
root directory of this project:

`make clean`

Makefile dependency map
-----------------------

Please consider the following dependency map for the make processes for
the make file.

<img src="Makefile.png" width="4336" />

Dependencies
------------

-   Python 3.8.3 and Python packages:
    -   pandas==1.1.1
    -   scikit-learn==0.23.2
    -   docopt==0.6.2
    -   matplotlib==3.3.3
-   R version 3.6.1 and R packages:
    -   knitr==1.26
    -   docopt==0.7.1
    -   tidyverse==1.3.0
    -   ggridges==0.5.2
    -   ggthemes==4.2.0
    -   ggplot2==3.3.2
    -   stringr==1.4.0
    -   caret==4.0.3
    -   reticulate==1.18
-   GNU make 4.2.1

License
-------

The Diabetes Analysis material here are licensed under the MIT LICENSE.
Anyone can’t publish this repository as their own work and should give
credit to the contributors of the project if the results of this project
are being used.

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
