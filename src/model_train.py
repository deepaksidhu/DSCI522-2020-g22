# author: Kevin Shahnazari
# date: 2020-11-25

"""This script reads the Train and test data and creates 7 different output files.
this script runs hyperparameter optimization and chooses the best models based on some predefined hyperparameter settings using the f1_score metric.
The models are Decision Tree, GaussianNB and Logistic Regression.
The chosen models are saved using the pickle library and saved into the save_dir_models directory with the name of the models(folder) (3 files)
The scores for the hyperparameter optimization step for the models are saved into the save_dir_results directory(folder) with the name models__hyperparameters.csv format (3 files)
The f1 scores of the testing data is saved into the save_dir_results directory with the name test_f1_scores.csv (1 file)

Usage: clean_data.py --train_data_path=<train_data_path>  --test_data_path=<test_data_path>  --save_dir_models=<save_dir_models> --save_dir_results=<save_dir_results>

Options:
--train_data_path=<train_data_path=>   Path to the training data file
--test_data_path=<test_data_path>  Path the testing data file
--save_dir_models=<save_dir_models>   Path to save the models
--save_dir_results=<save_dir_results>  Path the save the results
"""

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, make_scorer, precision_score, f1_score

import pandas as pd
import pickle
import string
from docopt import docopt

opt = docopt(__doc__)


def main(train_data_path, test_data_path, save_dir_models, save_dir_results):

    # Read data
    try:
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
    except Exception as e:
        print(f"The script failed to read the file with the error {e}")
        return -1

    # Genarate all the output files
    res = model_result_generator(
        train_data, test_data, save_dir_models, save_dir_results
    ).genarate_result()


class model_result_generator:
    """
    This class would be used to genarate the outputs of the script
    Basically it does all the things mentioned in the docopt documentation
    For the files to be generated the genarate_result function must be call
    after initilization.
    """

    def __init__(self, train_data, test_data, save_dir_models, save_dir_results):
        """initilize class
        Args:
            train_data (pd.DataFrame): A pandas dataframe containing the data for training the models
            test_data (pd.DataFrame): A pandas dataframe containing the data for testing the models
            save_dir_models (string): The path were the best models are gonna be saved
            save_dir_results (string): The path were the hyperparameter optimization of the models are gonna be saved

        """
        self.train_data = train_data
        self.test_data = test_data
        self.save_dir_models = save_dir_models
        self.save_dir_results = save_dir_results

    def genarate_result(self):
        """
        This function creates the suitable column transformer and Also defines
        our three models which are The Decision tree, GaussianNB amd Logistic Regression
        with some suitable hyper parameters for them and calls the train_test_report function
        to genarate the best model and hyperparameter f1scores for each model.
        Also this function saves the test f1score of all
        the models into the save_dir_results folder in a csv file with the name test_f1_scores.csv

        Returns:
        -1 if there was a problem in the program
        """

        # Split into X and y

        try:
            self.X_train, self.y_train = (
                self.train_data.iloc[:, :-1],
                self.train_data.iloc[:, -1],
            )
            self.X_test, self.y_test = (
                self.test_data.iloc[:, :-1],
                self.test_data.iloc[:, -1],
            )
        except:
            print(
                "Looks like the training and testing data are not dataframe. Make sure you have read it from the train and test files"
            )
            return -1

        # Defining the Column transformer and Pipeline

        num = ["age"]
        cat = list(self.X_train.columns)
        cat.remove("age")

        col_trans = ColumnTransformer(
            [
                ("num", StandardScaler(), num),
                ("cat", OneHotEncoder(drop="if_binary"), cat),
            ]
        )

        # Hyperparameter tuning of the models and saving recall score of tests
        scores = []
        models = []

        # Decision Tree
        model = DecisionTreeClassifier()

        decision_tree_pipe_hyperparamters = {
            "decisiontreeclassifier__max_depth": range(1, 10),
            "decisiontreeclassifier__min_samples_leaf": range(1, 5),
        }

        score = self.__train_report_save_model(
            model, decision_tree_pipe_hyperparamters, col_trans
        )
        if score == -1:
            print("error hyperparameter tuning decision tree")
            return -1

        scores.append(score)
        models.append("Decision_Tree")
        # GaussianNB
        model = GaussianNB()

        gaussiannb_pipe_hyperparamters = {
            "gaussiannb__var_smoothing": [10 ** pow for pow in range(-7, 5)]
        }

        score = self.__train_report_save_model(
            model, gaussiannb_pipe_hyperparamters, col_trans
        )

        scores.append(score)
        if score == -1:
            print("error hyperparameter tuning GaussianNB")
            return -1
        models.append("GaussianNB")

        # Logistic Regression

        model = LogisticRegression(random_state=123, max_iter=1000)

        logisticregression_pipe_hyperparamters = {
            "logisticregression__C": [10 ** pow for pow in range(-7, 2)],
            "logisticregression__solver": [
                "newton-cg",
                "lbfgs",
                "liblinear",
                "sag",
                "saga",
            ],
        }

        score = self.__train_report_save_model(
            model, logisticregression_pipe_hyperparamters, col_trans
        )

        if score == -1:
            print("error hyperparameter Logistic Regression")
            return -1
        scores.append(score)
        models.append("LogisticRegression")

        # Save f1 scores for models to a file

        pd.DataFrame({"model_name": models, "f1_score": scores}).to_csv(
            self.save_dir_results + "test_f1_scores.csv",
            index_label=False,
            index=False,
        )

    def __train_report_save_model(self, model, hyperparameters, col_trans):
        """
        This function uses GridSearchCV from sklearn to search through all the
        possible hyperparameters and find the best model. After the best model
        has been selected 2 files would be genarated.
        First the model would be saved to save_dir_model folder with
        the name [model_name]. Also the results of the
        Second the crossvalidation score results for each of the hyperparameters
        would be saved to save_dir_results

        Args:
            model (sklearn.model): A sklearn model
            hyperparameters (dictionary): dictionary containing the proper hyperparameters for the model
            col_trans (sklearn ColumnTransformer): The ColumnTransformer used in the pipeline for the model

        Returns:
            -1 : if the functions gets an error
            f1score : if the functions runs correctly
        """
        try:
            # make the pipe line which has the column transformer inside
            pipe = make_pipeline(col_trans, model)

            # define the f1 scorer we want to give the GridSearchCV to optimize
            def score_func(y_true, y_pred, **kwargs):
                return f1_score(y_true, y_pred, pos_label="Positive")
                # Change to bellow if we want recall score
                # return recall_score(y_true, y_pred, pos_label="Positive")

            scorer = make_scorer(score_func)

            # Hyperparameter optimize step
            clf = GridSearchCV(
                pipe,
                hyperparameters,
                n_jobs=-1,
                return_train_score=True,
                scoring=scorer,
            )

            res = clf.fit(self.X_train, self.y_train)

            # Save the best model to pickle file
            model_name = type(model).__name__.lower()
            best_model = res.best_estimator_[model_name]
            pickle.dump(best_model, open(self.save_dir_models + model_name, "wb"))

            # Save hyperparameter cross-validation results to csv file
            pd.DataFrame(res.cv_results_).to_csv(
                self.save_dir_results + model_name + "_hyperparameters.csv",
                index_label=False,
                index=False,
            )

            # Return f1 of best metric on testing data
            y_test_predict = res.best_estimator_.predict(self.X_test)
            return f1_score(self.y_test, y_test_predict, pos_label="Positive")
            # Change to below if we want recall score
            # return recall_score(y_test, y_test_predict, pos_label="Positive")
        except Exception as e:
            print(
                f"Something unexpected happend in the train_report_save_model of the model_result_generator class with error {e}"
            )
            return -1


if __name__ == "__main__":
    main(
        opt["--train_data_path"],
        opt["--test_data_path"],
        opt["--save_dir_models"],
        opt["--save_dir_results"],
    )
