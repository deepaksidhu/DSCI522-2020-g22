from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, make_scorer, precision_score

import pandas as pd
import pickle
import string

train_data = "train_data.csv"
test_data = "test_data.csv"
save_dir_models = "./"
save_dir_results = "./"


def train_report_save_model(
    model,
    hyperparameters,
    X_train,
    y_train,
    X_test,
    y_test,
    save_dir_models,
    save_dir_results,
):
    pipe = make_pipeline(col_trans, model)

    def score_func(y_true, y_pred, **kwargs):
        return recall_score(y_true, y_pred, pos_label="Positive")

    scorer = make_scorer(score_func)

    clf = GridSearchCV(
        pipe, hyperparameters, n_jobs=-1, return_train_score=True, scoring=scorer
    )

    res = clf.fit(X_train, y_train)

    # Save the best model to pickle file
    model_name = type(model).__name__.lower()
    best_model = res.best_estimator_[model_name]
    pickle.dump(best_model, open(save_dir_models + model_name, "wb"))

    # Save hyperparameter cross-validation results to csv file
    pd.DataFrame(res.cv_results_).to_csv(
        save_dir_results + model_name + "_hyperparameters.csv",
        index_label=False,
        index=False,
    )

    # Return Recall of best metric on testing data
    y_test_predict = res.best_estimator_.predict(X_test)
    return recall_score(y_test, y_test_predict, pos_label="Positive")


# Read data and split X and y

train_data = pd.read_csv(train_data)
test_data = pd.read_csv(test_data)

X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

# Defining the Column transformer and Pipeline

num = ["age"]
cat = list(X_train.columns)
cat.remove("age")

col_trans = ColumnTransformer(
    [("num", StandardScaler(), num), ("cat", OneHotEncoder(drop="if_binary"), cat)]
)

# Hyperparameter tuning of the models and saving recall score of tests

# Decision Tree
model = DecisionTreeClassifier()

decision_tree_pipe_hyperparamters = {
    "decisiontreeclassifier__max_depth": range(1, 10),
    "decisiontreeclassifier__min_samples_leaf": range(1, 5),
}

print(
    train_report_save_model(
        model,
        decision_tree_pipe_hyperparamters,
        X_train,
        y_train,
        X_test,
        y_test,
        save_dir_models,
        save_dir_results,
    )
)
