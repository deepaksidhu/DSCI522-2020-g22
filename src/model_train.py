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
        return f1_score(y_true, y_pred, pos_label="Positive")
        # Change to bellow if we want recall score
        # return recall_score(y_true, y_pred, pos_label="Positive")

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
    return f1_score(y_test, y_test_predict, pos_label="Positive")
    # Change to below if we want recall score
    # return recall_score(y_test, y_test_predict, pos_label="Positive")


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
scores = []
models = []

# Decision Tree
model = DecisionTreeClassifier()

decision_tree_pipe_hyperparamters = {
    "decisiontreeclassifier__max_depth": range(1, 10),
    "decisiontreeclassifier__min_samples_leaf": range(1, 5),
}

score = train_report_save_model(
    model,
    decision_tree_pipe_hyperparamters,
    X_train,
    y_train,
    X_test,
    y_test,
    save_dir_models,
    save_dir_results,
)

scores.append(score)
models.append("Decision_Tree")
# GaussianNB
model = GaussianNB()

gaussiannb_pipe_hyperparamters = {
    "gaussiannb__var_smoothing": [10 ** pow for pow in range(-7, 5)]
}

score = train_report_save_model(
    model,
    gaussiannb_pipe_hyperparamters,
    X_train,
    y_train,
    X_test,
    y_test,
    save_dir_models,
    save_dir_results,
)

scores.append(score)
models.append("GaussianNB")

# Ridge

model = LogisticRegression(random_state=123, max_iter=1000)

logisticregression_pipe_hyperparamters = {
    "logisticregression__C": [10 ** pow for pow in range(-7, 2)],
    "logisticregression__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
}

score = train_report_save_model(
    model,
    logisticregression_pipe_hyperparamters,
    X_train,
    y_train,
    X_test,
    y_test,
    save_dir_models,
    save_dir_results,
)

scores.append(score)
models.append("LogisticRegression")

# Save Recall scores for models to a file

pd.DataFrame({"model_name": models, "recall_score": scores}).to_csv(
    save_dir_results + "_test_recall_scores.csv",
    index_label=False,
    index=False,
)
