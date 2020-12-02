# Makefile
# Gurdeepak Sidhu & Kevin Shahnazari, Dec 2020
#
# This driver script completes the predictive modelling of
# the diabetes dataset from (https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv) 
# by creating three predictive models and comparing the accuracy of each model.
# This script takes no arguments.
#
# Usage: make directory     # This is needed to create all the directories
# Usage: make all           # To execute all the scripts to create figures,csv files and final report
# Usage: clean

all :  report/diabetes_predict_report.md  src/diabetes_eda.md
	
# download the data
data/raw_data.csv : src/downloadData.py 
	python src/downloadData.py --file_path=https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv --saving_path=data/raw_data.csv

# clean, pre-process data
data/cleaned_data.csv : data/raw_data.csv src/clean_data.py
	python src/clean_data.py --file_path=data/raw_data.csv --saving_path=data/cleaned_data.csv

# Split data into 80% train, 20% test
data/train_data.csv data/test_data.csv : data/cleaned_data.csv src/split_data.py 
	python src/split_data.py --input_file_path=data/cleaned_data.csv --saving_path_train=data/train_data.csv --saving_path_test=data/test_data.csv --test_size=0.2

#run eda report
src/diabetes_eda.md : data/train_data.csv data/test_data.csv src/diabetes_eda.Rmd
	Rscript -e "rmarkdown::render('src/diabetes_eda.Rmd',output_format='github_document')"

# create exploratory data analysis figure and write to file 
results/figures/age_distributions.png results/figures/categorical_distributions.png  : data/train_data.csv src/eda_diab.r
	if [ ! -d "results/figures" ] ; then mkdir results/figures; fi
	Rscript src/eda_diab.r --train=data/train_data.csv --out_dir=results/figures/

# tune model and output results
results/models/decisiontreeclassifier results/models/gaussiannb results/models/logisticregression results/model_scores/decisiontreeclassifier_hyperparameters.csv results/model_scores/gaussiannb_hyperparameters.csv results/model_scores/logisticregression_hyperparameters.csv results/model_scores/test_scores.csv : data/train_data.csv data/test_data.csv src/model_train.py
	if [ ! -d "results/model_scores" ] ; then mkdir results/model_scores; fi
	if [ ! -d "results/models" ] ; then mkdir results/models; fi
	python src/model_train.py --train_data_path="data/train_data.csv" --test_data_path="data/test_data.csv" --save_dir_models="results/models/" --save_dir_results="results/model_scores/"

results/figures/decision_tree.png results/figures/gaussian_hyperparameter.png results/figures/logistic_reg.png : results/model_scores/decisiontreeclassifier_hyperparameters.csv results/model_scores/gaussiannb_hyperparameters.csv results/model_scores/logisticregression_hyperparameters.csv src/model_figures.r 
	if [ ! -d "results/figures" ] ; then mkdir results/figures; fi
	Rscript src/model_figures.r --model=results/model_scores/ --save_figures=results/figures

# render final report
report/diabetes_predict_report.md : report/diabetes_predict_report.Rmd results/figures/decision_tree.png results/figures/gaussian_hyperparameter.png results/figures/logistic_reg.png results/model_scores/decisiontreeclassifier_hyperparameters.csv results/model_scores/gaussiannb_hyperparameters.csv results/model_scores/logisticregression_hyperparameters.csv results/model_scores/test_scores.csv results/figures/age_distributions.png results/figures/categorical_distributions.png 
	Rscript -e "rmarkdown::render('report/diabetes_predict_report.Rmd', output_format = 'github_document')"

clean :
	rm -rf data/*
	rm -rf src/diabetes_eda_files/ src/diabetes_eda.md
	rm -rf results/*
	rm -rf report/diabetes_predict_report.html report/diabetes_predict_report.md
