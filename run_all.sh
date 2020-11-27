# run_all.sh
# Gurdeepak Sidhu, November 2020
#
# This driver script completes the predictive modelling of
# the diabetes dataset from (https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv) 
# by creating three predictive models and comparing the accuracy of each model.
# This script takes no arguments.
#
# Usage: bash run_all.sh

# download the data
python src/downloadData.py --file_path=https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv --saving_path=data/raw_data.csv

#run eda report
Rscript -e "rmarkdown::render('src/diabetes_eda.Rmd',output_format='github_document')"

# clean, pre-process data
python src/clean_data.py --file_path=data/raw_data.csv --saving_path=data/cleaned_data.csv

# Split data into 80% train, 20% test
python src/split_data.py --input_file_path=data/cleaned_data.csv --saving_path_train=data/train_data.csv --saving_path_test=data/test_data.csv --test_size=0.2

# create exploratory data analysis figure and write to file 
Rscript src/eda_diab.r --train=data/train_data.csv --out_dir=results/figures/

# tune model and output results
python src/model_train.py --train_data_path="data/train_data.csv" --test_data_path="data/test_data.csv" --save_dir_models="results/models/" --save_dir_results="results/model_scores/"


# model figures
Rscript src/model_figures.r --model=results/model_scores/ --save_figures=results/figures

# render final report


