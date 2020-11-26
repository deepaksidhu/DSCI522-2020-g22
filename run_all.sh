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

# create exploratory data analysis figure and write to file 
Rscript src/eda_diab.r --train=data/train_data.csv --out_dir=results/figures/

# tune model

# model results

# render final report


