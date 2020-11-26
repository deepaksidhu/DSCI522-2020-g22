# author: Gurdeepak Sidhu
# date: 2020-11-25

"Creates eda plots for the pre-processed/cleaned training data from the Diabetes data set(from https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv).
Saves the figure as a pdf and png file.

Usage: src/eda_diab.r --preprocessed=<preprocessed> --out_dir=<out_dir>
  
Options:
--preprocessed=<preprocessed>     Path (including filename) to preprocessed data (which needs to be saved as a csv file)
--out_dir=<out_dir>               Path to directory where the plots should be saved
" -> doc

library(tidyverse)
library(docopt)
library(ggthemes)
theme_set(theme_minimal())

opt <- docopt(doc)

main <- function(train, out_dir) {
  
  # visualize predictor distributions by class
  train_data <- read_csv(train) %>% 
    gather(key = predictor, value = value, -class) %>% 
    mutate(predictor = str_replace_all(predictor, "_", " "),
           predictor = str_to_title(predictor)) %>% 
    ggplot(aes(x = class, colour = class, fill = class)) +
    facet_wrap(. ~ predictor, scale = "free", ncol = 4) +
    geom_histogram(stat = "count") +
    scale_fill_tableau() +
    scale_colour_tableau() +
    guides(fill = FALSE, color = FALSE) +
    theme(axis.title.x = element_blank(),
          axis.title.y = element_blank())
    ggsave(paste0(out_dir, "/predictor_distributions.png"), 
         train_data,
         width = 8, 
         height = 10)
}

main(opt[["--preprocessed"]], opt[["--out_dir"]])