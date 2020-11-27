# author: Gurdeepak Sidhu
# date: 2020-11-25

"Creates eda plots for the pre-processed/cleaned training data from the Diabetes data set(from https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv).
Saves the figure as a pdf and png file.

Usage: src/eda_diab.r --train=<train> --out_dir=<out_dir>
  
Options:
--train=<train>                   Path (including filename) to preprocessed data (which needs to be saved as a csv file)
--out_dir=<out_dir>               Path to directory where the plots should be saved
" -> doc

library(tidyverse)
library(docopt)
library(ggthemes)
library(ggridges)
theme_set(theme_minimal())

opt <- docopt(doc)

main <- function(train, out_dir) {
  # Reading the train data
  train_data <- read.csv(train)
  
  # visualize  age predictor distribution by class
  age_plot <- train_data %>% 
    select(age,class) %>% 
    gather(key = predictor, value = value, -class) %>% 
    mutate( predictor = str_to_title(predictor))  %>% 
    ggplot(aes(x = value, y = class, colour = class, fill = class)) +
    geom_density_ridges(alpha = 0.8) +
    ggtitle("Age") +
    scale_fill_tableau() +
    scale_colour_tableau() +
    guides(fill = FALSE, color = FALSE) +
    theme(axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          plot.title = element_text(hjust = 0.5))
    ggsave(paste0(out_dir, "/age_distributions.png"), 
         age_plot,
         width = 8, 
         height = 10)
  
  # visualize categorical predictor distributions by class
    categorical_plot <-  train_data %>% 
    select(-age) %>% 
    gather(key = predictor, value = value, -class) %>% 
    mutate(predictor = str_replace_all(predictor, "_", " "),
           predictor = str_to_title(predictor)) %>% 
    ggplot(aes(x = value, colour = class, fill = class)) +
    facet_wrap(. ~ predictor, scale = "free", ncol = 5) +
    geom_bar(position = "dodge") +
    scale_fill_tableau() +
    scale_colour_tableau() +
    guides(fill = FALSE, color = FALSE) +
    theme(axis.title.x = element_blank(),
          axis.title.y = element_blank())
    ggsave(paste0(out_dir, "/categorical_distributions.png"), 
         categorical_plot,
         width = 8, 
         height = 10)
}

main(opt[["--train"]], opt[["--out_dir"]])