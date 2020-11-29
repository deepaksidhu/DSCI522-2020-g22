# author: Heidi Ye, Gurdeepak Sidhu
# date: 2020-11-27

"This script reads the model scores from three different classification algorithms(Decision Tree, Logistic Regression, Naive Bayes model)
and creates a plot for the train score and test score for the hyperparameter optimization for each model.

Usage: report_figures.py --model=<model> --save_figures=<save_figures> 

Options:
--model=<model>                 - Path to the model scores
--save_figures=<save_figures>   - Path to saving the figures
" -> doc

library(tidyverse)
library(docopt)
library(ggthemes)
library(ggridges)
theme_set(theme_minimal())


opt <- docopt(doc)
  
  
main <- function(model, out_dir) {
   
    #Checking if the file path exists or not
    if (file.exists(paste0(model, "/gaussiannb_hyperparameters.csv")) ==TRUE){
      # Reading the train data
      gaussian <- read.csv(paste0(model, "/gaussiannb_hyperparameters.csv"))
    } else {
      print("No such directory exist, please check whether correct directory path is entered")
    }
    #Plotting the hyperparameters for the Gausssian Distribution
    gaussian_plot <- gaussian %>% select(mean_train_score,mean_test_score, param_gaussiannb__var_smoothing) %>%
    gather(key = split, value = score, -param_gaussiannb__var_smoothing) %>%
    mutate( split = str_to_title(split),
              split = str_replace_all(split, "_", " ")) %>%
    ggplot(aes(x= param_gaussiannb__var_smoothing, y = score, color = split  ))+
    geom_point()+
    geom_line(alpha = 0.5)+
    scale_x_continuous(trans='log10')+
    labs(y="Score" , x = "Variable Smoothing", color="", title ="f1 score, Naive Bayes")+
    theme_grey()
    ggsave(paste0(out_dir, "/gaussian_hyperparameter.png"),
           gaussian_plot,
           width = 8,
           height = 10)
    
    
    #Checking if the file path exists or not
    if (file.exists(paste0(model, "/decisiontreeclassifier_hyperparameters.csv")) ==TRUE){
      # Reading the train data
      decision <- read.csv(paste0(model, "/decisiontreeclassifier_hyperparameters.csv"))
    } else {
      print("No such directory exist, please check whether correct directory path is entered")
    }
    
    #Plotting the hyperparameters for Decisiton tree classifier
    decision_plot <-decision %>% 
    select(mean_train_score,mean_test_score, 
           param_decisiontreeclassifier__min_samples_leaf, 
           param_decisiontreeclassifier__max_depth ) %>% 
    gather(key = split, value = score, 
           -c(param_decisiontreeclassifier__min_samples_leaf, 
              param_decisiontreeclassifier__max_depth) ) %>%
    mutate( split = str_to_title(split),
              split = str_replace_all(split, "_", " ")) %>%
    ggplot(aes(x= param_decisiontreeclassifier__max_depth, y = score, color = split  ))+
    geom_point()+
    geom_line(alpha = 0.5) +
    facet_wrap(. ~param_decisiontreeclassifier__min_samples_leaf , scale = "free", ncol = 2)+
    labs(y="Score" , x = "Max Depth", color="", title ="f1 score, sample leaves = 1,2,3,4")+ theme_grey()
    ggsave(paste0(out_dir, "/decision_tree.png"),
           decision_plot,
           width = 8,
           height = 10)
    
    
   #Check if file for logistic regression exists
    if (file.exists(paste0(model, "/logisticregression_hyperparameters.csv")) ==TRUE){
      # Reading the train data
      logistic <- read.csv(paste0(model, "/logisticregression_hyperparameters.csv"))
    } else {
      print("No such directory exist, please check whether correct directory path is entered")
    }
    
    #Plotting the hyperparameter distribution for logistic regression
    logistic_plot <- logistic %>% 
      select(mean_train_score,mean_test_score, 
             param_logisticregression__solver, param_logisticregression__C) %>% 
      gather(key = split, value = score, 
             -c(param_logisticregression__solver, param_logisticregression__C) ) %>%
      mutate( split = str_to_title(split),
              split = str_replace_all(split, "_", " ")) %>% 
      ggplot(aes(x= param_logisticregression__C, y = score, color = split  ))+
      geom_point()+
      geom_line(alpha = 0.5) +
      facet_wrap(. ~param_logisticregression__solver , scale = "free", ncol = 2)+
      scale_x_continuous(trans='log10')+
      labs(y="Score" , x = "Hyperparameter C", color="", title ="f1 score, Linear Solver")+
      theme_grey()
    ggsave(paste0(out_dir, "/logistic_reg.png"),
           logistic_plot,
           width = 8,
           height = 10)
}

main(opt[["--model"]], opt[["--save_figures"]])           
           
           
          




