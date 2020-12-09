FROM continuumio/anaconda3:2020.11

RUN conda install docopt==0.6.2 -y

RUN apt-get update

RUN apt-get install r-base -y

RUN Rscript -e "install.packages('knitr')"

RUN Rscript -e 'install.packages(c("docopt", "tidyverse", "ggridges", "ggthemes"))'

RUN Rscript -e 'install.packages(c("ggplot2", "stringr", "caret", "reticulate"))'