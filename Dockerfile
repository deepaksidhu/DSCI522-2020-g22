#DSCI 522 Project
#author: Kevin Shahnazari
#Date: 7th Dec 2020

# ana Conda Docker  image
FROM continuumio/anaconda3:2020.11

# install required packages for python
RUN conda install docopt==0.6.2 -y

# Install R
RUN apt-get update

RUN apt-get install r-base -y

RUN apt-get install -y libxml2-dev libcurl4-openssl-dev libssl-dev

RUN apt-get install locate

# Install R packages
RUN Rscript -e "install.packages('knitr')"

RUN Rscript -e 'install.packages(c("docopt", "ggridges", "ggthemes"))'

RUN Rscript -e 'install.packages(c("ggplot2", "stringr", "caret", "reticulate"))'

# tidyverse needs some extra libraries so they should be installed
Run updatedb

RUN locate libicui18n.so.58

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/

ENV export LD_LIBRARY_PATH

RUN Rscript -e 'install.packages("tidyverse")'

# create r-reticulate because if this environment doesen't exist
# reticulate in r will start creating it and pre installing some 
# libraries itself so it's better to create it ourselves.

RUN conda create -n r-reticulate python=3.8 pandas=1.1.3 -y
