FROM continuumio/anaconda3:2020.11

RUN conda install docopt==0.6.2 -y

RUN apt-get update

RUN apt-get install r-base -y

RUN apt-get install -y libxml2-dev libcurl4-openssl-dev libssl-dev

RUN apt-get install locate

RUN Rscript -e "install.packages('knitr')"

RUN Rscript -e 'install.packages(c("docopt", "ggridges", "ggthemes"))'

RUN Rscript -e 'install.packages(c("ggplot2", "stringr", "caret", "reticulate"))'

Run updatedb

RUN locate libicui18n.so.58

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/

ENV export LD_LIBRARY_PATH

RUN Rscript -e 'install.packages("tidyverse")'