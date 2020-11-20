# author: Sicheng Marc Sun
# date: 2020-11-20
#
"This script downloads data from a given URL
usage: downloadData.R --file_path=<file_path>
options:
--file_path=<file_path>  Path to the data file
" -> doc


library(tidyverse)
library(docopt)

opt <- docopt(doc)

main <- function(file_path){
    df <- read.csv(file_path)
    write.csv(x = df,file = 'data/written.csv')
}

main(opt$file_path)
