rm(list = ls())
gc()

  # ------------------------------------------- #
  # A. Installing and Loading required packages # 
  # ------------------------------------------- #

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(rvest)) install.packages("rvest", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(purrr)) install.packages("purr", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(httr)) install.packages("httr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(dslabs)
library(ggplot2)
library(dplyr)
library(rvest)
library(lubridate)
library(purrr)
library(matrixStats)
library(readxl)
library(caret)
library(httr)

  # ------------------------------- #
  # B. Data loading and preparation #
  # ------------------------------- #

  # ~ Data loading from github ~ #
url <- "https://github.com/estFerrer/edx_choose_your_own/blob/master/finviz.xlsx?raw=true"
temp_file <- tempfile(fileext = ".xlsx")
req <- GET(url, 
           write_disk(path = temp_file))

fin_info <- readxl::read_excel(temp_file)

rm(url,req,temp_file)
gc()

  # ~ Cleaning ~ #

# 1. We create two categories: "Keep" for those observations which may provide useful information,
#    and "Remove" for those which doesn't:
fin_info$cleaning <- ifelse(fin_info$PER=="-"&fin_info$PEG=="-","Keep",
                     ifelse((fin_info$PER=="-"&fin_info$PEG!="-")|(fin_info$PER!="-"&fin_info$PEG=="-"),"Remove","Keep"))

# 2. We filter those we want to keep
fin_info <- fin_info %>% filter(cleaning=="Keep")

# 3. And assign the number 0 since those observations presented no earnings in past balance sheets
fin_info$PER[fin_info$PER=="-"] <- 0
fin_info$PEG[fin_info$PEG=="-"] <- 0

fin_info$PER <- as.numeric(fin_info$PER)
fin_info$PEG <- as.numeric(fin_info$PEG)

  # ~ Preparing main dataset ~ #

main_ds <- fin_info %>% 
  select(-c(Ticker,Industry,cleaning)) %>% # 1. Removing useless information,
  mutate(Sector = str_replace(Sector,"\\s","_"), # 2. Replacing spaces with "_" for future one-hot encoding,
         E_growth = ifelse(is.na(PER/PEG)=="TRUE",0,(PER/PEG)/100), # 3. We calculate earnings estimated growth avoiding divisions by 0,
         payer = ifelse(Dividend==0,"NO","YES")) %>% # 4. and we finally encode our dependent variable.
  select(-c(Dividend,PER,PEG))

rm(fin_info)
gc()

  # ~ Train and Validation datasets ~ #

ratio <- 1/sqrt(ncol(main_ds)-1) # Validation set ratio
set.seed(10,sample.kind = "Rounding")
validation_index <- createDataPartition(main_ds$payer, times=1, p=ratio, list=FALSE)

train_set <- main_ds %>% slice(-validation_index)
validation <- main_ds %>% slice(validation_index)

rm(validation_index)
gc()

  # ~ Scaling dataset ~ #

# 1. First we obtain scaling parameters from train_set,
normParam <- preProcess(train_set) 

# 2. Then we proceed to scale both train set and validation set with same parameters.
norm_train <- predict(normParam, train_set)
norm_validation <- predict(normParam, validation)

rm(train_set,validation,normParam)
gc()

# 3. After scaling datasets, we encode the variable "sector":
#    Each sector will compute a different variable. The company's corresponding 
#    sector will have the value 1 and all the rest will assume 0 value:
norm_train <- norm_train %>% 
  mutate(Sector = paste("Sec",Sector,sep="_"), Sec=1) %>%
  spread(key = Sector, value = Sec, fill = 0) 

norm_validation <- norm_validation %>% 
  mutate(Sector = paste("Sec",Sector,sep="_"), Sec=1) %>%
  spread(key = Sector, value = Sec, fill = 0)

  # -------------------------- #
  # C. Training and Optimizing #
  # -------------------------- #

  # ~ Single tree optimization ~ #

# 1. Sequence of possible values for complex parameter (cp):
complex <- seq(0,0.1,0.01) 

# 2. Then we use caret function train() to run the algorithm on the different values for "complex"
single_tree <- train(payer~.,method = "rpart", data = norm_train, tuneGrid = data.frame(cp = complex))

  # ~ Forest optimization ~ #

# 1. Sequence of possible values for number of variables per tree (mtry):
mtry <- seq(1,(ncol(norm_train)-1))

# 2. Then we use caret function train() to run the algorithm on the different values for "mtry"
forest <- train(payer~.,method = "rf", data = norm_train, tuneGrid = data.frame(mtry = mtry))

  # ~ ANN optimization ~ #

# 1. Sequence of possible values for number of nodes (size), regularization parameter (decay):
size <- 1:10
decay <- seq(0.001,0.01,0.001)

# 2. We use the expand.grid() function to compute all possible combinations of size-decay parameters:
annGrid <- expand.grid(size,decay)
colnames(annGrid) <- c("size","decay")

# 3. Then we use caret function train() to run the algorithm on the different values for "size" and "decay"
ANN <- train(payer~.,method = "nnet", data = norm_train, tuneGrid = annGrid, trace = FALSE)

  # ------------------- #
  # D. Final evaluation # 
  # ------------------- #

# For each model, we use the $bestTune from caret function train() to train the model with optimizing parameters
# And then we proceed to predict dividend policy on validation dataset.

final_tree <- train(payer~.,method = "rpart", data = norm_train, tuneGrid = single_tree$bestTune)
y_hat_tree <- predict(final_tree, newdata = norm_validation)

final_forest <- train(payer~.,method = "rf", data = norm_train, tuneGrid = forest$bestTune)
y_hat_forest <- predict(final_forest, newdata = norm_validation)

final_ann <- train(payer~.,method = "nnet", data = norm_train, tuneGrid = ANN$bestTune, trace = FALSE)
y_hat_ann <- predict(final_ann, newdata = norm_validation)

# Once we obtain predictions for each model, we create a data.frame with each algorithm's output
results <- data.frame(y_hat_tree = y_hat_tree,
                      y_hat_forest = y_hat_forest,
                      y_hat_ann = y_hat_ann)

# And we compute the final prediction by applying the "majority vote" rule:
results$y_hat_ens <- ifelse((rowMeans(results=="YES"))>=2/3,"YES","NO")

# Final model is evaluated on validation dataset (norm_validation)
confusionMatrix(data = as.factor(results$y_hat_ens), reference = as.factor(norm_validation$payer))$byClass

