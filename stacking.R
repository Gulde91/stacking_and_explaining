# Creating a stacked model on the boston house dataset
# with xgboost and NN (keras) as meta learners and RF
# as meta learner. Explaining single instance predictions
# through Shap values (with the DALEX package)

library(keras)
library(DALEX)
library(xgboost)
library(DALEXtra)
library(dplyr)
library(ranger)

# Getting data ----
dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

gns <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)

train_data <- scale(train_data, center = gns, scale = std)
test_data <- scale(test_data, center = gns, scale = std)

train_data <- as.data.frame(train_data)
test_data <- as.data.frame(test_data)
train_targets <- as.numeric(train_targets)
test_targets <- as.numeric(test_targets)
