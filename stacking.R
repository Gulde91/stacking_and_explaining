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
library(ParBayesianOptimization)

# Sourcing helper-functions
source("./helper_functions.R")

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

# Resampling data and creating cv folds
reshuffel <- sample(1:nrow(train_data))
train_shuffled <- train_data[reshuffel, ]
train_targets_shuffled <- train_targets[reshuffel]

cv_folds <- 5
cv_index <- cut(1:nrow(train_data), cv_folds, labels = 1:cv_folds)

# Creating parameters bound for tuning
bounds <- list(eta = c(0.01, 0.5),
               max_depth = c(2L, 8L),
               subsample = c(0.3, 0.8),
               min_child_weight = c(0, 10), 
               colsample_bytree = c(0.1, 1), 
               scale_pos_weight = c(0L, 10L),
               gamma = c(0, 5), 
               lambda = c(0, 4), 
               alpha = c(0, 4),
               units_l1 = c(1L, 64L),
               units_l2 = c(1L, 64L)
               )

# Tuning (stacked) model with bayes optimization 
system.time(
bayesres <- bayesOpt(FUN = cv_bayes, # scorefunction to optimize
                     bounds = bounds,
                     initPoints = 12,
                     iters.n = 1,
                     iters.k = 1,
                     acq = "ei",
                     gsPoints = 10,
                     parallel = FALSE,
                     otherHalting = list(minUtility = 0.005, timeLimit = 60*60*30),
                     verbose = 2)
)
bayesres$scoreSummary

opt_xgb_params <- bayesres$scoreSummary[which.max(bayesres$scoreSummary$Score), 
                                        c("eta", "max_depth", "subsample", 
                                          "min_child_weight", "colsample_bytree", 
                                          "scale_pos_weight", "gamma", "lambda", "alpha")]

opt_nn_params <- bayesres$scoreSummary[which.max(bayesres$scoreSummary$Score), 
                                        c("units_l1", "units_l2")]


# Creating stacked model ----
xgb_params <- list(
  eta = 0.05,
  max_depth = 4,
  gamma = 0,
  min_child_weight = 0,
  subsample = 0.8,
  colsample_bytree = 0.8,
  lambda = 0
)

nn_params <- list(
  units = c(64, 64, 1),
  input_shape = dim(train_data)[[2]],
  activation = c("relu", "relu")
)

cv_preds <- list()

for (i in 1:cv_folds) {
  
  # Defining train and validation data
  cv_val <- train_data_shuffled[cv_index == i, ] %>% as.matrix()
  cv_train <- train_data_shuffled[cv_index != i, ] %>% as.matrix()
  
  cv_val_target <- train_targets_shuffled[cv_index == i]
  cv_train_target <- train_targets_shuffled[cv_index != i]
  
  # Fitting xgboost model
  cv_xgb_model <- xgboost(
    data = as(cv_train, "dgCMatrix"), 
    label = cv_train_target,
    objective = "reg:squarederror",
    params = xgb_params, 
    nrounds = 100,
    verbose = 0)
  
  # Fitting nn model
  cv_nn_model <- keras_model_sequential() %>%
    layer_dense(units = nn_params$units[1], 
                activation = nn_params$activation[1],
                input_shape = dim(train_data)[[2]]) %>%
    layer_dense(units = nn_params$units[2], 
                activation = nn_params$activation[2]) %>% 
    layer_dense(units = nn_params$units[3])
  
  cv_nn_model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = "mae"
  )
  
  cv_nn_model %>% fit(cv_train, cv_train_target, epochs = 80, 
                      batch_size = 16, verbose = 0)
  
  # Predicting on validation data
  cv_xgb_preds <- predict(cv_xgb_model, cv_val)
  cv_nn_preds <- predict(cv_nn_model, cv_val) %>% as.numeric()
  
  cv_preds[[i]] <- list(target = cv_val_target,
                        xgb_preds = cv_xgb_preds,
                        nn_preds = cv_nn_preds)
  
}

cv_preds <- bind_rows(cv_preds) %>% as.data.frame()

# Fitting meta learner
meta_learner <- ranger(target ~ ., data = cv_preds,
                       mtry = 2, num.trees = 20, 
                       importance = "impurity")

# Fitting base learners
xgb_model <- xgboost(
  data = as(as.matrix(train_data), "dgCMatrix"), 
  label = train_targets,
  objective = "reg:squarederror",
  params = xgb_params, 
  nrounds = 100,
  verbose = 0)

nn_model <- keras_model_sequential() %>%
  layer_dense(units = nn_params$units[1], 
              activation = nn_params$activation[1],
              input_shape = dim(train_data)[[2]]
             ) %>%
  layer_dense(units = nn_params$units[2], 
              activation = nn_params$activation[2]) %>% 
  layer_dense(units = nn_params$units[3]) 

nn_model %>% compile(optimizer = "rmsprop", loss = "mse", metrics = "mae")

history <- nn_model %>% 
  fit(as.matrix(train_data), train_targets, epochs = 80, 
      batch_size = 16, verbose = 0)

# Creating predict function for the stacked model
predict_stacked <- function(model_list, input_data) {
  
  stopifnot(is.list(model_list), 
            names(model_list) == c("xgb", "nn", "meta_learner"),
            is.data.frame(input_data))
  
  score_matrix <- as.matrix(input_data)
  
  # Individual model predictions
  xgb_preds <- predict(model_list$xgb, as(score_matrix, "dgCMatrix"), type = "prob")
  nn_preds <- predict(model_list$nn, score_matrix, type = "prob") %>% as.numeric()
  
  # Meta learner predictions
  pred_df <- data.frame(xgb_preds = xgb_preds, nn_preds = nn_preds)
  preds <- predict(model_list$meta_learner, pred_df)$prediction
  
  return(preds)
}

# Evaluation
rmse(predict(xgb_model, as.matrix(test_data)), test_targets)
rmse(as.numeric(predict(nn_model, as.matrix(test_data))), test_targets)
rmse(predict_stacked(list(xgb = xgb_model, nn = nn_model, meta_learner = meta_learner),
                     test_data), test_targets)

rmse(predict(xgb_model, as.matrix(train_data)), train_targets)
rmse(as.numeric(predict(nn_model, as.matrix(train_data))), train_targets)
rmse(predict_stacked(list(xgb = xgb_model, nn = nn_model, meta_learner = meta_learner),
                     train_data), train_targets)


# explaining stacked model ----
model_list <- list(xgb = xgb_model, nn = nn_model, meta_learner = meta_learner)

stacked_explain <- DALEX::explain(model_list, 
                                  train_data,
                                  y = train_targets,
                                  predict_function = predict_stacked,
                                  label = "stacked model"
)

system.time(
  shap_stacked_model <- predict_parts(
    explainer = stacked_explain,
    new_observation = test_data[1, ],
    type = "shap",
    B = 20)
)

plot(shap_stacked_model) %>% plotly::ggplotly()

