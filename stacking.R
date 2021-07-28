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

# Resampling data and creating cv folds ----
reshuffel <- sample(1:nrow(train_data))
train_shuffled <- train_data[reshuffel, ]
train_targets_shuffled <- train_targets[reshuffel]

cv_folds <- 5
cv_index <- cut(1:nrow(train_data), cv_folds, labels = 1:cv_folds)

# Creating parameters bound for tuning ----
bounds <- list(eta = c(0.01, 0.5),
               max_depth = c(2L, 8L),
               subsample = c(0.3, 0.8),
               min_child_weight = c(0, 10), 
               colsample_bytree = c(0.1, 1), 
               scale_pos_weight = c(0L, 10L),
               gamma = c(0, 5), 
               lambda = c(0, 4), 
               alpha = c(0, 4),
               nround = c(10L, 100L),
               units_l1 = c(8L, 128L),
               units_l2 = c(8L, 128L)
               )

# Tuning (stacked) model with bayes optimization ----
system.time(
bayesres <- bayesOpt(FUN = cv_bayes, # scorefunction to optimize
                     bounds = bounds,
                     initPoints = 13,
                     iters.n = 1,
                     iters.k = 1,
                     acq = "ei",
                     gsPoints = 100,
                     parallel = FALSE,
                     otherHalting = list(minUtility = 0.005, timeLimit = 60*60*30),
                     verbose = 2)
)

opt_xgb_params <- bayesres$scoreSummary[
  which.max(bayesres$scoreSummary$Score), 
  c("eta", "max_depth", "subsample", "min_child_weight", 
    "colsample_bytree", "scale_pos_weight", "gamma", "lambda", "alpha")] %>% 
  as.list()

opt_xgb_nround <- bayesres$scoreSummary[which.max(bayesres$scoreSummary$Score), 
                                        "nround"] %>% as.numeric()

opt_nn_params <- bayesres$scoreSummary[which.max(bayesres$scoreSummary$Score), 
                                       c("units_l1", "units_l2")] %>% as.list()

# Training base learners with optimal parameters ----
system.time(
xgb_model <- xgboost(
  data = as(as.matrix(train_data), "dgCMatrix"), 
  label = train_targets,
  objective = "reg:squarederror",
  params = opt_xgb_params,
  nrounds = opt_xgb_nround,
  nthread = 1,
  verbose = 0)
)

tictoc::tic()
nn_model <- keras_model_sequential() %>%
  layer_dense(units = opt_nn_params$units_l1,
              activation = "relu",
              input_shape = dim(train_data)[[2]]) %>%
  layer_dense(units = opt_nn_params$units_l2,
              activation = "relu") %>% 
  layer_dense(units = 1)

nn_model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = "mae"
  )

nn_model %>% fit(as.matrix(train_data), train_targets, epochs = 80, 
                 batch_size = 16, verbose = 0)
tictoc::toc()

# Training meta learner
tictoc::tic()
base_learners_preds <- cv_base_learners_preds(
  train_shuffled, train_targets_shuffled, cv_folds, cv_index,
  list(xgb_params = list(params = opt_xgb_params, nround = opt_xgb_nround), 
       nn_params = opt_nn_params)
  )
tictoc::toc()

meta_learner <- ranger(target ~ ., data = base_learners_preds,
                       mtry = 2, num.trees = 20, 
                       importance = "impurity",
                       num.threads = 1)

# Evaluation
model_list <- list(xgb = xgb_model, nn = nn_model, meta_learner = meta_learner)

rmse(predict(xgb_model, as.matrix(test_data)), test_targets)
rmse(as.numeric(predict(nn_model, as.matrix(test_data))), test_targets)
rmse(predict_stacked(model_list, test_data), test_targets)

rmse(predict(xgb_model, as.matrix(train_data)), train_targets)
rmse(as.numeric(predict(nn_model, as.matrix(train_data))), train_targets)
rmse(predict_stacked(model_list, train_data), train_targets)

# Explaining stacked model ----
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

