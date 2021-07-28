



rmse <- function(preds, label){
  sqrt(mean((preds - label)^2))
}


cv_base_learners_preds <- function(df_train, target, cv_folds = 5, cv_index, model_params) {
  
  stopifnot(is.data.frame(df_train), is.numeric(target), is.numeric(cv_folds),
            is.factor(cv_index), is.list(model_params), 
            names(model_params) %in% c("xgb_params", "nn_params"))
  
  cv_preds <- list()
  
  for (i in 1:cv_folds) {
    
    # Defining train and validation data
    cv_val <- df_train[cv_index == i, ] %>% as.matrix()
    cv_train <- df_train[cv_index != i, ] %>% as.matrix()
    
    cv_val_target <- target[cv_index == i]
    cv_train_target <- target[cv_index != i]
    
    # Fitting xgboost model
    cv_xgb_model <- xgboost(
      data = as(cv_train, "dgCMatrix"), 
      label = cv_train_target,
      objective = "reg:squarederror",
      params = model_params$xgb_params$params, 
      nrounds = model_params$xgb_params$nround,
      nthread = 1,
      verbose = 0)
    
    # Fitting nn model
    nn_params <- model_params$nn_params
    
    cv_nn_model <- keras_model_sequential() %>%
      layer_dense(units = nn_params$units_l1,
                  activation = "relu",
                  input_shape = dim(cv_train)[[2]]) %>%
      layer_dense(units = nn_params$units_l2, 
                  activation = "relu") %>% 
      layer_dense(units = 1)
    
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
                          nn_preds = cv_nn_preds#,
                          #cv_index = i
                          )
    
  }
  
  cv_preds <- bind_rows(cv_preds) %>% as.data.frame()
  
  return(cv_preds)
}

cv_bayes <- function(eta, max_depth, subsample, colsample_bytree, min_child_weight,
                     scale_pos_weight, gamma, lambda, alpha, nround,
                     units_l1, units_l2, mtry, num_trees) {
  
  # Defining base learners parameters
  xgb_params <- list(
    params = list(eta = eta, 
                  max_depth = max_depth, 
                  subsample = subsample, 
                  min_child_weight = min_child_weight, 
                  colsample_bytree = colsample_bytree, 
                  scale_pos_weight = scale_pos_weight, 
                  gamma = gamma, 
                  lambda = lambda, 
                  alpha = alpha
                  ),
    nround = nround)
  
  nn_params <- list(
    units_l1 = units_l1,
    units_l2 = units_l2
  )
  
  # Cross validation base learners
  cv_preds <- cv_base_learners_preds(train_shuffled, train_targets_shuffled, 
                                     cv_folds, cv_index,
                                     list(xgb_params = xgb_params, 
                                          nn_params = nn_params))
  # Cross validation meta learner
  rmse_result <- numeric()
  
  for (i in 1:cv_folds) {
    
    rf_train_data <- cv_preds[cv_index != i, ]
    rf_val_data <- cv_preds[cv_index == i, c("xgb_preds", "nn_preds")]
    
    cv_rf_model <- ranger(target ~ ., data = rf_train_data,
                          mtry = mtry, num.trees = num_trees,
                          importance = "impurity",
                          num.threads = 1)
    
    rf_preds <- predict(cv_rf_model, rf_val_data)$prediction
    rmse_result[i] <- rmse(rf_preds, cv_preds$target[cv_index == i])    
  }
  
  output <- list(
    Score = -mean(rmse_result)
  )
  
  return(output)
}

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
