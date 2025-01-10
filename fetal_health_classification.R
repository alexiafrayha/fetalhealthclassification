## FETAL HEALTH CLASSIFICATION PROJECT
# Alexia 


# Identifying if packages are already installed and, if not, installing them
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(pROC)


# Downloading dataset
url <- "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/fetal-health-classification"
dl <- "fetalhealthclassification.zip"
if(!file.exists(dl))
  download.file(url, dl)
dl <- unzip(dl)
fetalhealthclassification <- read.csv(dl, header = TRUE)

# Analysing the summary of the dataset
summary(fetalhealthclassification)

# BUILDING A HEATMAP BASED ON THE CORRELATION MATRIX
# Building the correlation matrix 
correlation_matrix <- cor(fetalhealthclassification)
# Converting the matrix into long format for plotting
correlation_df <- as.data.frame(correlation_matrix)
correlation_df$rowname <- rownames(correlation_matrix)
correlation_long <- correlation_df %>%
  pivot_longer(
    cols = -rowname,            
    names_to = "variable",      
    values_to = "correlation"   
  )
# Plotting the heatmap
ggplot(correlation_long, aes(x = rowname, y = variable, fill = correlation)) +
  geom_tile() +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0) +
  theme_minimal() +
  geom_text(aes(label = sprintf("%.2f", correlation)), size = 3) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Matrix Heatmap",
       x = "",
       y = "", 
       fill = "Correlation")

# Selecting features that will be used for prediction based on the confusion matrix - 
# we will remove features poorly correlated with fetal_health (abs(< 0.15)) and highly correlated components (> 0.8)
features <- c("baseline.value", "accelerations", "prolongued_decelerations", "uterine_contractions", "abnormal_short_term_variability", "mean_value_of_long_term_variability", "percentage_of_time_with_abnormal_long_term_variability", "histogram_variance")
outcome <- "fetal_health"
# Building a new dataset with features that will be utilized
fetalhealthclassification_optimized <- fetalhealthclassification[,c(outcome, features)]

# Plotting a bar graph to illustrate the frequency of each outcome
fetalhealthclassification_optimized %>% 
  group_by(fetal_health) %>% 
  ggplot(aes(x = factor(fetal_health))) +
  geom_bar(fill = "#F09090", alpha = 0.6) +  
  theme_minimal() +                                            
  labs(
    title = "Distribution of Fetal Health Classes",          
    x = "Fetal Health Class",                                 
    y = "Count"                                               
  ) +
  geom_text(
    stat = "count",                                           
    aes(label = after_stat(count)),                                  
    vjust = -0.5,                                            
    color = "black",                                         
    size = 4                                                 
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),   
    axis.text = element_text(size = 10),                     
    panel.grid.major = element_blank(),                     
    panel.grid.minor = element_blank()                      
  )
# From the graph, we can see that we have an imbalanced dataset. 
# We will join health class 2 and 3 into the same group ("abnormal") as both require evaluation by a healthcare professional.

# Making fetal_health a factor for training the ML models and grouping category 2 and 3 into "abnormal" 
fetalhealthclassification_optimized <- fetalhealthclassification_optimized %>%
  mutate(fetal_health = factor(ifelse(fetal_health == 1, "normal", "abnormal")))


# ANALYSING FEATURES

# Creating a long table to allow us to plot boxplots for each feature in the same plot
fetalhealthclassification_long <- fetalhealthclassification_optimized %>%
  pivot_longer(
    cols = seq(2,9), #Columns which correspond to features
    names_to = "variable",       
    values_to = "value"         
  )
# Plotting the graph
ggplot(fetalhealthclassification_long, aes(x = variable, y = value, fill = fetal_health)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    title = "Boxplots for each feature",
    x = "Feature",
    y = "Value",
    fill = "Fetal Health"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)  
  )
# From the boxplots, we see that features are measured in different scales. We must scale the features for training. 
# Scaling the features
fetalhealthclassification_scaled <- fetalhealthclassification_optimized
fetalhealthclassification_scaled[,2:9] <- scale(fetalhealthclassification_scaled[,2:9])
# Repeating the boxplots for the scaled features
# Creating a long table to allow us to plot boxplots for each scaled feature in the same plot
fetalhealthclassification_long <- fetalhealthclassification_scaled %>%
  pivot_longer(
    cols = seq(2,9), #Columns which correspond to features
    names_to = "variable",       
    values_to = "value"         
  )
# Plotting the graph
ggplot(fetalhealthclassification_long, aes(x = variable, y = value, fill = fetal_health)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    title = "Boxplots for each feature",
    x = "Feature",
    y = "Value",
    fill = "Fetal Health"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)  
  )
# Plotting the graph between abnormal short term variability and accelerations, the two features with a larger
# visual difference between normal and abnormal fetal health
# Identifying the intercept and slope for a line representing the correlation between the variables abnormal_short_term_variability and accelerations
lm(accelerations ~ abnormal_short_term_variability, data = fetalhealthclassification_scaled)
# Plotting the graph
fetalhealthclassification_scaled %>%
  ggplot(aes(abnormal_short_term_variability, accelerations)) +
  geom_point(aes(col = fetal_health)) + geom_abline(intercept = 4.461e-15, slope = -0.2796, color = "black", linewidth = 0.5)


# TRAINING AND TESTING MODELS
set.seed(1, sample.kind = "Rounding")
# Creating our training and testing sets
index <- createDataPartition(fetalhealthclassification_scaled$fetal_health, times = 1, p = 0.2, list = FALSE)
train_set <- fetalhealthclassification_scaled[-index,]
test_set <- fetalhealthclassification_scaled[index,]
# Creating our parameters for cross validation
control <- trainControl(method = "cv", number = 10)

# Generalized linear model
set.seed(1, sample.kind = "Rounding")
# Training the model with a glm algorithm
fit_glm <- train(fetal_health ~., 
                 data = train_set, 
                 method = "glm", 
                 trControl = control)
predictions_glm <- predict(fit_glm, newdata = test_set)
# Obtaining the confusion matrix between the predictions and the reference
conf_matrix_glm <- confusionMatrix(data = predictions_glm, reference = test_set$fetal_health)
# Predicting probabilities for each class
probabilities_glm <- predict(fit_glm, newdata = test_set, type = "prob")[, 2]
# Obtaining the roc curve and the area under the curve
roc_curve_glm <- roc(test_set$fetal_health, probabilities_glm)
auc_glm <- auc(roc_curve_glm)
# Saving metrics to a separate variable for comparison
metrics_glm <- c(
  conf_matrix_glm$byClass["Sensitivity"],
  conf_matrix_glm$byClass["Precision"],
  conf_matrix_glm$byClass["Specificity"],
  conf_matrix_glm$overall["Accuracy"],
  conf_matrix_glm$byClass["F1"],
  auc_glm
)
names(metrics_glm) <- c("sensitivity", "precision", "specificity", "accuracy", "f1_score", "auc")
# Plotting ROC curve and printing metrics for evaluation
plot(roc_curve_glm, main = "ROC Curve GLM", col = "blue")
knitr::kable(metrics_glm, caption = "Evaluation metrics GLM")
# The model has adequate accuracy, specificity and area under the curve. 
# However, due to the imbalanced nature of our dataset, 
# it is also important to evaluate precision, sensitivity and the F1 score, which evaluate our minority class.

# Decision tree
set.seed(1, sample.kind = "Rounding")
# Training the model 
# The tuneGrid argument defines a data frame of complexity parameters (which control the size of the decision tree) for testing
fit_dt <- train(fetal_health ~., 
                data = train_set, 
                method = "rpart", 
                tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)), 
                trControl = control)
predictions_dt <- predict(fit_dt, newdata = test_set)
# Obtaining the confusion matrix between the predictions and the reference
conf_matrix_dt <- confusionMatrix(data = predictions_dt, reference = test_set$fetal_health)
# Predicting probabilities for each class
probabilities_dt <- predict(fit_dt, newdata = test_set, type = "prob")[, 2]
# Obtaining the ROC curve and AUC
roc_curve_dt <- roc(test_set$fetal_health, probabilities_dt)
auc_dt <- auc(roc_curve_dt)
# Saving metrics to a separate variable
metrics_dt <- c(
  conf_matrix_dt$byClass["Sensitivity"],
  conf_matrix_dt$byClass["Precision"],
  conf_matrix_dt$byClass["Specificity"],
  conf_matrix_dt$overall["Accuracy"],
  conf_matrix_dt$byClass["F1"],
  auc_dt
)
names(metrics_dt) <- c("sensitivity", "precision", "specificity", "accuracy", "f1_score", "auc")
# Plotting tree structure
plot(fit_dt$finalModel, margin = 0.05) 
text(fit_dt$finalModel, cex = 0.5)
# Plotting ROC curve and printing metrics for evaluation
plot(roc_curve_dt, main = "ROC Curve Decision Tree", col = "blue")
knitr::kable(metrics_dt, caption = "Evaluation metrics DT")
# Compared to our GLM model, our decision tree-based model has similar sensitivity, 
# but better accuracy, precision, specificity and F1 score. 
# The AUC, however, is smaller for the decision tree model. 


# K Nearest Neighbours
set.seed(1, sample.kind = "Rounding")
# Training the model
# The tuneGrid argument defines a data frame of k-neighbours for testing
fit_knn <- train(fetal_health ~., 
                 data = train_set, 
                 method = "knn", 
                 tuneGrid = data.frame(k = seq(1,15,2)), 
                 trControl = control)
# Obtaining the confusion matrix between the predictions and the reference
predictions_knn <- predict(fit_knn, newdata = test_set)
conf_matrix_knn <- confusionMatrix(data = predictions_knn, reference = test_set$fetal_health)
# Predicting probabilities for each class
probabilities_knn <- predict(fit_knn, newdata = test_set, type = "prob")[, 2]
# Obtaining the ROC curve and AUC
roc_curve_knn <- roc(test_set$fetal_health, probabilities_knn)
auc_knn <- auc(roc_curve_knn)
# Saving metrics to a separate variable
metrics_knn <- c(
  conf_matrix_knn$byClass["Sensitivity"],
  conf_matrix_knn$byClass["Precision"],
  conf_matrix_knn$byClass["Specificity"],
  conf_matrix_knn$overall["Accuracy"],
  conf_matrix_knn$byClass["F1"],
  auc_knn
)
names(metrics_knn) <- c("sensitivity", "precision", "specificity", "accuracy", "f1_score", "auc")
# Plotting ROC curve and printing metrics for evaluation
plot(roc_curve_knn, main = "ROC Curve KNN", col = "blue")
knitr::kable(metrics_knn, caption = "Evaluation metrics KNN")
# This model has the highest sensitivity thus far. 
# However, its precision, accuracy, F1 score  and specificity are inferior to the decision tree model, 
# and its AUC is inferior to the glm model.


# Random Forest
set.seed(1, sample.kind = "Rounding")
# Training the model
# The tuneGrid argument defines a data frame of number of features to consider for each split in the tree for testing
fit_rf <- train(fetal_health ~., 
                data = train_set, 
                method = "rf", 
                tuneGrid = data.frame(mtry = seq(1, 8)), 
                trControl = control)
# Obtaining the confusion matrix between the predictions and the reference
predictions_rf <- predict(fit_rf, newdata = test_set)
conf_matrix_rf <- confusionMatrix(data = predictions_rf, reference = test_set$fetal_health)
# Predicting probabilities for each class
probabilities_rf <- predict(fit_rf, newdata = test_set, type = "prob")[, 2]
# Obtaining the ROC curve and AUC
roc_curve_rf <- roc(test_set$fetal_health, probabilities_rf)
auc_rf <- auc(roc_curve_rf)
# Saving metrics to a separate variable
metrics_rf <- c(
  conf_matrix_rf$byClass["Sensitivity"],
  conf_matrix_rf$byClass["Precision"],
  conf_matrix_rf$byClass["Specificity"],
  conf_matrix_rf$overall["Accuracy"],
  conf_matrix_rf$byClass["F1"],
  auc_rf
)
names(metrics_rf) <- c("sensitivity", "precision", "specificity", "accuracy", "f1_score", "auc")
# Plotting ROC curve and printing metrics for evaluation
plot(roc_curve_rf, main = "ROC Curve Random Forest", col = "blue")
knitr::kable(metrics_rf, caption = "Evaluation metrics RF")
# The Random Forest algorithm appears to be superior in all metrics when compared to the other algorithms. 


# COMPARING EVALUATION METRICS
# Building dataframe with the metrics on each model
model_comparison <- data.frame(
  GLM = metrics_glm,
  DT = metrics_dt,
  KNN = metrics_knn,
  RF = metrics_rf
)
# Plotting the table
knitr::kable(model_comparison, caption = "Comparison of Model Metrics")

# Random Forest model performed better in all metrics. 

# IMPROVING SENSITIVITY FOR OUR USE CASE
# Testing different decision thresholds 
# Creating a variable containing different thresholds for testing 
thresholds <- seq(0.1, 0.95, 0.05)
# Creating a function to predict outcomes given the probability values and a threshold, 
# generate a confusion matrix and extract our evaluation metrics 
ideal_threshold <- function(threshold){
  predicted <- as.factor(ifelse(probabilities_rf > as.numeric(threshold), "normal", "abnormal"))
  conf <- confusionMatrix(data = predicted, reference = test_set$fetal_health)
  metrics <- c(
    conf$byClass["Sensitivity"],
    conf$byClass["Precision"],
    conf$byClass["Specificity"],
    conf$overall["Accuracy"],
    conf$byClass["F1"]
  )
  names(metrics) <- c("sensitivity", "precision", "specificity", "accuracy", "f1_score")
  return(metrics)
}
# Creating a table showing the results for each threshold
threshold_results <- lapply(thresholds, ideal_threshold)
threshold_results <- as.data.frame(do.call(rbind, threshold_results)) %>% 
  mutate(threshold = thresholds)

# Plotting a graph to illustrate sensitivity and specificity across different thresholds
threshold_results %>%
  ggplot() +
  geom_line(aes(x = threshold, y = sensitivity, color = "Sensitivity")) +  
  geom_line(aes(x = threshold, y = specificity, color = "Specificity")) +  
  labs(
    title = "Sensitivity and Specificity Across Thresholds",  
    y = "Value",  
    x = "Threshold",  
    color = "Measure"  
  ) +
  theme_minimal()

# Identifying thresholds for which sensitivity is greater than 0.94
threshold_results %>% filter(sensitivity > 0.94)

# Testing values between 0.75 and 0.8 to identify ideal threshold
thresholds_refined <- seq(0.75, 0.8, 0.01)
threshold_results_refined <- lapply(thresholds_refined, ideal_threshold)
threshold_results_refined <- as.data.frame(do.call(rbind, threshold_results_refined)) %>% 
  mutate(threshold = thresholds_refined)
knitr::kable(threshold_results_refined, caption = "Comparison of Thresholds")

# Making our final predictions and confusion matrix which a threshold of 0.78
final_predictions <- as.factor(ifelse(probabilities_rf > 0.78, "normal", "abnormal"))
conf_matrix_final <- confusionMatrix(data = final_predictions, reference = test_set$fetal_health)
metrics_final <- c(
  conf_matrix_final$byClass["Sensitivity"],
  conf_matrix_final$byClass["Precision"],
  conf_matrix_final$byClass["Specificity"],
  conf_matrix_final$overall["Accuracy"],
  conf_matrix_final$byClass["F1"]
)
names(metrics_final) <- c("sensitivity", "precision", "specificity", "accuracy", "f1_score")

# Printing final confusion matrix and metrics
conf_matrix_final$table
knitr::kable(metrics_final, caption = "Final metrics for our model")