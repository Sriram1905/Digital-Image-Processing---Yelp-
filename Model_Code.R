install.packages("caTools")
install.packages("randomForest")

#Importing dataset
library(xgboost)
library(Matrix)
library(caTools)
library(randomForest)
library(plyr)
dataset = read.csv("C:/Users/sriram rajagopalan/Downloads/Uconn/R Proj/PCA_data_target_500.csv")

#Training - test split

set.seed(123)
split = sample.split(dataset$Level_Zero, SplitRatio = 0.75)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split == FALSE)
training_set1 = training_set[-1:-10]
test_set1 = test_set[-1:-10]

#---------------------------------------------------------------------------------------------
#RandomForest

#Level 0

#Target Variable
Y_Target = training_set$Level_Zero

#Fitting the model in Training Dataset
rf_model = randomForest(x = training_set1, y= Y_Target,ntree=500)
#Predict test results
pred_rf_0 = predict(rf_model, newdata=test_set1)
predictions_0 = (ifelse(pred_rf_0>0.5,1,0))
#Confusion Matrix
actual_0 = test_set$Level_Zero
table(actual_0,predictions_0)

#--------------------------------------------------------------------------------------------

#Level 1

#Target Variable
Y_Target = training_set$Level_One

#Fitting the model in Training Dataset
rf_model_1 = randomForest(x = training_set1, y= Y_Target,ntree=500)
#Predict test results
pred_rf_1 = predict(rf_model_1, newdata=test_set1)
predictions_1 = (ifelse(pred_rf_1>0.5,1,0))
#Confusion Matrix
actual_1 = test_set$Level_One
table(actual_1,predictions_1)

#--------------------------------------------------------------------------------------------

#Level 2

#Target Variable
Y_Target = training_set$Level_two

#Fitting the model in Training Dataset
rf_model_2 = randomForest(x = training_set1, y= Y_Target,ntree=500)
#Predict test results
pred_rf_2 = predict(rf_model_2, newdata=test_set1)
predictions_2 = (ifelse(pred_rf_2>0.5,1,0))
#Confusion Matrix
actual_2 = test_set$Level_two
table(actual_2,predictions_2)

#--------------------------------------------------------------------------------------------

#Level 3

#Target Variable
Y_Target = training_set$Level_three

#Fitting the model in Training Dataset
rf_model_3 = randomForest(x = training_set1, y= Y_Target,ntree=500)
#Predict test results
pred_rf_3 = predict(rf_model_3, newdata=test_set1)
predictions_3 = (ifelse(pred_rf_3>0.5,1,0))
#Confusion Matrix
actual_3 = test_set$Level_three
table(actual_3,predictions_3)

#-------------------------------------------------------------------------------------------

#Level 4

#Target Variable
Y_Target = training_set$Level_four

#Fitting the model in Training Dataset
rf_model_4 = randomForest(x = training_set1, y= Y_Target,ntree=500)
#Predict test results
pred_rf_4 = predict(rf_model_4, newdata=test_set1)
predictions_4 = (ifelse(pred_rf_4>0.5,1,0))
#Confusion Matrix
actual_4 = test_set$Level_four
table(actual_4,predictions_4)

#-------------------------------------------------------------------------------------------

#Level 5

#Target Variable
Y_Target = training_set$Level_five

#Fitting the model in Training Dataset
rf_model_5 = randomForest(x = training_set1, y= Y_Target,ntree=500)
#Predict test results
pred_rf_5 = predict(rf_model_5, newdata=test_set1)
predictions_5 = (ifelse(pred_rf_5>0.5,1,0))
#Confusion Matrix
actual_5 = test_set$Level_five
table(actual_5,predictions_5)

#--------------------------------------------------------------------------------------------

#Level 6

#Target Variable
Y_Target = training_set$Level_six

#Fitting the model in Training Dataset
rf_model_6 = randomForest(x = training_set1, y= Y_Target,ntree=500)
#Predict test results
pred_rf_6 = predict(rf_model_6, newdata=test_set1)
predictions_6 = (ifelse(pred_rf_6>0.5,1,0))
#Confusion Matrix
actual_6 = test_set$Level_six
table(actual_6,predictions_6)

#---------------------------------------------------------------------------------------------

#Level 7

#Target Variable
Y_Target = training_set$Level_seven

#Fitting the model in Training Dataset
rf_model_7 = randomForest(x = training_set1, y= Y_Target,ntree=500)
#Predict test results
pred_rf_7 = predict(rf_model_7, newdata=test_set1)
predictions_7 = (ifelse(pred_rf_7>0.5,1,0))
#Confusion Matrix
actual_7 = test_set$Level_seven
table(actual_7,predictions_7)

#----------------------------------------------------------------------------------------------

#Level 8

#Target Variable
Y_Target = training_set$Level_eight

#Fitting the model in Training Dataset
rf_model_8 = randomForest(x = training_set1, y= Y_Target,ntree=500)
#Predict test results
pred_rf_8 = predict(rf_model_8, newdata=test_set1)
predictions_8 = (ifelse(pred_rf_8>0.5,1,0))
#Confusion Matrix
actual_8 = test_set$Level_eight
table(actual_8,predictions_8)

#-----------------------------------------------------------------------------------------------

# prediting training data (Random Forest)

#Level 0
pred_rf_train_0 = predict(rf_model, newdata=training_set1)
predictions_train_0 = (ifelse(pred_rf_train_0>0.5,1,0))
actual_train_0 <- training_set$Level_Zero
table(actual_train_0,predictions_train_0)

#Level 1
pred_rf_train_1 = predict(rf_model_1, newdata=training_set1)
predictions_train_1 = (ifelse(pred_rf_train_1>0.5,1,0))
actual_train_1 <- training_set$Level_One
table(actual_train_1,predictions_train_1)

#Level 2
pred_rf_train_2 = predict(rf_model_2, newdata=training_set1)
predictions_train_2 = (ifelse(pred_rf_train_2>0.5,1,0))
actual_train_2 <- training_set$Level_two
table(actual_train_2,predictions_train_2)

#Level 3
pred_rf_train_3 = predict(rf_model_3, newdata=training_set1)
predictions_train_3 = (ifelse(pred_rf_train_3>0.5,1,0))
actual_train_3 <- training_set$Level_three
table(actual_train_3,predictions_train_3)

#Level 4
pred_rf_train_4 = predict(rf_model_4, newdata=training_set1)
predictions_train_4 = (ifelse(pred_rf_train_4>0.5,1,0))
actual_train_4 <- training_set$Level_four
table(actual_train_4,predictions_train_4)

#Level 5
pred_rf_train_5 = predict(rf_model_5, newdata=training_set1)
predictions_train_5 = (ifelse(pred_rf_train_5>0.5,1,0))
actual_train_5 <- training_set$Level_five
table(actual_train_5,predictions_train_5)

#Level 6
pred_rf_train_6 = predict(rf_model_6, newdata=training_set1)
predictions_train_6 = (ifelse(pred_rf_train_6>0.5,1,0))
actual_train_6 <- training_set$Level_six
table(actual_train_6,predictions_train_6)

#Level 7
pred_rf_train_7 = predict(rf_model_7, newdata=training_set1)
predictions_train_7 = (ifelse(pred_rf_train_7>0.5,1,0))
actual_train_7 <- training_set$Level_seven
table(actual_train_7,predictions_train_7)

#Level 8
pred_rf_train_8 = predict(rf_model_8, newdata=training_set1)
predictions_train_8 = (ifelse(pred_rf_train_8>0.5,1,0))
actual_train_8 <- training_set$Level_eight
table(actual_train_8,predictions_train_8)

# Predicting Training Data
rf_train_pred = data.frame(predictions_train_0,predictions_train_1,predictions_train_2,predictions_train_3,predictions_train_4,predictions_train_5,predictions_train_6,predictions_train_7,predictions_train_8)
rf_train_pred_prob = data.frame(pred_rf_train_0,pred_rf_train_1,pred_rf_train_2,pred_rf_train_3,pred_rf_train_4,pred_rf_train_5,pred_rf_train_6,pred_rf_train_7,pred_rf_train_8)

# Predicting Testing Data
rf_test_pred = data.frame(predictions_0,predictions_1,predictions_2,predictions_3,predictions_4,predictions_5,predictions_6,predictions_7,predictions_8)
rf_test_pred_prob = data.frame(pred_rf_0,pred_rf_1,pred_rf_2,pred_rf_3,pred_rf_4,pred_rf_5,pred_rf_6,pred_rf_7,pred_rf_8)

# Aggregating the predictions and adding the tags
rf_tags = list()
for (i in seq(1:nrow(rf_test_pred))) 
{
  rf_tags[i] = list(which(rf_test_pred[i,]==1)-1)
}

rf_tags = vapply(rf_tags, paste, collapse = ", ", character(1L))

# Writing the Predicted Testing data & its probabilities to file Random_Forest_Test_Predictions
write.csv(cbind(rf_test_pred,rf_test_pred_prob,rf_tags),"C:/Users/sriram rajagopalan/Downloads/Uconn/R Proj/Results/Random_Forest_Test_Predictions.csv")

# Writing the Predicted Training data & its probabilities to file Random_Forest_Train_Predictions
#write.csv(cbind(rf_train_pred,rf_train_pred_prob),"C:/Users/sriram rajagopalan/Downloads/Uconn/R Proj/Results/Random_Forest_Train_Predictions.csv")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#XGBoost

# Converting the training and testing data to Matrix format.
train_matrix = as.matrix(training_set1)
test_matrix = as.matrix(test_set1)

#-----------------------------------------------------------------------------------------------------------
#Level 0

#Target Variable
Y_Target = training_set$Level_Zero

#Fitting the model in Training Dataset
XG_model_0 = xgboost(data = train_matrix, label = Y_Target, nrounds = 100, objective = "binary:logistic")
#Predict test results
pred_XG_0 = predict(XG_model_0, newdata=test_matrix)
XG_predictions_0 = (ifelse(pred_XG_0>0.5,1,0))
#Confusion Matrix
XG_actual_0 = test_set$Level_Zero
table(XG_actual_0,XG_predictions_0)

#----------------------------------------------------------------------------------------------------------
#Level 1

#Target Variable
Y_Target = training_set$Level_One

#Fitting the model in Training Dataset
XG_model_1 = xgboost(data = train_matrix, label = Y_Target, nrounds = 100 , objective = "binary:logistic")
#Predict test results
pred_XG_1 = predict(XG_model_1, newdata=test_matrix)
XG_predictions_1 = (ifelse(pred_XG_1>0.5,1,0))
#Confusion Matrix
XG_actual_1 = test_set$Level_One
table(XG_actual_1,XG_predictions_1)

#---------------------------------------------------------------------------------------------------------------------

#Level 2

#Target Variable
Y_Target = training_set$Level_two

#Fitting the model in Training Dataset
XG_model_2 = xgboost(data = train_matrix, label = Y_Target, nrounds = 100 , objective = "binary:logistic")
#Predict test results
pred_XG_2 = predict(XG_model_2, newdata=test_matrix)
XG_predictions_2 = (ifelse(pred_XG_2>0.5,1,0))
#Confusion Matrix
XG_actual_2 = test_set$Level_two
table(XG_actual_2,XG_predictions_2)

#----------------------------------------------------------------------------------------------------------------------

#Level 3

#Target Variable
Y_Target = training_set$Level_three

#Fitting the model in Training Dataset
XG_model_3 = xgboost(data = train_matrix, label = Y_Target, nrounds = 100 , objective = "binary:logistic")
#Predict test results
pred_XG_3 = predict(XG_model_3, newdata=test_matrix)
XG_predictions_3 = (ifelse(pred_XG_3>0.5,1,0))
#Confusion Matrix
XG_actual_3 = test_set$Level_three
table(XG_actual_3,XG_predictions_3)

#---------------------------------------------------------------------------------------------------------------------

#Level 4

#Target Variable
Y_Target = training_set$Level_four

#Fitting the model in Training Dataset
XG_model_4 = xgboost(data = train_matrix, label = Y_Target, nrounds = 100 , objective = "binary:logistic")
#Predict test results
pred_XG_4 = predict(XG_model_4, newdata=test_matrix)
XG_predictions_4 = (ifelse(pred_XG_4>0.5,1,0))
#Confusion Matrix
XG_actual_4 = test_set$Level_four
table(XG_actual_4,XG_predictions_4)

#----------------------------------------------------------------------------------------------------------------------


#Level 5

#Target Variable
Y_Target = training_set$Level_five

#Fitting the model in Training Dataset
XG_model_5 = xgboost(data = train_matrix, label = Y_Target, nrounds = 100 , objective = "binary:logistic")
#Predict test results
pred_XG_5 = predict(XG_model_5, newdata=test_matrix)
XG_predictions_5 = (ifelse(pred_XG_5>0.5,1,0))
#Confusion Matrix
XG_actual_5 = test_set$Level_five
table(XG_actual_5,XG_predictions_5)

#--------------------------------------------------------------------------------------------------------------------------

#Level 6

#Target Variable
Y_Target = training_set$Level_six

#Fitting the model in Training Dataset
XG_model_6 = xgboost(data = train_matrix, label = Y_Target, nrounds = 100 , objective = "binary:logistic")
#Predict test results
pred_XG_6 = predict(XG_model_6, newdata=test_matrix)
XG_predictions_6 = (ifelse(pred_XG_6>0.5,1,0))
#Confusion Matrix
XG_actual_6 = test_set$Level_six
table(XG_actual_6,XG_predictions_6)

#--------------------------------------------------------------------------------------------------------------------------
#Level 7

#Target Variable
Y_Target = training_set$Level_seven

#Fitting the model in Training Dataset
XG_model_7 = xgboost(data = train_matrix, label = Y_Target, nrounds = 100 , objective = "binary:logistic")
#Predict test results
pred_XG_7 = predict(XG_model_7, newdata=test_matrix)
XG_predictions_7 = (ifelse(pred_XG_7>0.5,1,0))
#Confusion Matrix
XG_actual_7 = test_set$Level_seven
table(XG_actual_7,XG_predictions_7)

#--------------------------------------------------------------------------------------------------------------------------

#Level 8

#Target Variable
Y_Target = training_set$Level_eight

#Fitting the model in Training Dataset
XG_model_8 = xgboost(data = train_matrix, label = Y_Target, nrounds = 100 , objective = "binary:logistic")
#Predict test results
pred_XG_8 = predict(XG_model_8, newdata=test_matrix)
XG_predictions_8 = (ifelse(pred_XG_8>0.5,1,0))
#Confusion Matrix
XG_actual_8 = test_set$Level_eight
table(XG_actual_8,XG_predictions_8)
#---------------------------------------------------------------------------------------------------------------------------

XG_test_pred = data.frame(XG_predictions_0,XG_predictions_1,XG_predictions_2,XG_predictions_3,XG_predictions_4,XG_predictions_5,XG_predictions_6,XG_predictions_7,XG_predictions_8)
XG_test_pred_prob = data.frame(pred_XG_0,pred_XG_1,pred_XG_2,pred_XG_3,pred_XG_4,pred_XG_5,pred_XG_6,pred_XG_7,pred_XG_8)

XG_tags = list()
for (i in seq(1:nrow(XG_test_pred))) 
{
  XG_tags[i] = list(which(XG_test_pred[i,]==1)-1)
}

XG_tags = vapply(XG_tags, paste, collapse = ", ", character(1L))

write.csv(cbind(XG_test_pred,XG_test_pred_prob,XG_tags),"C:/Users/sriram rajagopalan/Downloads/Uconn/R Proj/Results/XG_test_predictions.csv")

#---------------------------------------------------------------------------------------------------------------------------
# Averaged Model: (Averaging the Probabilies of Random forest and XG Boost)

avg_prob = data.frame()
for (i in seq(1:9))
{
  for (j in seq(1:nrow(XG_test_pred_prob)))
  {
    avg_prob[j,i] = (XG_test_pred_prob[j,i] + rf_test_pred_prob[j,i])/2
  }
}

avg_pred = (ifelse(avg_prob>0.5,1,0))
colnames(avg_pred) <- c("Label_0","Label_1","Label_2","Label_3","Label_4","Label_5","Label_6","Label_7","Label_8")
avg = as.data.frame(avg_pred)
avg_model_pred = avg$Label_8
table(XG_actual_8,avg_model_pred)
avg_tags = list()
for (i in seq(1:nrow(avg_pred))) 
{
  avg_tags[i] = list(which(avg_pred[i,]==1)-1)
}

Predicted_tags = vapply(avg_tags, paste, collapse = ", ", character(1L))

#---------------------------------------------------------------------------------------------------------------------------
# Model Comparison
set.seed(123)
split_labels = sample.split(train$labels, SplitRatio = 0.75)
test_labels = subset(train, split == FALSE)

colnames(test_labels) <- c("Business ID","Actual Tags")
write.csv(cbind(test_labels,rf_tags,XG_tags, Predicted_tags),"C:/Users/sriram rajagopalan/Downloads/Uconn/R Proj/Results/Final_Tags.csv",row.names = F)
#----------------------------------------------------------------------------------------------------------------------------
# Final Result

write.csv(cbind(test_labels,Predicted_tags),"C:/Users/sriram rajagopalan/Downloads/Uconn/R Proj/Results/Final_Result.csv",row.names = F)

#-----------------------------------------------------------------------------------------------------------------------------------
