library(e1071)
library(caret)


raw_data = read.csv("F:/SEMESTER2(4subjects)/R/R Project/Project - Yelp Image Processing/features.csv")
raw_data_subset = raw_data[,3:1000]
ncol(raw_data_subset)
pca_500 = preProcess(x = raw_data_subset, method = 'pca', pcaComp = 500)
training_set = predict(pca_500, raw_data_subset)
ncol(training_set)
View(training_set)
write.csv(training_set, file = "F:/SEMESTER2(4subjects)/R/R Project/Project - Yelp Image Processing/PCA_data_target_500.csv")