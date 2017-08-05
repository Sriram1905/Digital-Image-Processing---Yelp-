# Digital-Image-Processing---Yelp-
Deep Learning using MXnet and Inception-BN in R
 Yelp is a multinational company which hosts online reservation services and user reviews about the local businesses and restaurants around the world. 
 It has one of the largest online user community and hosts over 121 million user reviews, with tens of millions of photos uploaded. 
 Yelp extensively deploys analytics for its business solutions and to enhance its user experience, for instance, to recommend most helpful and relevant reviews for its users. 
 Every restaurant in the Yelp community is assigned a set of predefined tags based on image uploaded by the end user. 
 Our objective was to classify these restaurants with the predefined tags by processing the images without human intervention. 
 This project involves a seven-layered approach. 
 The project starts with the feature extraction process for each photo and proceeds to aggregate these features at the business level. 
 These business level features act as inputs to the binary classification models that are used for prediction. 
 XGBoost and Random Forest models were selected to yield the desired results  
 The Accuracy for each model: XGBoost 78.76% Random Forest 78.35% Averaged Model 78.84% •
 Averaged the predictions of XGBoost and Random Forest to take the best of both the models and achieve a marginally better prediction.
 This algorithm can be used to process the images uploaded in the Yelp community and automate the process of classifying a restaurant to the predefined tag
