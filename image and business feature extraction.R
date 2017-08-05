myrange = 3

## GETTING THE LIST OF DIRECTORY
files    <- list.files(path = source_path, full.names = TRUE, recursive = FALSE)
final_file = c()
for (f in files)
{
  if(as.numeric(basename(f)) %in% myrange)
  {
    final_file = c(final_file,f)  
  }
}

###################
#Finding the features for each photo and aggregating at business level

business_feature <- list()
business_name <- list()
for (i in final_file)
{
  business_name = rbind(business_name,basename(i))
  files    <- list.files(path = i, full.names = TRUE, recursive = TRUE)
  feature = list()
  for (file in files)
  {
    image <- load.image(file)
    processed_image <- preproc.image(image, rm_noise_img)
    prob <- predict(Feature_extraction, X=processed_image)
    feature <- cbind(feature,prob)
  }
  trans_feature <- t(feature)
  .
  mean_feature <- rowMeans(apply(trans_feature,1,as.numeric),2,dims =1)
  business_feature <- rbind(business_feature,mean_feature)
}
business_feature <- cbind(business_name,business_feature)
colnames(business_feature)[1] <- "Business Name" 
colnames(business_feature)[-1] <- 1:1000
#-----------------------------------------------------------------------------------------------------------------------------
write.csv(business_feature,"F:/SEMESTER2(4subjects)/R/R Project/Project - Yelp Image Processing/features.csv")