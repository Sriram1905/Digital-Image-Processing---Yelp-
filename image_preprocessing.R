library(devtools)

install.packages("imager")

install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("visNetwork", repos="https://cran.rstudio.com")

install.packages("mxnet")
library(imager)
library(mxnet)
library(readr)
library(plyr)
#-------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# Using Mxnet
Feature_extraction = mx.model.load("F:/SEMESTER2(4subjects)/R/R Project/Project - Yelp Image Processing/Inception/Inception_BN", iteration=39)


rm_noise_img = as.array(mx.nd.load("F:/SEMESTER2(4subjects)/R/R Project/Project - Yelp Image Processing/Inception/Inception/mean_224.nd")[["mean_img"]])
##train_photo_to_biz_ids <- read.csv("C:/Users/vineeth raghav/Downloads/Uconn/R Proj/train_photo_to_biz_ids/train_photo_to_biz_ids.csv")
source_path = "F:/SEMESTER2(4subjects)/R/R Project/Project - Yelp Image Processing/Processed Images"

myrange = 3

preproc.image <- function(image, rm_noise_img) 
{
  # Image Resizing
  shape <- dim(image)
  #Getting the the shortest size of the image in order to resize it
  short_width <- min(shape[1:2])
  #Normalizing the picture to the lowest dimension
  xx <- floor((shape[1] - short_width) / 2)
  yy <- floor((shape[2] - short_width) / 2)
  crop_img <- crop.borders(image, xx, yy)
  # resized the image to 224 x 224, which is required as the input format to the incpetion network
  resize_img <- resize(crop_img, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resize_img) * 255
  dim(arr) <- c(224, 224, 3)
  # Removing the noise from the image
  processed_image <- arr - rm_noise_img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(processed_image) <- c(224, 224, 3, 1)
  return(processed_image)
}
