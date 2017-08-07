# install.packages("stringr")
# install.packages("tools")
# 
 library(stringr)
 library(tools)


# Place the pictures and the files contaiting the folder - file mapping in the same directory as below
source_path = "C:/Users/sriram rajagopalan/Downloads/Uconn/R Proj/train_photos/train_photos"
target_path = "C:/Users/sriram rajagopalan/Downloads/Uconn/R Proj/train_photos/Processed_Images"
file_name = "C:/Users/sriram rajagopalan/Downloads/Uconn/R Proj/train_photo_to_biz_ids/train_photo_to_biz_ids.csv"
file_extn= '.JPG'
bad_file_start = c(".")
except_folders = c("Archives")
setwd(source_path)
getwd()


# Add / at the end if not present
source_path = ifelse(str_sub(source_path, -1) == "/", source_path, paste0(source_path,"/"))
target_path = ifelse(str_sub(target_path, -1) == "/", target_path, paste0(target_path,"/"))
# Read the csv file
source_file = read.csv(file_name)
# Get the list of folders to be created
folder_names = unique(source_file$business_id)



# Create the folders
for (folder_name in folder_names) {
  if(!(folder_name %in% except_folders))
  {
    # Create a directory. Ignore if the directory already exists
    dir.create(file.path(target_path, folder_name), showWarnings = FALSE)
    # Get the list of files under the directory
    file_names = source_file[source_file$business_id == folder_name,][1]
    print(paste(nrow(file_names), "Number of files found for the folder",folder_name))
    # Move the files from the parent folder to the sub folders
    for(i in 1:nrow(file_names)) 
    {
      file_name =  file_names[i,]
      # Move (not copy) the files to the target folder
      file.rename(paste0(source_path,file_name,file_extn),paste0(target_path,folder_name,"/",file_name,file_extn))
    }
  }
}
