# Hello-FOSS-Image-Segmentation
This project is a part of HELLO-FOSS: Celebration of Open Source by the Web and Coding Club. In this project we will be seeing a few applications of machine 
learning in image classification and segmentation. 

## Guidelines
To contribute to these projects you should have  a basic grasp on Python.You can go through [these resources](https://github.com/wncc/TSS-2021/tree/main/Python%20%26%20its%20Applications/Week-1) if you want to revise or learn Python. You should also have a basic understanding of Machine Learning and Convolutional Networks. **NOTE: before sending any pull request, rename your file to include your initials as - filename_RollNum.extension**.

# 1) MRI based Brain Tumor Detection
Brain tumors, are one of the most common and aggressive type of cancers, leading to a very short life expectancy in their highest grade. Misdiagnosis can often prevent effective response against the disease and decrease the chance of survival among patients. One conventional method to identify brain tumors is by inspecting the MRI images of the patient’s brain. For large amount of data and different specific types of brain tumors, this method is time consuming and prone to human errors.

In this challenge we will ask you to build a model to try and identify which of the patients have a brain tumor. We will be using a simple Convolutional Neural Network for the model. 

### Dataset
We will be using the LGG MRI Segmentation dataset. The images used in the dataset were obtained from The Cancer Imaging Archive (TCIA).
They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection.
You can find the dataset [here](https://drive.google.com/drive/u/1/folders/1vdmDnEGVQhu8gnJvYup2-1ajwHujTLrE). As you can see there are a 110 folders each containg the MRI scans for that patient. The brain tumor will be visible in only some of those scans. 
