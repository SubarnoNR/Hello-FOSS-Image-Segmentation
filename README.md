# Hello-FOSS-d(AI)gnose
This project is a part of HELLO-FOSS: Celebration of Open Source by the Web and Coding Club. In this project we will be working on classifying and identifying brain tumors from MRI scans. 

## Guidelines
To contribute to these projects you should have  a basic grasp on Python.You can go through [these resources](https://github.com/wncc/TSS-2021/tree/main/Python%20%26%20its%20Applications/Week-1) if you want to revise or learn Python. You should also have a basic understanding of Machine Learning techniques and Convolutional Networks. **NOTE: before sending any pull request, rename your file to include your initials as - filename_RollNum.extension**.

# 1) MRI based Brain Tumor Detection
Brain tumors, are one of the most common and aggressive type of cancers, leading to a very short life expectancy in their highest grade. Misdiagnosis can often prevent effective response against the disease and decrease the chance of survival among patients. One conventional method to identify brain tumors is by inspecting the MRI images of the patient’s brain. For large amount of data and different specific types of brain tumors, this method is time consuming and prone to human errors.

In this challenge we will ask you to build a model to try and identify which of the patients have a brain tumor. We will be using a simple Convolutional Neural Network for the model. 

# 2) Brain Tumor Segmentation 
After classifiying that whether the patient has a brain tumor or what kind of brain tumor does he have, the next step is to identify the brain tumor from the MRI scan. Now this becomes a semantic image segmentation problem. Image segmentation is a computer vision task in which we label specific regions of an image according to what's being shown. To solve this we will move on from CNNs to FCNs(Fully Convolutional Networks).

A fully convolution network is a neural network that only performs convolution i.e. an FCN is a CNN without having fully connected layers. The main benefit of using FCNs is that it allows us to associate a label with each pixel of the image since we have not lost any kind of spatial information in fully connceted layers. You can look at [this](https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1) article and video to know more about usin FCNs for image sgmentation.

### Dataset
We will be using the LGG MRI Segmentation dataset. The images used in the dataset were obtained from The Cancer Imaging Archive (TCIA).
They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection.
You can find the dataset [here](https://github.com/SubarnoNR/Hello-FOSS-Image-Segmentation/tree/main/Dataset). As you can see there are a 110 folders each containg the MRI scans for that patient. Each patient has a number of MRI scans for different positions and slices. 

### The Tasks
3) **The testing metrics**: Right now we are only using binary accuracy as a metric to evaluate the performance of the model which can be very misleading. There are a number of other metrics such as IOU, precision, recall or the confusion matrix which are widely used to judge how are model is doing. Add these metrics in the model as well. You can read about these metrics [here](https://www.kite.com/blog/python/image-segmentation-tutorial/#validation2) if you don't know about them.

1) **Imporoving model architecture and hyperparameter tuning**: A basic model implemented using this dataset can be found [here](https://www.kaggle.com/yassinealouini/all-the-segmentation-metrics). As you can see even though the binary accuracy looks high but the low iou clearly tells that the model is not at all optimum. Thus add/modify the layers, adjust the hyperparameters and try to improve the model for better IOU. 

2) **Data Augmentation**: The dataset can also be modified and improved to better train the model.  Use data augmentation techniques, rotate, translate and warp images to improve the diversity of images being used for training. This will help solve any overfitting issues and help boost up accuracy. You can go through [this](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/) to learn about data augmentation. The techniques discussed here are a bit more general but in this project we have used Image Data Generator so you can modify that itself to do data augmentation.

4) **Implementing UNet**: UNet is one of the most popular and effective architechtures used for semantic segmentation especially for bio medical tasks. You can read and learn about it [here](https://blog.paperspace.com/unet-architecture-image-segmentation/). Try and implement this architecture for this problem.
