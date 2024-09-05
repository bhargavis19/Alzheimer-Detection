# Brain Diagnosis using fMRI

## Overview

This project leverages Functional Magnetic Resonance Imaging (fMRI) data and machine learning techniques for brain diagnosis, particularly focusing on Alzheimer's disease (AD). We developed and evaluated multiple classifier models to predict dementia stages based on neuroimaging data. Additionally, an image recognition model was built using deep learning techniques to classify MRI images into different stages of dementia.

## Introduction

Alzheimer's disease (AD) is a neurodegenerative disorder that significantly impacts memory and cognitive abilities. It is the leading cause of dementia, especially in older adults. Our project utilizes machine learning techniques on fMRI data to detect Alzheimer's disease and assess its severity. We also employ image recognition methods to classify MRI scans and assist in early-stage diagnosis of dementia.

## Datasets

We used two primary datasets for this project:

### 1. OASIS Dataset:
The **Open Access Series of Imaging Studies (OASIS)** dataset was sourced from Kaggle and includes 150 individuals aged between 60 and 96. It contains a total of 373 T1-weighted MRI scans, grouped into three categories: non-demented, demented, and converted. Each subject has undergone at least two MRI scans, allowing us to track longitudinal changes over time.

- **Features:**
  - Subject ID
  - MRI ID
  - Group (Non-demented, Demented, Converted)
  - Age, Gender, Handedness
  - Education Level (EDUC), Socioeconomic Status (SES)
  - Clinical Dementia Rating (CDR), Mini-Mental State Examination (MMSE)
  - Estimated Total Intracranial Volume (eTIV), Normalized Whole-Brain Volume (nWBV)
  
This dataset was used for training machine learning models to classify dementia based on numerical and categorical data.

### 2. Image-Based Dataset:
The second dataset consists of **MRI images** categorized into four distinct classes:
- **Mild Demented**: 896 images
- **Moderate Demented**: 64 images
- **Non-Demented**: 3200 images
- **Very Mild Demented**: 2240 images

The images were pre-processed and uniformly resized to 128x128 pixels for training the image classification model. These images capture various stages of brain deterioration and are crucial for diagnosing the severity of dementia.

- **Key Highlights:**
  - The dataset contains a total of 6,400 MRI images.
  - The images are organized into separate folders for each class (Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented).
  - Data augmentation techniques such as random flips, rotations, and zooms were applied during training to improve the generalization of the model.
  
This image dataset was used to develop a **deep learning model** for classifying the dementia stage of patients based on visual patterns in MRI scans.

## Methodology

The project is divided into two main components:
1. **Data Preprocessing and Machine Learning**: 
   - Cleaned the OASIS dataset by removing null values and performed exploratory data analysis (EDA).
   - Used several classification models such as Random Forest, SVM, and AdaBoost to predict dementia stages.
   
2. **Image Processing and Deep Learning**:
   - Preprocessed the MRI image dataset by resizing images to 128x128 pixels and performing data augmentation.
   - Developed a **Convolutional Neural Network (CNN)** using TensorFlow to classify images into different dementia stages.
   - The CNN model architecture includes multiple convolution and pooling layers followed by fully connected layers to output the dementia stage.
   
## Results

Here are the results of the classification models for the first dataset (OASIS):

| Model                     | Accuracy (%) |
|----------------------------|--------------|
| Random Forest              | 68.89        |
| Support Vector Machine      | 77.78        |
| Decision Tree              | 60.66        |
| Gradient Boosting          | 72.77        |
| AdaBoost                   | 82.22        |
| K Neighbours               | 62.22        |
| MLP                        | 62.22        |
| Gaussian NB                | 80           |
| Logistic Regression        | 75.5         |

For the second dataset (image-based), the CNN model achieved high accuracy in distinguishing between different stages of dementia, particularly in detecting **Very Mild Demented** and **Non-Demented** cases. The model's performance improved significantly through data augmentation techniques.

## Technologies Used

- **Python**: Programming language
- **TensorFlow**: Deep learning library for image classification
- **Scikit-learn**: Machine learning library for model building and evaluation
- **Matplotlib**: For data visualization
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Kaggle**: Dataset source for MRI data

## Future Scope

In future iterations, we plan to:
- Extend the model to include other imaging modalities such as PET and CT scan data.
- Incorporate real-time data integration for continuous learning and improvement.
- Develop a more comprehensive system for early detection of Alzheimer's, integrating patient symptoms and other clinical factors.
- Implement a cloud-based deployment for real-time MRI analysis to aid in clinical diagnostics.
