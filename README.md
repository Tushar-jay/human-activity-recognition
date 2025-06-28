# human-activity-recognition
feature classification on human activity recognition 
 Human Activity Recognition using WISDM Dataset

This project implements a machine learning pipeline to classify human physical activities like walking, jogging, sitting, standing, upstairs, and downstairs using accelerometer sensor data from smartphones.

 Dataset

- Name: WISDM v1.1 Dataset  
- Source: [WISDM Lab - Fordham University](https://www.cis.fordham.edu/wisdm/dataset.php)  
- Sensor: Smartphone accelerometer  
- Sampling Rate:20 Hz  
- Activities:
  - Walking  
  - Jogging  
  - Sitting  
  - Standing  
  - Upstairs  
  - Downstairs


Features

- Preprocessing using sliding window segmentation
- Statistical feature extraction: mean, std, min, max, etc.
- Label encoding and feature scaling
- Trained on multiple models:
  - Random Forest
  - SVM
  - K-Nearest Neighbors
  - Logistic Regression
  - Naive Bayes
  - MLP (Neural Network)
- Accuracy comparison and result visualization using Seaborn
- Models saved using `joblib`
- Ready for deployment or real-time prediction


