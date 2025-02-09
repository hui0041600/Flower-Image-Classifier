📌 Overview
This repository contains Project 3 for CS672 - Introduction to Deep Learning (Fall 2024, Pace University). The project focuses on image classification using transfer learning, where a pre-trained CNN model is fine-tuned to classify images of flowers.

📊 Project Objective
Build a Deep Learning Image Classifier using CNN-based models.
Use transfer learning to fine-tune a pre-trained model (ResNet, VGG, MobileNet, etc.).
Train the model on the "Flowers Recognition" dataset from Kaggle.
Evaluate model performance using accuracy, precision, recall, and F1-score.
Implement the model using both TensorFlow and PyTorch

📂 Dataset
Flowers Recognition Dataset (5 classes: Daisy, Dandelion, Rose, Sunflower, Tulip)
Dataset: Kaggle Flowers Recognition https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
The dataset is split 75% for training and 25% for testing.

🛠 Tools & Libraries Used
Data Handling & Preprocessing

pandas, numpy – Data processing
matplotlib.pyplot, seaborn – Data visualization
sklearn.preprocessing – Data transformation
Deep Learning Frameworks

TensorFlow/Keras – Pre-trained model loading & fine-tuning
PyTorch – Alternative implementation of transfer learning
Models Used

Pre-trained CNNs: ResNet, VGG, MobileNet, Inception
Optimizers: SGD, Adam
Loss Function: CrossEntropyLoss
