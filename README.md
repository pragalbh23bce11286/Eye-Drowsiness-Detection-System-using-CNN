# Eye-Drowsiness-Detection-System-using-CNN
Overview:

The Eye Drowsiness Detection System is a Deep Learning project that detects whether a person's eyes are open or closed using a Convolutional Neural Network (CNN).

This system can help prevent accidents caused by driver fatigue by detecting drowsiness in real time.

# Features

Detects open and closed eyes

Deep Learning based CNN model

Image classification

Real-time prediction capability

Training visualization graphs

Automatic model saving

# Technologies Used

Python

TensorFlow / Keras

OpenCV (optional for real-time)

NumPy

Matplotlib

Deep Learning Model

Model Type: Convolutional Neural Network (CNN)

# Architecture:

Conv2D Layer (32 filters)

MaxPooling Layer

Conv2D Layer (64 filters)

MaxPooling Layer

Conv2D Layer (128 filters)

MaxPooling Layer

Fully Connected Layer

Dropout Layer

Output Layer (Sigmoid)

# Dataset Structure
drowsiness_data/

    train/
        open/
        closed/

    test/
        open/
        closed/
        
# How It Works

Step 1: Load dataset

Step 2: Preprocess images

Step 3: Train CNN model

Step 4: Validate model

Step 5: Save trained model

Step 6: Predict eye state

# Output

Example:

Prediction: Eyes Open

or

Prediction: Eyes Closed (Drowsy)
# Training Graphs

The system generates:

Accuracy vs Epoch graph

Loss vs Epoch graph

# Project Structure
Eye-Drowsiness-Detection/

│
├── archive.zip
├── drowsiness_data/
├── eye_drowsiness_model.keras
├── drowsiness_detection.py
├── README.md
└── requirements.txt

# Installation
pip install tensorflow numpy matplotlib
Run the Project
python drowsiness_detection.py

# Applications

Driver drowsiness detection

Accident prevention systems

Automotive safety

AI surveillance systems

Smart vehicle systems

# Model Performance

Typical Accuracy: 90% – 98%

(depending on dataset quality)

# Future Improvements

Real-time detection using webcam

Mobile app integration

Integration with alarm system

Deployment on embedded systems (Raspberry Pi)
