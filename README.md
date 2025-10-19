# Pneumonia-detection-from-chest-X-Ray

Chest X-Ray Pneumonia Classification

A convolutional neural network (CNN) that classifies chest X-ray images as Normal or Pneumonia using the Kaggle Chest X-Ray Pneumonia dataset. The project is designed to run in Google Colab with GPU support.

Dataset
	•	Source: Kaggle
	•	Classes: NORMAL, PNEUMONIA
	•	Images: ~5,800 training, 600+ test

Data Processing
	•	Images are resized to 150x150 pixels
	•	Pixel values are normalized by dividing by 255
	•	Data augmentation applied to training data:
	•	Random rotations (±10°)
	•	Width and height shifts (±10%)
	•	Zoom (±10%)
	•	Horizontal flips
	•	Brightness adjustment (±10%)
	•	Class imbalance handled using computed class weights

Model
	•	3 convolutional layers with MaxPooling
	•	BatchNormalization & SpatialDropout2D to reduce overfitting
	•	Dense layer (32 units) + output layer (sigmoid)
	•	Binary crossentropy loss, Adam optimizer

Training
	•	Epochs: 10, batch size: 32
	•	EarlyStopping (patience=3)
	•	ReduceLROnPlateau to adjust learning rate
	•	Class weights for imbalanced data

Results
	•	Test Accuracy: 81%
	•	Test Loss: 0.53
	•	Confusion matrix 
    

Usage
