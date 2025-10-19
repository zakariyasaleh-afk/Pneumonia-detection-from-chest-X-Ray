# Pneumonia-detection-from-chest-X-Ray

Chest X-Ray Pneumonia Classification

A convolutional neural network (CNN) that classifies chest X-ray images as Normal or Pneumonia using the Kaggle Chest X-Ray Pneumonia dataset. The project is designed to run in Google Colab with GPU support.

## Dataset
- Source: Kaggle
- Classes: NORMAL, PNEUMONIA
- Images: ~5,800 training, 600+ test

##Data Processing
-	Images are resized to 150x150 pixels
-	Pixel values are normalized by dividing by 255
-	Data augmentation applied to training data:
-	Random rotations (±10°)
-	Width and height shifts (±10%)
-	Zoom (±10%)
-	Horizontal flips
-	Brightness adjustment (±10%)
-	Class imbalance handled using computed class weights

##Model
-	3 convolutional layers with MaxPooling
-	BatchNormalization & SpatialDropout2D to reduce overfitting
-	Dense layer (32 units) + output layer (sigmoid)
-	Binary crossentropy loss, Adam optimizer

##Training
-	Epochs: 10, batch size: 32
-	EarlyStopping (patience=3)
-	ReduceLROnPlateau to adjust learning rate
-	Class weights for imbalanced data

##Results
-	Test Accuracy: 81%
-	Test Loss: 0.53
-	Confusion matrix :
	   <img width="517" height="547" alt="image" src="https://github.com/user-attachments/assets/f35b2bee-a394-4c48-afba-36128cf0dbb8" />
  - Precision/Recall/F1-score:
	   <img width="412" height="133" alt="Screenshot 2025-10-19 215043" src="https://github.com/user-attachments/assets/63919b3c-5e00-4f1b-84ab-b9d90c5d320d" />

	  
    
##Usage
'''python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('my_model.h5')
img_array = image.load_img('example.jpg', target_size=(150,150))
img_array = image.img_to_array(img_array)/255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print("PNEUMONIA" if prediction[0][0]>0.5 else "NORMAL")
