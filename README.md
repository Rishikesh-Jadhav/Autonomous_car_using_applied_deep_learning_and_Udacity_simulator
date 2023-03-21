# Autonomous_car_using_applied_deep_learning_and_Udacity_simulator
Created a deep learning model from scratch using the data collected from the simulator and tested the car in new environments

## Deep Learning Model for Autonomous Car Navigation
This project aims to create a deep learning model from scratch using data collected from a car simulator to enable the car to navigate in new environments.

<img align="center" alt="Coding" width="250" src="https://github.com/Rishikesh-Jadhav/Autonomous_car_using_applied_deep_learning_and_Udacity_simulator/blob/main/Screenshot%202023-03-21%20233652.png">

## Requirements
The simulator package is in the folder with the name simulator-windows-64
The Final_Test.ipynb notebook contains the code for the deep learning model creation.
The model.h5 is the saved model.
the drive.py file is used for connecting the simulator and running the model on it. 

To run this project, you need the following:

Python 3.10.  
keras 2.3.  
NumPy.  
Matplotlib.  
UdacityCar simulator with the data collection feature.  

## Steps to run code
Download the model.h5, drive.py files.
Open the terminal, run the drive.py file and open the simulator in either the training or autonomous mode to test the model on the car.

## Data Collection
The car simulator provided a data collection feature that recorded the car's movement along with sensor readings such as speed, steering angle, and camera images. This data was then used to train the deep learning model.

## Model Architecture
The deep learning model used in this project was a nvidea model for self driving cars with convolutional neural network (CNN) that takes the camera images as input and predicts the steering angle. The model consists of multiple convolutional and fully connected layers that learn the features from the images and make predictions based on them.

## Training
To train the model, we used the data collected from the car simulator. The data was split into training and validation sets, and the model was trained using the Adam optimizer and mean squared error loss function. The training process was stopped when the validation loss stops improving, indicating that the model has learned all it can from the data.

## Testing
Once the model was trained, it was tested in new environments to evaluate its performance. The car simulator was used to test the model, and the car's movement was controlled by the model's predictions. The testing process involved measuring the car's performance, such as its ability to stay on the road, avoid obstacles, and navigate through the environment.

## Conclusion
This project demonstrates the creation of a deep learning model from scratch using data collected from a car simulator to enable the car to navigate in new environments. The CNN-based model is trained on the collected data and tested in new environments to evaluate its performance.
