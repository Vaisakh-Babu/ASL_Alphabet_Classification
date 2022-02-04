# ASL_Alphabet_Classification

### Problem Statement  
For a given image of a road, classify whether it contains potholes or not. 

### Data
The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes. Source: https://www.kaggle.com/grassknoted/asl-alphabet

### Approach
preprocessing module from Keras is used to load and scale the images into same size(200,200,3). A VGG16 architecture is used to build the neural network.
![alt text](https://media.geeksforgeeks.org/wp-content/uploads/20200219152327/conv-layers-vgg16.jpg)
The model is able to give 95% accuracy.
