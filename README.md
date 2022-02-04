# ASL_Alphabet_Classification

### Problem Statement  
For a given image of a road, classify whether it contains potholes or not. 

### Data
The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes. Source: https://www.kaggle.com/grassknoted/asl-alphabet

### Approach
preprocessing module from Keras is used to load and scale the images into same size(200,200,3). A VGG16 architecture is used to build the neural network.
![alt text](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fvgg-16-cnn-model%2F&psig=AOvVaw0Xfz6BuXmBZJ0g0HjNrt2c&ust=1644087318863000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCKjDw9jc5vUCFQAAAAAdAAAAABAD)
The model gave 88% accuracy on the test dataset(train-test split = 90:10).
