# Image Classifier
Image classifier using PyTorch and pre-trained neural networks

Most of us are using image classification algorithms in our daily activities. Some of the applications include:
1. Photo organization apps in mobile phones,
2. Stock Photography and Video Websites,
3. Visual search API which helps people to search a similar product using a reference image they clicked on the phone, and
4. Image recognition on social media

Image classification is the primary domain, in which deep neural networks play the most important role in image analysis. Here the computer is fed with data in form of images, the computer tries to understand the relationship between the data and the classes into which they are classified which we call the training process and then use the same understanding to classify an image which the computer hasn't seen before which we call the testing process or inference.

In this project, we will be using a supervised classification algorithm where we define the classes that the data are classified into and provide the training data of each defined class. Here we will be creating a neural network model that can recognize different species of flowers. We have a dataset of 102 flower categories. We will also cover how we can use ```torchvision``` package to load pre-trained models and carry out model inference to classify an image.

At the end of the project, we will have a command-line application that can be applied on any user-specified dataset to train, the user can choose between resnet18, alexnet, vgg16, and densenet121 models and can also provide test image path for inference.

This is the final Project of the Udacity AI with Python Nanodegree


# Data and label
I could not attach the data in the repo because of size constraint but this shouldn't stop you running it on your own dataset. Please make sure of the following points:
1. Create three folders: test, train, and valid
2. Create subfolders with the category number as folder name
For eg: if you have the images of lily and the category assigned to it is 5, folder structure would be
* \flowers\train\5\image.jpg,  
* \flowers\test\5\image.jpg,  
* \flowers\valid\5\image.jpg  
Try to provide as many images as u can (recommended more than 20) for each category. It would be even better if these images are in different angles and lighting which will help the model to generalize. Also, maintain the labels in the cat_to_name.json file.

# Library dependency 
Numpy, Pandas, PyTorch, PIL, json, re, and matplotlib. You can install them using pip
```
pip install numpy pandas matplotlib pil
```
or 
```
conda install numpy pandas matplotlib pil
```
Follow [PyTorch](https://pytorch.org/get-started/locally/) to get updated pip command
* Checkout [torchvision](https://pytorch.org/docs/stable/torchvision/models.html) for details on pre-trained model

# Command Line Application
Train a new network on a data set with ```train.py```

* Basic Usage : ```python train.py data_directory```
* Prints out current epoch, training loss, validation loss, and validation accuracy as the network trains
* Options:
  * Set directory to save checkpoints: python train.py data_dor ```--save_dir save_directory```
  * Choose arcitecture (resnet, alexnet, densenet121 or vgg16 available): ```pytnon train.py data_dir --arch "vgg16"```
  * Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_layer1 120 --epochs 20```
  * Use GPU for training: ```python train.py data_dir --gpu```
  
Predict flower name from an image with predict.py along with the probability of that name. That is you'll pass in a single image ```/path/to/image``` and return the flower name and class probability

* Basic usage: python predict.py /path/to/image checkpoint
* Options:
  * Return top K most likely classes: ```python predict.py input checkpoint ---top_k 3```
  * Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
  * Use GPU for inference: ```python predict.py input checkpoint --gpu```

# GPU
As the network makes use of a sophisticated deep convolutional neural network the training process is impossible to be done by a common laptop. In order to train your models to your local machine, you have three options

1. Cuda -- If you have an NVIDIA GPU then you can install CUDA from here. With Cuda you will be able to train your model however the process will still be time consuming
2. Cloud Services -- There are many paid cloud services that let you train your models like AWS or Google Cloud
3. Google Colab -- Google Colab gives you free access to a tesla K80 GPU for 12 hours at a time. Once 12 hours have elapsed you can just reload and continue! The only limitation is that you have to upload the data to Google Drive and if the dataset is massive you may run out of space.
However, once a model is trained then a normal CPU can be used for the predict.py file and you will have an answer within some seconds.

# Hyperparameters
The network consists of only 1 hidden layer, uses ReLU activation function, LogSoftmax function for the output layer and LogLoss for the loss function
The model has the following default hypermeter values:
1. Model - densenet121
2. Learning rate - 0.003
3. Hidden nodes - 500
4. Epocs - 1
5. Top probabilities to show = 5  
Please feel free to changes the values as needed

# Accuracy
With the default settings of hyperparameters, my model accuracy is 89% on the validation set

# Output
Please provide the path from the test dataset to do inference. The model will return the top 5 most probable answers in the descending order.

# Checkpoint
I could not attach a copy of checkpoint.pth because of space constraint but the logic to save your trained model is provided.
