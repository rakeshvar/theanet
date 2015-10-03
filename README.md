theanet
=======

Overview
--------
Uses Theano to build a convolutional neural network with features like 

1. Elastic Distortion and noising of inputs
2. Convolutional Layers
3. Hidden Layers with Dropout, Maxnorm Regularization
4. Various kinds of output layers like Softmax, Mixture of Gaussians etc. 

Dependencies
------------
* python 3.0
* theano
	* pip install git+git://github.com/Theano/Theano.git

Example Usage
-------------
1. Make a file like params/5numbers.prms to have a dictionary that defines 
the CNN you want to build.
2. Run it using train.py

```sh
python3 train.py mnist params/mnist.prms
```

Your own data
-------------
To make it work for your own data (call it ```galaxy``` data). Create a 
module in the data directory (like ```mnist.py```), called ```galaxy.py```, 
which will have the following four attributes, when loaded. 
 ```training_x, training_y, testing_x, testing_y```. The images should be a 
 stack of 2D or 3D numpy arrays. Also specify a CNN like ```galaxy.prms```. 
 Then you can use theanet as:

```sh
python3 train.py galaxy galaxy.prms
```

Code
----
* train.py
	* Harnesses the workhorse. Used to train. 
* theanet/neuralnet.py 
	* Has the main class NeuralNet which is the workhorse
* params/<name>.prms
	* Files that contain parameters for the network and training.
	One of them goes as input to train.py
* data/mnist.py
    Loads the MNIST dataset.