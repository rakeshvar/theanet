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
* theano
	* pip install git+git://github.com/Theano/Theano.git
* sharedmem
	* pip install git+git://github.com/rainwoodman/sharedmem.git

Code
----
* neuralnet.py 
	* Has the main class NeuralNet which is the workhorse
* nn.py
	* Harness the workhorse. 
* prms_*.py
	* Files that contain parameters for the network and training.
	One of them goes as input to nn.py
* *.x.bz2, *.y.bz2
	* x.bz2 contains the raterized images. 
	* Ex.:- numbers.x.bz2 contains numerous rasterized images of numbers 0-9
	* y.bz2 contains the labels of the corresponding images in x.bz2

Testing
-------
1. Edit the prms_*.py to your liking and pass it as input to nn.py
2. Sit back and watch the horse get to work.

