# Visual-Network 🧠

This project visualizes a custom feedforward artifical neural network training against the MNIST dataset. The network and training algorithms are implemented from scratch (no Tensorflow, PyTorch, etc).

The network is made up of dense layers, each with sigmoid activation. The loss function is mean squared error, and the training algorithm is stochastic gradient descent with no momentum.

## Steps for Using

To run the program, simply clone this reposity and run the following command.
```
$ python src/visual_network.py
```

Make sure you are using Python 2. I have tested the project with Python 2.7.17. Also, make sure to have numpy and pygame installed.<br />
If you wish to run the network slower, you may run the following command. 

```
$ python src/visual_network_slow.py
```

## In the Works

In the future, I would like to change the rendering routines for this project to instead use the manim library written by 3blue1brown. The primary goal of this project is to serve as a resource for education and review. The goal is that if I would like to gain a refresher on the fundamental concepts driving the backpropagation algorithm, I should look no further than this repo.
