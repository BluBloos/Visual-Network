# visualnetwork 🧠
This project visualizes a custom feedforward artifical neural network training against the MNIST dataset. The network and training algorithms are implemented from scratch (no tensorflow, pytorch, etc).

## Details
The network is made up of dense layers, each with sigmoid activation. The loss function is mean squared error, and the training algorithm is stochastic gradient descent with no momentum.

## Running the program
To run the program, simply clone this reposity and run the following command.
```
$ python src/visual_network.py
```
Make sure you are using Python 2. I have tested the project with Python 2.7.17. Also, make sure to have numpy and pygame installed.<br />
If you wish to run the network slower, you may run the following command. 
```
$ python src/visual_network_slow.py
```
This is useful to see smaller training details.
