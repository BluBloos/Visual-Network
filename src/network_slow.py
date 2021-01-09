import random
import numpy as np
import threading
import time

# note that when the input z is a vector or numpy array, numpy automatically applies the function sigmoid elementwise, that is, in vectorized form
def sigmoid(z):
    # 1 / 1+e^-z
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # derivate of the sigmoid function
    return sigmoid(z)*(1-sigmoid(z))

class Network(threading.Thread):
    #layers is a list containing the number of
    #neurons in the respective layers
    def __init__(self, layers, training_data, learningRate, test_data, mini_batch_size):
        #threading related things!
        threading.Thread.__init__(self)
        #make the thread daemonic since it may run forever
        self.daemon = True
        self.setDaemon(True)
        self.netLock = threading.Lock()

        self.gotMessage = False
        self.window = None #handle to the neural network graphics module
        self.isTraining = False #boolean whether or not currently training
        self.trainingSpeed = 1.0
        self.miniBatches = None
        self.miniBatchPos = 0

        self.test = False
        self.training_data = training_data
        self.learningRate = learningRate
        self.test_data = test_data
        self.mini_batch_size = mini_batch_size
        self.num_layers = len(layers)
        self.layers = layers
        self.biases = []
        for j in layers[1:]:
            self.biases.append(np.random.randn(j,1))
        self.weights = []
        for k, j in zip(layers[:-1], layers[1:]):
            self.weights.append(np.random.randn(j,k))
        self.activations = []
        for l in layers:
            self.activations.append(np.random.random(l))

    def startTraining(self):
        #here we do a modification of the variables in the neural network
        #so there needs to be some sort of unlocking
        self.isTraining = True
        self.miniBatchPos = 0
        self.miniBatches = None

    def startTest(self):
        self.test = True

    def MakeRect(self, maxWidth, maxHeight, anchorX, anchorY):
        minX = int(maxWidth * anchorX[0])
        width = min(maxWidth, int(maxWidth - maxWidth * (1.0 - anchorX[1]) )) - minX
        minY = int(maxHeight * anchorY[0])
        height = min(maxHeight, int(maxHeight - maxHeight * (1.0 - anchorY[1]) )) - minY
        return (minX, minY, width, height)

    def continueTraining(self):
        self.isTraining = True

    def pauseTraining(self):
        print("pause")
        self.isTraining = False

    def run(self):
        while(1):
            if self.window: #make sure we have a window

                #check if we should make a new set of miniBatches
                self.window.gfxLock.acquire()
                try:
                    if self.isTraining and not self.miniBatches: #do we have a mini batch?
                        #if not, make a some mini_batches
                        n = len(self.training_data)
                        random.shuffle(self.training_data)
                        self.miniBatches = []
                        for k in xrange(0, n, self.mini_batch_size):
                            self.miniBatches.append(self.training_data[k:k+self.mini_batch_size])
                    elif not self.isTraining:
                        if self.test:
                            print "Accuracy: {0} / {1}".format(self.evaluate(self.test_data), len(self.test_data))
                            self.test = False
                finally:
                    self.window.gfxLock.release()

                default = False
                self.window.gfxLock.acquire()
                if not self.miniBatches:
                    #if there are no miniBatches then I can assume we aren't training
                    #we are going to have to draw something otherwise numpy will get mad
                    default = True
                self.window.gfxLock.release()

                if default:
                    while(1):
                        self.window.gfxLock.acquire()
                        if self.window.doingGraphics:
                            self.window.drawSystem()
                            self.window.graphicsPoll = True #just did graphics
                            self.window.gfxLock.release()
                            break
                        self.window.gfxLock.release()
                    continue

                for mini_batch in self.miniBatches[self.miniBatchPos:]:
                    #check to make sure we aren't paused or anything
                    self.window.gfxLock.acquire()
                    if not self.isTraining:
                        self.window.gfxLock.release()
                        break
                    self.window.gfxLock.release()

                    self.update_mini_batch(mini_batch, self.learningRate)
                    self.miniBatchPos += 1

                self.window.gfxLock.acquire()
                if not self.isTraining or self.miniBatchPos == len(self.miniBatches):
                    default = True
                    self.isTraining = False
                self.window.gfxLock.release()

                if default:
                    while(1):
                        self.window.gfxLock.acquire()
                        if self.window.doingGraphics:
                            self.window.drawSystem()
                            self.window.graphicsPoll = True #just did graphics
                            self.window.gfxLock.release()
                            break
                        self.window.gfxLock.release()

    #a'=sigmoid(wa+b)
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, learningRate, test_data=None):
        #NOTE(Noah): What's the differnce between xrange and range?
        #I don't really care about the specific details, because I don't use
        #python too rigoursly, but basically xrange is better for large things,
        #and it won't eat your memory
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = []
            for k in xrange(0, n, mini_batch_size):
                mini_batches.append(training_data[k:k+mini_batch_size])
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learningRate)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, learningRate):
        #NOTE(Noah): numpy.zeros(shape)
        #shape: int or tuple of ints
        nabla_b = []
        for b in self.biases:
            nabla_b.append(np.zeros(b.shape))
        nabla_w = []
        for w in self.weights:
            nabla_w.append(np.zeros(w.shape))
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop2(x, y)
            #sum the deltas
            old_nabla_b = nabla_b
            nabla_b = []
            for nb, dnb in zip(old_nabla_b, delta_nabla_b):
                nabla_b.append(nb + dnb)
            old_nabla_w = nabla_w
            nabla_w = []
            for nw, dnw in zip(old_nabla_w, delta_nabla_w):
                nabla_w.append(nw + dnw)
        #nudge all the weights and all the biases
        self.window.gfxLock.acquire()
        try:
            #self.ovveride = False
            self.weights = [w-(learningRate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(learningRate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        finally:
            self.window.gfxLock.release()

    def callback(self):
        self.gotMessage = True

    def speedUp(self):
        self.window.gfxLock.acquire()
        self.trainingSpeed += 0.10
        self.window.gfxLock.release()

    def speedDown(self):
        self.window.gfxLock.acquire()
        self.trainingSpeed = max(self.trainingSpeed - 0.10, 0.1)
        self.window.gfxLock.release()

    def backprop2(self, x, y):
        #gradient = Gradient(self.layers)
        nabla_b = []
        for b in self.biases:
            nabla_b.append(np.zeros(b.shape))
        nabla_w = []
        for w in self.weights:
            nabla_w.append(np.zeros(w.shape))

        #next we build the network
        activation = x
        Z = []

        self.window.gfxLock.acquire()
        try:
            self.activations = [x]
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation) + b
                Z.append(z)
                activation = sigmoid(z)
                self.activations.append(activation)
            self.window.ovveride = True
        finally:
            self.window.gfxLock.release()

        #when we calculate each activation we need to do some sort of animation
        #glowWeights(self, l, weights, callback, duration)

        for l in range(1, len(self.activations) - 1): #here we exclude the first layer
            #construct the weight request
            for j in range(len(self.activations[l+1])):
                weights = []
                for k in range(len(self.activations[l])):
                    weights.append((j,k))

                timeElapsed = 0.0
                initialTime = time.time()
                displayTime = self.trainingSpeed**-1 * 0.1
                while(timeElapsed < displayTime):
                    while(1):
                        self.window.gfxLock.acquire()
                        if self.window.doingGraphics:
                            self.window.glowWeights(l, weights)
                            self.window.drawMatrixRowHighlight(self.weights[l], self.MakeRect(self.window.width, self.window.height, (0.3, 0.9), (0.52, 1.0)), j)
                            self.window.drawVector(self.activations[l], self.MakeRect(self.window.width, self.window.height, (0.95, 1.0), (0.52, 1.0)))

                            self.window.graphicsPoll = True #we are no longer drawing
                            self.window.gfxLock.release()
                            break
                        self.window.gfxLock.release()
                    timeElapsed = time.time() - initialTime
                    #timeElapsed = displayTime


        #then we compute the error of the last layer
        error = self.cost_derivative(self.activations[-1], y) * sigmoid_prime(Z[-1])
        #set the gradients for last layer
        nabla_b[-1] = error
        nabla_w[-1] = np.dot(error, self.activations[-2].transpose())
        #now we do a backwards pass
        for l in xrange(2, self.num_layers):
            error = np.dot(self.weights[-l+1].transpose(), error) * sigmoid_prime(Z[-l])
            nabla_b[-l] = error
            nabla_w[-l] = np.dot(error, self.activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
'''
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
network = Network([784,16,16,10])
network.sgd(training_data, 30,10,3.0,test_data)
'''
