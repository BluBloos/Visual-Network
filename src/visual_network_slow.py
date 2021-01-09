import gfx
import random
import network_slow

def buttonAction():
    print("test")

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network_slow.Network([784,16,16,10], training_data, 3.0, test_data, 10)

nodesColorSpace = ((1.0,1.0,1.0),(255.0,255.0,255.0))
weightsColorSpace = ((255.0,0.0,0.0),(0.0,255.0,0.0))
inputRect = (1280, 720, (0,0.5), (0,0.5))
outputRect = (1280, 720, (0.8,1.0), (0,0.5))

window = gfx.gfxNetwork(1280, 720, net, nodesColorSpace, weightsColorSpace, ("tiles", inputRect), ("nodes", outputRect), network_slow.sigmoid, caption="Network Visualization")

net.window = window
net.start()

#net.startTraining()
#window.clearScreen((255, 255, 255))
#window.buildNetwork()
#then we make a network
#and sgd now becomes a thread

window.attachButton(gfx.MakeRect(window.width, window.height, (0.0, 0.2), (0.5, 0.6)), net.startTraining, "train")
window.attachButton(gfx.MakeRect(window.width, window.height, (0.0, 0.2), (0.65, 0.75)), net.pauseTraining, "pause")
window.attachButton(gfx.MakeRect(window.width, window.height, (0.0, 0.2), (0.8, 0.90)), net.continueTraining, "continue")
window.attachButton(gfx.MakeRect(window.width, window.height, (0.2, 0.4), (0.95, 1.00)), net.startTest, "test")

#trainingSpeed buttons
window.attachButton(gfx.MakeRect(window.width, window.height, (0.0, 0.2), (0.90, 0.95)), net.speedUp, "speed up")
window.attachButton(gfx.MakeRect(window.width, window.height, (0.0, 0.2), (0.95, 1.0)), net.speedDown, "speed down")



#window.glowWeights(1, ( (0,1),(0,3),(1,4),(1,3),(2,0) ), buttonAction, 2.0)

window.gfxLoop()
