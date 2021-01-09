import sys, pygame
import threading
import math
import time

def FloatVecToIntVec(vec):
    newVec = []
    for element in vec:
        newVec.append(int(element))
    return newVec

def MakeRect(maxWidth, maxHeight, anchorX, anchorY):
    minX = int(maxWidth * anchorX[0])
    width = min(maxWidth, int(maxWidth - maxWidth * (1.0 - anchorX[1]) )) - minX
    minY = int(maxHeight * anchorY[0])
    height = min(maxHeight, int(maxHeight - maxHeight * (1.0 - anchorY[1]) )) - minY
    return (minX, minY, width, height)

#a and b must be vectors of the same length
def biLinearInterpolation(a, b, percent):
    newVec = []
    for i in range(len(a)):
        newVec.append(a[i] * (1.0-percent) + b[i] * percent)
    return newVec

class Input:
    def __init__(self):
        self.A = False
        self.W = False
        self.S = False
        self.D = False
        self.mouse = [False] * 3
    def update(self, event):
        set = False
        if event.type == pygame.KEYDOWN:
            set = True
        if event.key == pygame.K_a:
            self.A = set
        elif event.key == pygame.K_w:
            self.W = set
        elif event.key == pygame.K_s:
            self.S = set
        elif event.key == pygame.K_d:
            self.D = set

class Button():
    def __init__(self, rect, text, font, action, color=(255,255,255)):
        self.rect = rect
        self.text = text
        self.action = action
        self.color = color
        self.font = font

    def isMouseBounded(self, mousePos):
        if mousePos[0] >= self.rect[0] and mousePos[0] <= self.rect[0] + self.rect[2] and mousePos[1] >= self.rect[1] and mousePos[1] <= self.rect[1] + self.rect[3]:
            return True
        return False

    def drawAndUpdate(self, screen, mousePos, mouseState):
        color = self.color
        if self.isMouseBounded(mousePos):
            #dim the color
            color = (int(color[0] * 0.5),int(color[1] * 0.5),int(color[2] * 0.5))
            if mouseState[0] == True:
                self.action()
        pygame.draw.rect(screen, color, self.rect)
        #draw text ontop
        textSurface = self.font.render(self.text, True, (0,0,0))
        screen.blit(textSurface, (self.rect[0], self.rect[1]))

class gfxNetwork:
    def __init__(self, width, height, network, nodeColorSpace, weightColorSpace, inputSpec, outputSpec, sigmoid, caption="Pygame"):
        #threading.Thread.__init__(self)
        #make the thread daemonic since it may run forever
        #self.daemon = True
        #self.setDaemon(True)
        #initialize mutex lock
        self.gfxLock = threading.Lock()
        pygame.init()

        # --------- bus -----------
        self.glowingWeights = False
        self.glowWeightsParams = ()
        self.timeElapsed = 0.0
        self.timerStart = 0.0
        self.ovveride = False
        # -------------------------

        self.network = network
        self.sigmoid = sigmoid
        self.nodePos = []
        self.nodeSize = []
        self.drawingConfig = (inputSpec, outputSpec)
        self.nodeColorSpace = nodeColorSpace
        self.weightColorSpace = weightColorSpace
        self.binaryColorSpace = ((1.0,1.0,1.0),(255.0,255.0,255.0))

        self.width = width
        self.height = height
        self.buttons = []
        self.inputState = Input()
        self.graphicsPoll = False
        self.doingGraphics = False

        self.fontHeight = 16
        self.font = pygame.font.Font(pygame.font.get_default_font(), self.fontHeight)

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(caption)
        #self.clock = pygame.time.Clock()

    def updateNetwork(self, network):
        self.network = network

    def drawGrid(self, surface, pos, gridlength, tileSize, colorSpace, samples):
        nodePos = []
        xPos = pos[0]
        yPos = pos[1]

        for y in range(gridlength):
            xPos = pos[0]
            for x in range(gridlength):
                color = biLinearInterpolation(colorSpace[0], colorSpace[1], samples[y * gridlength + x])
                color = (int(color[0]), int(color[1]), int(color[2]))
                pygame.draw.rect(surface, color, (xPos, yPos, tileSize, tileSize))

                nodePos.append((float(xPos + tileSize / 2.0), float(yPos + tileSize / 2.0)))

                xPos += tileSize
            yPos += tileSize

        return nodePos

    def drawNodes(self, surface, pos, circleRadius, colorSpace, samples):
        nodePos = []
        xPos = pos[0] + circleRadius
        yPos = pos[1] + circleRadius

        for sample in samples:
            color = biLinearInterpolation(colorSpace[0], colorSpace[1], sample)
            color = (int(color[0]), int(color[1]), int(color[2]))
            pygame.draw.circle(surface, color, (xPos, yPos), circleRadius)
            nodePos.append((float(xPos), float(yPos)))
            yPos += circleRadius * 2

        return nodePos

    def buildNetwork(self, inputSpec, outputSpec):
        inputRect = MakeRect(inputSpec[1][0], inputSpec[1][1], inputSpec[1][2], inputSpec[1][3])
        outputRect = MakeRect(outputSpec[1][0], outputSpec[1][1], outputSpec[1][2], outputSpec[1][3])
        self.nodePos = []
        self.nodeSize = []

        nodeSurface = pygame.Surface((self.width, self.height))
        nodeSurface.set_colorkey((0,0,0))
        #draw inputLayer
        if inputSpec[0] == "tiles":
            samples = self.network.activations[0].flatten().tolist()
            gridLength = int(math.floor(len(samples)**(1.0/2.0)))
            tileSize = int(min(inputRect[2] - inputRect[0], inputRect[3] - inputRect[1]) / gridLength)
            self.nodePos.append(self.drawGrid(nodeSurface, inputRect[:2], gridLength, tileSize, self.nodeColorSpace, samples))
            self.nodeSize.append(tileSize)
        elif inputSpec[0] == "nodes":
            samples = self.network.activations[0].flatten().tolist()
            circleRadius = int((inputRect[3] - inputRect[1]) / (len(samples) * 2))
            self.nodePos.append(self.drawNodes(nodeSurface, inputRect[:2], circleRadius, self.nodeColorSpace, samples))
            self.nodeSize.append(circleRadius)
        #draw hiddenlayers!
        hiddenLayerCount = float(len(self.network.activations[1:-1]))
        percentDelta = (outputSpec[1][2][0] - inputSpec[1][2][1]) / max(hiddenLayerCount, 0.01)
        currentPercentOffset = 0.0
        for samples in self.network.activations[1:-1]:
            space = MakeRect(self.width, self.height, (inputSpec[1][2][1] + currentPercentOffset, inputSpec[1][2][1] + currentPercentOffset * 2.0), (0.0,0.5))
            circleRadius = int((space[3] - space[1]) / (len(samples) * 2))
            self.nodePos.append(self.drawNodes(nodeSurface, space[:2], circleRadius, self.nodeColorSpace, samples))
            self.nodeSize.append(circleRadius)
            currentPercentOffset += percentDelta

        #draw output layer
        if outputSpec[0] == "tiles":
            samples = self.network.activations[-1].flatten().tolist()
            gridLength = int(math.floor(len(samples)**(1.0/2.0)))
            tileSize = int(min(outputRect[2] - outputRect[0], outputRect[3] - outputRect[1]) / gridLength)
            self.nodePos.append(self.drawGrid(nodeSurface, outputRect[:2], gridLength, tileSize, self.nodeColorSpace, samples))
            self.nodeSize.append(tileSize)
        if outputSpec[0] == "nodes":
            samples = self.network.activations[-1].flatten().tolist()
            circleRadius = int((outputRect[3] - outputRect[1]) / (len(samples) * 2))
            self.nodePos.append(self.drawNodes(nodeSurface, outputRect[:2], circleRadius, self.nodeColorSpace, samples))
            self.nodeSize.append(circleRadius)
        return nodeSurface

    def drawNetwork(self):
        nodeSurface = self.buildNetwork(self.drawingConfig[0], self.drawingConfig[1])
        self.drawWeights(self.screen, self.weightColorSpace)
        self.screen.blit(nodeSurface, (0,0))

    #draw weights and biases?
    #nodePos is a multidimensional array
    #nodePos = [ [(0.2,0.3),...],[(0.1,0.0)....]...  ]
    def drawWeights(self, surface, colorSpace):
        #draw weights from last hidden layer to output layer
        for l in range(1,len(self.nodePos) - 1):
            matrix = self.network.weights[l].tolist()
            for k in range(len(self.nodePos[l])):
                for j in range(len(self.nodePos[l+1])):
                    w = matrix[j][k]
                    color = biLinearInterpolation(colorSpace[0], colorSpace[1], self.sigmoid(w))
                    color = (int(color[0]), int(color[1]), int(color[2]))
                    pygame.draw.line(surface, color, self.nodePos[l][k], self.nodePos[l+1][j], 1)

    #weights is a list of tuples (j,k)
    def glowWeights(self, l, weights):
        nodeSurface = self.buildNetwork(self.drawingConfig[0], self.drawingConfig[1])
        matrix = self.network.weights[l].tolist()
        for item in weights:
            w = matrix[item[0]][item[1]]
            color = biLinearInterpolation(self.weightColorSpace[0], self.weightColorSpace[1], self.sigmoid(w))
            color = (int(color[0]), int(color[1]), int(color[2]))
            pygame.draw.line(self.screen, color, self.nodePos[l][item[1]], self.nodePos[l+1][item[0]], 1)
            pygame.draw.circle(self.screen, (255,0,0), FloatVecToIntVec(self.nodePos[l][item[1]]), int(self.nodeSize[l]) + 2, 4)
        #pygame.draw.circle(self.screen, (255,0,0), FloatVecToIntVec(self.nodePos[l+1][weights[0][0]]), int(self.nodeSize[l]) + 2, 4)
        self.screen.blit(nodeSurface, (0,0))

    #this gets called from your boy
    def drawMatrixRowHighlight(self, matrix, space, row):
        padding = 1
        length = self.drawWeightMatrix(matrix, space)
        pygame.draw.rect(self.screen, (255,0,0), (space[0] - padding, space[1] - padding + row * (self.fontHeight + 2), length + padding, self.fontHeight + padding), 2)

    #this function takes a numpy matrix
    def drawWeightMatrix(self, matrix, space):
        xPos = space[0] + 8
        yPos = space[1]
        longestLine = 0

        lines = []
        matrix = matrix.tolist()
        for y in range(len(matrix)):
            line = ""
            for x in range(len(matrix[0])):
                line += str(round(matrix[y][x], 2)) + "  "
            lines.append(line)

        for line in lines:
            textSurface = self.font.render(line, True, (255,255,255))
            self.screen.blit(textSurface, (xPos, yPos))
            yPos += self.fontHeight + 2
            if textSurface.get_width() > longestLine:
                longestLine = textSurface.get_width()

        #draw the two lines
        pygame.draw.rect(self.screen, (255,255,255), (xPos - 8, space[1], 5, yPos - space[1]))
        pygame.draw.rect(self.screen, (255,255,255), (xPos + longestLine, space[1], 5, yPos - space[1]))
        #draw the tips of the left bois
        pygame.draw.rect(self.screen, (255,255,255), (xPos - 8, space[1] - 5, 20, 5))
        pygame.draw.rect(self.screen, (255,255,255), (xPos - 8, yPos, 20, 5))
        #draw the tips of the right bois
        pygame.draw.rect(self.screen, (255,255,255), (xPos + longestLine - 20, space[1] - 5, 25, 5))
        pygame.draw.rect(self.screen, (255,255,255), (xPos + longestLine - 20, yPos, 25, 5))

        return longestLine

    def drawVector(self, vector, space):
        xPos = space[0] + 8
        yPos = space[1]
        longestLine = 0

        lines = []
        #vector = vector.tolist()
        for element in vector:
            lines.append(str(round(element, 2)))

        for line in lines:
            textSurface = self.font.render(line, True, (255,255,255))
            self.screen.blit(textSurface, (xPos, yPos))
            yPos += self.fontHeight + 2
            if textSurface.get_width() > longestLine:
                longestLine = textSurface.get_width()

        #draw the two lines
        pygame.draw.rect(self.screen, (255,255,255), (xPos - 8, space[1], 5, yPos - space[1]))
        pygame.draw.rect(self.screen, (255,255,255), (xPos + longestLine, space[1], 5, yPos - space[1]))
        #draw the tips of the left bois
        pygame.draw.rect(self.screen, (255,255,255), (xPos - 8, space[1] - 5, 20, 5))
        pygame.draw.rect(self.screen, (255,255,255), (xPos - 8, yPos, 20, 5))
        #draw the tips of the right bois
        pygame.draw.rect(self.screen, (255,255,255), (xPos + longestLine - 20, space[1] - 5, 25, 5))
        pygame.draw.rect(self.screen, (255,255,255), (xPos + longestLine - 20, yPos, 25, 5))

    def drawSystem(self):
        self.drawNetwork()
        self.drawWeightMatrix(self.network.weights[1], MakeRect(self.width, self.height, (0.3, 0.9), (0.52, 1.0)))
        self.drawVector(self.network.activations[1], MakeRect(self.width, self.height, (0.95, 1.0), (0.52, 1.0)))

    def clearScreen(self, color):
        pygame.draw.rect(self.screen, color, (0,0,self.width, self.height))

    def attachButton(self, rect, action, text="Button"):
        self.gfxLock.acquire()
        try:
            self.buttons.append(Button(rect, text, self.font, action))
        finally:
            self.gfxLock.release()

    #graphics loop
    def gfxLoop(self):
        running = True
        while(running):
            self.gfxLock.acquire()
            try:
                #process input
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYUP or event.type == pygame.KEYDOWN:
                        self.inputState.update(event)
                    if event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            self.inputState.mouse[0] = True
                        elif event.button == 2:
                            self.inputState.mouse[1] = True
                        elif event.button == 3:
                            self.inputState.mouse[2] = True

                self.clearScreen((0,0,0))
            finally:
                self.gfxLock.release()

            self.doingGraphics = True
            while(not self.graphicsPoll): #here we wait for graphics
                pass
            self.doingGraphics = False
            self.graphicsPoll = False
            #process buttons
            for button in self.buttons:
                button.drawAndUpdate(self.screen, pygame.mouse.get_pos(), self.inputState.mouse)

            self.gfxLock.acquire()
            try:
                #draw trainingSpeed
                textSurface = self.font.render(str(round(self.network.trainingSpeed,2)) + "%", True, (255,255,255))
                self.screen.blit(textSurface, (0, 0))

                #reset buttons so that the release of the mouse has a clicking effect
                self.inputState.mouse[0] = False
                self.inputState.mouse[1] = False
                self.inputState.mouse[2] = False

                pygame.display.update()
                #self.graphicsOpen = False
            finally:
                self.gfxLock.release()


        pygame.quit()

#window = gfxWindow(640, 480)
#window.gfxLoop()

#we can use thread.join() to wait for it to exit
# window.join()

#threading libary has following objects for synchronizing threads
'''
lock
rlock
semaphore
boundedsemaphore
event
condition
'''
#no acquiring multiple locks at a time!
