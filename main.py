"""import random
from abc import ABC, abstractmethod

def normalize(x,minimum,maximum):
  return (float(x)-minimum)/(maximum-minimum)

#I moved the outside info into the array class, which I renamed Domain
#Domain is now an abstract class, meaning it has to be inherited by a subclass
#This way we can make three different domains instances
class Domain(ABC): 

  def __init__(self, test=None):
    self.test = test
    self.arrayOfTest = []
    self.arrayOfTraining = []
    self.numOfCorrect = 0
    self.totalCount = 0

  @abstractmethod
  def loadFile(self, fileName):
    lines = [line.rstrip('\n') for line in open(fileName)]
    return lines
      
  @abstractmethod
  def createNode(self, line):
    attributes = line.split(',')
    return attributes
    #PRINT THE TEXT COMING IN
    #print(attributes)
  

# Our Breast Cancer Domain
class BreastDomain(Domain):

  def loadFile(self, fileName):
    lines = super().loadFile(fileName)

    for line in lines: 
      #Removing examples with missing information
      if(line.find("?")!= -1): 
        continue
      line = line.replace("left_", "0,")
      line = line.replace("right_", "1,")
      line = line.replace("central", "0.5,0.5")
      line = line.replace("low", "0")
      line = line.replace("up", "1")
      # split into training and testing
      if (random.random() < .67):
        self.arrayOfTraining.append(self.createNode(line))
      else :
        self.arrayOfTest.append(self.createNode(line))
    print('Training set: ', len(self.arrayOfTraining))  
    print('Testing Set: ', len(self.arrayOfTest))

  def createNode(self,line):
    attributes = super().createNode(line)

    classify = 0 if attributes[0] == "no-recurrence-events" else 1

    age = attributes[1].split('-')[0]
    age = normalize(age,20,70)

    menopause = attributes[2]

    tumorSize = attributes[3].split('-')[0]
    tumorSize = normalize(tumorSize,0,50)

    invNodes = attributes[4].split('-')[0]
    invNodes = normalize(invNodes,0,24)

    nodeCaps = 0 if attributes[5] == "no" else 1 

    degMalign = int(normalize(attributes[6],0,3))

    breast = 0 if attributes[7] == "left" else 1

    breastQuadX = float(attributes[8])

    breastQuadY = float(attributes[9])

    irradiat = 0 if attributes[10] == "no" else 1

    return Node([classify,age,menopause,tumorSize,invNodes,nodeCaps,degMalign,breast,breastQuadX,breastQuadY,irradiat, 0, None])


class Node:   
  
    def __init__(self, attributes):
        
      self.attributes = attributes
    
      0)  self.classify = classify
      1)  self.age = age
      2)  self.menopause = menopause
      3)  self.tumorSize = tumorSize
      4)  self.invNodes = invNodes
      5)  self.nodeCaps = nodeCaps
      6)  self.degMalig = degMalig
      7)  self.breast = breast
      8)  self.breastQuadX = breastQuadX
      9)  self.breastQuadY = breastQuadY
      10) self.irradiat = irradiat
      11) self.correctGuess = 0
      12) self.distance = None
  
      print(self.attributes)

    #I think all these should be methods of Domain, since they are common functions to all domains and use arrays arrayOfTest and arrayOfTraining
    @classmethod
    def findDistance(testNode):
        pass
    
    @staticmethod
    def sortDistances(arrayOfNodes):
        pass
    
    def getKNN(arrayOfNodes, k):
        pass
    
    def getErrorRate():
        pass

    #This function is not doing anything
    @staticmethod
    def loadFile(fileName):
      lines = [line.rstrip('\n') for line in open(fileName)]
      for line in lines: 
        arrayofNodes.append(Node.createNode(line))

    def start(k): 
        
        load file, create node, find distance, sort, vote, get error rate
        
        pass
        
    
breastCancer = BreastDomain()
breastCancer.loadFile("yugoslavia.txt")
"""




"""
#Things to change for each dataset:
X = dataset[:,0:numAttributes]
Y = dataset[:,numAttributes]


the first model.add must have an input_dim = numAttributes
the last model.add must have an output_dim (the first argument) = 1.

# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers as Optimizer
import matplotlib.pyplot as plt
import numpy
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:60]
Y = dataset[:,60]



# create model
#for i in range(1, 10):
model = Sequential()
model.add(Dense(60, input_dim=60, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
#model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
Optimizer.RMSprop(learning_rate=0.01, rho=0.9)
model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['mse'])
# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=100, verbose=0)
# list all data in history
#print(history.history.keys())


# summarize history for accuracy
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
"""
print("OK")
print(__doc__)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# #############################################################################


from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap

n_neighbors = 10

# import some data to play with
iris = datasets.load_iris()
print(iris)
# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()

print("NN STUFFFF!?!?!?")



import pandas as pd
df = pd.read_csv('')
df.head(2)






















