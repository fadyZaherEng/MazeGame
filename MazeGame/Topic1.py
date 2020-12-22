from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import random
import math


# region SearchAlgorithms


class Node:
    id = None
    up = None
    down = None
    left = None
    right = None
    previousNode = None

    def __init__(self, value):
        self.value = value


class SearchAlgorithms:
    ''' * DON'T change Class, Function or Parameters Names and Order
        * You can add ANY extra functions,
          classes you need as long as the main
          structure is left as is '''
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    str=''
    end=Node('.')
    start=Node('.')
    r=0
    c=0
    tempPath = []
    tree=[]
    temp=[]
    def __init__(self, mazeStr):
        ''' mazeStr contains the full board
         The board is read row wise,
        the nodes are numbered 0-based starting
        the leftmost node'''
        self.str = mazeStr

    def create_2Darr_ofNode(self):
        arr = []
        arr = [k.split(',') for k in self.str.split(' ')]
        L1 = self.str.split(' ')
        row = len(L1)
        self.r=row
        L2 = L1[0].split(',')
        col = len(L2)
        self.c=col
        FList = []
        tempList = []
        Counter,a,b,c,d = 0,0,0,0,0
        for i in range(row):
            for j in range(col):
                node = Node(arr[i][j])
                node.i,node.j,node.id = i,j,Counter
                Counter += 1
                if ((i - 1 >= 0) and (i - 1 < row)) and ((j >= 0) and (j < col) and (arr[i][j] != '#')):  # up
                    node.up = 1
                    a = 1
                if ((i >= 0) and (i < row)) and ((j + 1 >= 0) and (j + 1 < col) and (arr[i][j] != '#')):  # right
                    node.right = 1
                    b = 1
                if ((i >= 0) and (i < row)) and ((j - 1 >= 0) and (j - 1 < col) and (arr[i][j] != '#')):#left
                    node.left = 1
                    c = 1
                if ((i + 1 >= 0) and (i + 1 < row)) and ((j >= 0) and (j < col) and (arr[i][j] != '#')):  # down
                    node.down = 1
                    d = 1
                if a == 0:
                    node.up = 0
                if b == 0:
                    node.right = 0
                if c == 0:
                    node.left = 0
                if d == 0:
                    node.down = 0
                if node.value == 'E':
                    self.end=node
                if node.value=='S':
                    self.start=node
                tempList.append(node)
            FList.append(tempList)
            tempList = []
            a, b, c, d = 0, 0, 0, 0
        #array = np.array(FList)
        return FList
    def cal_path(self):
        self.path.append(self.end.id)
        while self.end.id!=self.start.id:
            for i in range(self.r):
                for j in range(self.c):
                    if self.tree[i][j].id==self.end.previousNode:
                        self.end=self.tree[i][j]
            self.path.append(self.end.id)
        self.path.reverse()

    def BFS(self):
        self.tree=self.create_2Darr_ofNode()
        Not_Visited=[]
        visited = []
        Not_Visited.append(self.start)
        while Not_Visited:
            tempNode = Not_Visited.pop(0)
            i = tempNode.i
            j = tempNode.j
            visited.append(tempNode)
            if tempNode.value == 'E':
               break
            if tempNode.value=='#':
                continue
            if tempNode.up ==1:        #up
                node=self.tree[i-1][j]
                if node not in visited and node.value!='#' and (node not in Not_Visited):
                   Not_Visited.append(node)
                   self.tree[i-1][j].previousNode=self.tree[i][j].id

            if tempNode.down == 1:     #down
                node = self.tree[i + 1][j]
                if node not in visited and node.value!='#'and (node not in Not_Visited):
                    Not_Visited.append(node)
                    self.tree[i + 1][j].previousNode = self.tree[i][j].id
            if tempNode.left == 1:     #left
                node = self.tree[i ][j-1]
                if node not in visited and node.value!='#' and (node not in Not_Visited):
                    Not_Visited.append(node)
                    self.tree[i][j-1].previousNode = self.tree[i][j].id
            if tempNode.right == 1:    #right
                node = self.tree[i][j+1]
                if node not in visited and node.value!='#' and (node not in Not_Visited):
                    Not_Visited.append(node)
                    self.tree[i ][j+1].previousNode = self.tree[i][j].id
        for x in visited:
          self.fullPath.append(x.id)
        self.cal_path()
        return self.fullPath, self.path


# endregion

# region NeuralNetwork
class NeuralNetwork():
    def __init__(self, learning_rate, threshold):
        self.learning_rate = learning_rate
        self.threshold = threshold
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((2, 1)) - 1

    def step(self, x):
        if x >float(self.threshold):
            return 1
        else:
            return 0
    def train(self, training_inputs, training_outputs, training_iterations):
        for Counter in range(training_iterations):
            self.synaptic_weights+=np.dot(training_inputs.T,(training_outputs-self.think(training_inputs))*self.learning_rate)
    def think(self, inputs):
        return self.step(np.sum(np.dot((inputs.astype(float)),self.synaptic_weights)))




#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn

def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    searchAlgo.create_2Darr_ofNode()
    fullPath, path = searchAlgo.BFS()
    print('**BFS**\n Full Path is: ' + str(fullPath) + "\n Path: " + str(path))

# endregion

# region Neural_Network_Main_Fn
def NN_Main():
    learning_rate = 0.1
    threshold = -0.2
    neural_network = NeuralNetwork(learning_rate, threshold)

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]])

    training_outputs = np.array([[0, 0, 0, 1]]).T

    neural_network.train(training_inputs, training_outputs, 100)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    inputTestCase = [1, 1]

    print("Considering New Situation: ", inputTestCase[0], inputTestCase[1], end=" ")
    print("New Output data: ", end=" ")
    print(neural_network.think(np.array(inputTestCase)))
    print("Wow, we did it!")



######################## MAIN ###########################33
if __name__ == '__main__':
    SearchAlgorithm_Main()
    NN_Main()
  
