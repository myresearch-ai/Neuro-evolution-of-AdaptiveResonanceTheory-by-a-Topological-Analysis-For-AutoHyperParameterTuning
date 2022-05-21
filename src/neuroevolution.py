# -*- coding: utf-8 -*-

__author__  = "Ed Mwanza, AI Researcher - Comp.Sci"
__project__ = "Neuro-evolution of ART using Genetic Algorithms"
__credits__ = "Eyal Wirsansky, Jason Brownlee"


from pandas import read_csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score #confusion_matrix, accuracy_score
from adaptiveresonancetheory import FuzzyARTMAP


class HyperparameterTuningGenetic:

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.initWineDataset()

    def initWineDataset(self):
        source = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv'
        # load the dataset as a numpy array
        self.data = read_csv(source, header=None)
        # retrieve numpy array
        self.data = self.data.values
        # split into input and output elements
        self.X, self.y = self.data[:, :-1]/np.max(self.data[:,:-1], axis=0), self.data[:, -1]
        # label encode the target variable to have the classes 0 and 1
        self.y = LabelEncoder().fit_transform(self.y)
        self.x_train, self.x_test,self.y_train,self.y_test=train_test_split(self.X,self.y,
                                                                            test_size=0.25,random_state=self.randomSeed, 
                                                                            stratify = self.y)


    def convertParams(self, params):
        alpha   = params[0]  
        gamma   = params[1]  
        rho     = params[2]  
        epsilon = params[3]
        return alpha, gamma, rho, epsilon

    def getAccuracy(self, params):
        alpha, gamma, rho, epsilon = self.convertParams(params)
        self.classifier = FuzzyARTMAP(alpha = alpha,
                                      gamma = gamma,
                                      rho = rho,
                                      epsilon = epsilon)
        self.classifier.train(self.x_train, self.y_train, epochs=5)
        self.y_pred = self.classifier.test(self.x_test).astype(int)

        score = f1_score(self.y_test, self.y_pred, average = 'weighted')
        
        return score

    def formatParams(self, params):
        return "'alpha'=%1.3f, 'gamma'=%1.3f, 'rho'=%1.3f, 'epsilon = %1.3f'" % (self.convertParams(params))
