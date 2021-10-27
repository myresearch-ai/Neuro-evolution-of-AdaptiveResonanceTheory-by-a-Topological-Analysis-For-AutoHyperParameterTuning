# -*- coding: utf-8 -*-

from functools import partial
import numpy as np


l1_norm = partial(np.linalg.norm, ord=1, axis=-1)


class FuzzyART(object):
    """
    Fuzzy ART: An unsupervised lifelong clustering algorithm due to Stephen Grossberg & Gail Carpenter
    """
    def __init__(self, alpha=1.0, gamma=0.01, rho=0.5, complement_coding=True):
        """        
        :param alpha: learning rate [0,1] 
        :param gamma: regularization term >0
        :param rho: vigilance [0,1]
        :param complement_coding: use complement coding scheme for inputs
        """
        self.alpha = alpha  # learning rate
        self.beta = 1 - alpha
        self.gamma = gamma  # choice parameter
        self.rho = rho  # vigilance
        self.complement_coding = complement_coding

        self.w = None

    def _init_weights(self, x):
        self.w = np.atleast_2d(x)

    def _complement_code(self, x):
        if self.complement_coding:
            return np.hstack((x, 1-x))
        else:
            return x

    def _add_category(self, x):
        self.w = np.vstack((self.w, x))

    def _match_category(self, x):
        fuzzy_weights = np.minimum(x, self.w)
        fuzzy_norm = l1_norm(fuzzy_weights)
        scores = fuzzy_norm / (self.gamma + l1_norm(self.w))
        threshold = fuzzy_norm / l1_norm(x) >= self.rho
        if np.all(threshold == False):
            return -1
        else:
            return np.argmax(scores * threshold.astype(int))

    def train(self, x, epochs=1):
        """        
        :param x: 2d array of size (samples, features), where all features are
         in [0, 1]
        :param epochs: number of training epochs, the training samples are 
        shuffled after each epoch  
        :return: self
        """
        #np.random.seed(0)
        samples = self._complement_code(np.atleast_2d(x))

        if self.w is None:
            self._init_weights(samples[0])

        for epoch in range(epochs):
            for sample in np.random.permutation(samples):
                category = self._match_category(sample)
                if category == -1:
                    self._add_category(sample)
                else:
                    w = self.w[category]
                    self.w[category] = (self.alpha * np.minimum(sample, w) +
                                        self.beta * w)
        return self

    def test(self, x):
        """        
        :param x: 2d array of size (samples, features), where all features are
         in [0, 1] 
        :return: category IDs for each provided sample
        """
        samples = self._complement_code(np.atleast_2d(x))

        categories = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            categories[i] = self._match_category(sample)
        return categories


class FuzzyARTMAP(object):
    """
    Fuzzy ARTMAP: An unsupervised lifelong clustering algorithm due to Stephen Grossberg & Gail Carpenter
    """

    def __init__(self, alpha=1.0, gamma=0.01, rho=0.5, epsilon=-0.0001,
                 complement_coding=True):
        """        
        :param alpha: learning rate [0,1] 
        :param gamma: regularization term >0
        :param rho: vigilance [0,1]
        :param epsilon: match tracking [-1,1]
        :param complement_coding: use complement coding scheme for inputs
        """
        self.alpha = alpha  # learning rate
        self.beta = 1 - alpha
        self.gamma = gamma  # choice parameter
        self.rho = rho  # vigilance
        self.epsilon = epsilon  # match tracking
        self.complement_coding = complement_coding

        self.w = None
        self.out_w = None
        self.n_classes = 0

    def _init_weights(self, x, y):
        self.w = np.atleast_2d(x)
        self.out_w = np.zeros((1, self.n_classes))
        self.out_w[0, y] = 1

    def _complement_code(self, x):
        if self.complement_coding:
            return np.hstack((x, 1-x))
        else:
            return x

    def _add_category(self, x, y):
        self.w = np.vstack((self.w, x))
        self.out_w = np.vstack((self.out_w, np.zeros(self.n_classes)))
        self.out_w[-1, y] = 1

    def _match_category(self, x, y=None):
        _rho = self.rho
        fuzzy_weights = np.minimum(x, self.w)
        fuzzy_norm = l1_norm(fuzzy_weights)
        scores = fuzzy_norm + (1 - self.gamma) * (l1_norm(x) + l1_norm(self.w))
        norms = fuzzy_norm / l1_norm(x)

        threshold = norms >= _rho
        while not np.all(threshold == False):
            y_ = np.argmax(scores * threshold.astype(int))

            if y is None or self.out_w[y_, y] == 1:
                return y_
            else:
                _rho = norms[y_] + self.epsilon
                norms[y_] = 0
                threshold = norms >= _rho
        return -1

    def train(self, x, y, epochs=1):
        """        
        :param x: 2d array of size (samples, features), where all features are
         in [0, 1]
        :param y: 1d array of size (samples,) containing the class label of each
        sample
        :param epochs: number of training epochs, the training samples are 
        shuffled after each epoch  
        :return: self
        """
        # np.random.seed(0)
        samples = self._complement_code(np.atleast_2d(x))
        self.n_classes = len(set(y))

        if self.w is None:
            self._init_weights(samples[0], y[0])

        idx = np.arange(len(samples), dtype=np.uint32)

        for epoch in range(epochs):
            idx = np.random.permutation(idx)
            for sample, label in zip(samples[idx], y[idx]):
                category = self._match_category(sample, label)
                if category == -1:
                    self._add_category(sample, label)
                else:
                    w = self.w[category]
                    self.w[category] = (self.alpha * np.minimum(sample, w) +
                                        self.beta * w)
        return self

    def test(self, x):
        """        
        :param x: 2d array of size (samples, features), where all features are
         in [0, 1] 
        :return: class label for each provided sample
        """
        samples = self._complement_code(np.atleast_2d(x))

        labels = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            category = self._match_category(sample)
            labels[i] = np.argmax(self.out_w[category])
        return labels
    

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

if __name__ == '__main__':
    # TEST CASE
    np.random.seed(0)
    import sklearn.datasets as ds
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score
    iris = ds.load_iris()
    print('Iris length: {}'.format(len(iris)))
    data = iris['data'] / np.max(iris['data'], axis=0)
    
    print('{0}\n{1}\n{2}'.format('-'*50,'FuzzyART','-'*50))
    net = FuzzyART(alpha=0.5, rho=0.75)
    net.train(data, epochs=100)
    print(net.w.shape)
    print('TEST')
    print(net.test(data).astype(int))
    print('TARGET')
    print(iris['target'])
    print('{0}\n{1}\n{2}'.format('-'*50,'FuzzyARTMAP','-'*50))
    x = data
    y = iris['target']
    
    print('\n')
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=123)
    model = FuzzyARTMAP(alpha=1.0, gamma=0.001, rho=0.77)
    model.train(x=x_train, y=y_train, epochs=100)
    print(model.w.shape)
    print('TEST')
    print(model.test(x_test).astype(int))
    y_pred = model.test(x_test).astype(int)
    print('TARGET')
    #print(iris['target'])
    print(y_test)
    print('\n')
    #print(confusion_matrix(y_true=y_test, y_pred=y_pred))
    cnf_matrix = confusion_matrix(y_test, y_pred,labels=[0, 1, 2])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2],
                      title='Confusion matrix, without normalization')
    print('Accuracy Score: {}%'.format(round(accuracy_score(y_test, y_pred),2)))
