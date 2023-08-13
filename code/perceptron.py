import numpy as np

class Perception:
    def __init__(self , learn_rate = 0.1, n_iterations= 1000):
        self.lr = learn_rate
        self.epoch = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, x ,y):

        self.weights=np.zeros(x.shape[1])
        self.bias=0

        for j in range(self.epoch):

            for i in range(x.shape[0]):

                y_pred = self.activation(np.dot(self.weights,x[i])+ self.bias)

                self.weights=self.weights + self.lr*(y[i] - y_pred) * x[i]
                self.bias=self.bias + self.lr*(y[i] - y_pred)
        print("Training done")
        print(self.weights)
        print(self.bias)

    def activation(self,activation):
        if activation>=0:
            return 1
        else:
            return 0
        
    def predict(self,x):
        y_pred=[]
        for i in range(x.shape[0]):
            y_pred.append(self.activation(np.dot(self.weights,x[i])+self.bias))
        return np.array(y_pred)

