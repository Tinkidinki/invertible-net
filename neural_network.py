import numpy as np
import matplotlib.pyplot as plt

class neural_network:
    
    def __init__(self, X, y, layers):
        self.X = X
        self.y = y
        self.layers = [len(X[0])] + layers + [len(y[0])]
        self.num_layers = len(self.layers)
        self.n = self.num_layers - 1
        self.z = [np.zeros((l_i, 1)) for l_i in self.layers]
        self.a = [np.zeros((l_i, 1)) for l_i in self.layers]
        self.W = [np.random.rand(self.layers[i+1], self.layers[i])
                        for i in range(self.n)]
        
        self.dW = [np.zeros((self.layers[i+1], self.layers[i]))
                        for i in range(self.n)]
    
                           
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self,x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (0.1 if x<=0 else 1)
    
    def derivative_az(self, layer):
        return np.diag([self.relu_derivative(self.z[layer][j, 0]) 
                        for j in range(self.layers[layer])
                           ])
    
    def derivative_za(self, layer):
        return self.W[layer-1]

    def error(self, sample):
        # print(self.y[sample])
        # print(self.z[self.n])
        # print(self.layers[self.n])
        return sum((self.y[sample][j, 0] - self.z[self.n][j,0])**2 for j in range(self.layers[self.n])) 
    
    def negative_error_derivative(self, t):
        return self.learning_rate * np.array([[2*(self.y[t][j, 0] - self.z[self.n][j,0]) for j in range(self.layers[self.n])]]) 
    
    def feedforward(self):
        for layer in range(1, self.num_layers):
            self.z[layer] = self.W[layer-1] @ self.a[layer-1]
            self.a[layer] = self.relu(self.z[layer])
    
    def backprop(self, sample):
        delta = self.negative_error_derivative(sample)
        # print("Before")
        # print(self.dW)
        # print("delta")
        # print(delta)
        # print("a")
        # print(self.a)
        self.dW[self.n - 1] = self.dW[self.n - 1] + (np.transpose(delta) @ np.transpose(self.a[self.n - 1]))
        for layer in range(self.n-2, -1, -1):
            # print("layer "+str(layer))
            # print("delta")
            # print(delta)
            delta = delta @ self.derivative_za(layer+2) @ self.derivative_az(layer+1)
            # print(delta)
            self.dW[layer] = self.dW[layer] + (np.transpose(delta) 
                                               @ np.transpose(self.a[layer]))
            
    
    def train(self, epochs, learning_rate):

        self.learning_rate = learning_rate
        self.W = [-10 + (20 * np.random.rand(self.layers[i+1], self.layers[i])) for i in range(self.n)]

        epoch_error_array = []
        for epoch in range(epochs):
            self.dW = [np.zeros((self.layers[i+1], self.layers[i]))
                        for i in range(self.n)]
            
            
            epoch_error = 0
            for sample in range(len(self.X)):
                self.a[0] = self.X[sample]
                self.feedforward()
                epoch_error += self.error(sample)
                self.backprop(sample)
                   
            print("EPOCH: "+ str(epoch) + "   ERROR: "+str(epoch_error))
            epoch_error_array.append(epoch_error)

            for layer in range(0, self.n):
                print("dW ")
                print(self.dW)
                print("Weight")
                print(self.W)
                self.W[layer] = self.W[layer] + self.dW[layer]

        plt.plot(epoch_error_array)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()
        plt.close()
        print(self.W)

            
    def test(self, X_test):
        outputs = []
        for x in X_test:
            self.a[0] = x
            self.feedforward()
            outputs.append(self.z[self.n])

        
        return outputs
        

