# =============================================================================
# Artificial Neural Network from scratch
# =============================================================================

# Importing Libraries
import numpy as np

class ANN:
    """Artifical Neural Network building with numpy.
    Args:
        layers: number of layers to compose the ANN
        alpha: the step size to the weight regularization, 0.1 by defaut.
    """
    def __init__(self, layers, alpha=0.1):
        
        self.W = []
        self.layers = layers
        self.alpha = alpha
        
        for i in np.arange(0, len(layers) - 2):
            
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
        
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))
    
    #def __repr__(self):
        
        #return "NeuralNetwork: {}".format("-".join(str(1) for 1 in self.layers))
    
    # Defining the sigmoid function
    def sigmoid(self, x):
        
        
        return 1.0 / (1 + np.exp(-x))
    # Defining the derivate of the sigmoid function
    def sigmoid_deriv(self, x):
        
        ds = x*(1 - x)
        return ds
    # Defining the training function - Fit function
    def fit(self, X, y, epochs = 100, displayUpdate = 100):
        """Fit function to train the ANN.
        Args:
            X: feature matrix
            y: label matrix
            epochs: deseride number of epochs
            display update: the interval to display informations.
            
        """
        
        X = np.c_[X, np.ones((X.shape[0]))]
        
        for epoch in np.arange(0, epochs):
            
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch +1, loss))
    # Definng the derivate of the loss function
    def fit_partial(self, x, y):
        
        A = [np.atleast_2d(x)]
        
        # Feedforward
        for layer in np.arange(0, len(self.W)):
            
            net = A[layer].dot(self.W[layer])
            
            out = self.sigmoid(net)
            
            A.append(out)
        # Backpropagation
        
        error = A[-1] - y
        
        D = [error * self.sigmoid_deriv(A[-1])]
        
        for layer in np.arange(len(A)-2, 0, -1):
            
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        
        D = D[::-1]
        
        # Updating the weight
        # Loop over all layers
        for layer in np.arange(0, len(self.W)):
            
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
        # Defining the prediction function
    def predict(self, X, addBias=True):
        
        p = np.atleast_2d(X)
        
        # Adding the bias in the case that it is not defined
        
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        
        # Loop over all layers
        for layer in np.arange(0, len(self.W)):
            
            xw = np.dot(p, self.W[layer])
            p = self.sigmoid(xw)
        return p
    # Defining the loss function
        
    def calculate_loss(self, X, targets):
        
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets)**2)
        loss = loss / len(X[:])
        return loss
        