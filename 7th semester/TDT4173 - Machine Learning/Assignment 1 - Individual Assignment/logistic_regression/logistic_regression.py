import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, iterations = 200000, error = 1e-5, eta = 0.001):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.iter = iterations
        self.er = error
        self.eta = eta
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        X = pd.DataFrame(X)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.iter):
            grad_w = np.array(2 * np.sum((sigmoid(np.dot(X,self.w) + self.b) - y) * X.T, axis=1))
            grad_b = 2 * np.sum((sigmoid(np.dot(X,self.w) + self.b) - y))
            w_new = self.w - self.eta * grad_w
            b_new = self.b - self.eta * grad_b
            if ((np.linalg.norm(np.append(self.w, self.b), ord=2, axis=-1)) != 0):
                until = (np.linalg.norm(np.append(w_new, b_new) - np.append(self.w, self.b), ord=2, axis=-1))/(np.linalg.norm(np.append(self.w, self.b), ord=2, axis=-1))
                if ((until  < self.er)):
                    print("Iterations:", _+1)
                    break
            self.w = w_new
            self.b = b_new
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        X = pd.DataFrame(X)
        # return [0 if y<0.5 else 1 for y in sigmoid(np.dot(X, self.w) + self.b)]
        return sigmoid(np.dot(X, self.w) + self.b)
    
    def values(self):
        return self.w, self.b
    
    def plot_fun(self, X):
        return np.dot(X,self.w) + self.b
        

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(np.array(y_pred), eps, 1 - eps)  # Avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * (np.log(1 - y_pred)))


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        