import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

class Lowess:
    '''
    A class for fitting and predicting data on a LOWESS model.
    '''
    def __init__(self, kernel=None, tau=0.05):
        if kernel is None:
            self.kern = self.__Gaussian
        else:
            self.kern = kernel
        self.tau = tau
    
    def fit(self, x, y):
        '''
        Fits data on a LOWESS model.

        Args:
            x (numpy.array): Training data
            y (numpy.array): Target values
        '''
        self.xtrain_ = x
        self.yhat_ = y
    
    def predict(self, x_new):
        '''
        Predict using the fitted LOWESS model.

        Args:
            x_new (numpy.array|float): Sample data
        '''
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_
        return self.__lowess(x, y, x_new, self.kern, self.tau)
    
    def __Gaussian(self, x):
        '''A Gaussian kernel smoothing'''
        return np.where(np.abs(x)>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*x**2))
    
    def __dist(self, u, v):
        '''Euclidean distance'''
        if np.isscalar(v):
            return np.array([np.sqrt(np.sum(u[i]-v)**2) for i in range(len(u))])
        if len(v.shape)==1: # If v is a 1-D array
            v = v.reshape(1,-1)
        return np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))])
    
    def __kernel_function(self, xi, x0, kern, tau):
        return kern(self.__dist(xi,x0)/(2*tau))
    
    def __weights_matrix(self, x, x_new, kern, tau):
        '''Constructs a weights matrix for performing LOWESS.'''
        if np.isscalar(x_new):
            return np.array([self.__kernel_function(x,x_new,kern,tau)])
        else:
            n = len(x_new)
            return np.array([self.__kernel_function(x,x_new[i],kern,tau) for i in range(n)])
    
    def __lowess(self, x, y, x_new, kern, tau=0.05):
        '''
        Performs LOWESS smoothing on x and y, then calculates predictions for x_new.

        Args:
            x (np.array): Training data
            y (np.array): Target values
            x_new (numpy.array|float): Sample data
            kern: Kernel smoothing function
            tau (float): Bandwidth, or the width of points along the x-axis
        Returns:
            numpy.array: Return an array of predictions corresponding to each value in x_new.
            int: If x_new is a scalar (e.g., float), return a single prediction.
        '''
        w = self.__weights_matrix(x, x_new, kern, tau)
        lm = LinearRegression()
        if np.isscalar(x_new):
            lm.fit(w.dot(x.reshape(-1,1)), w.dot(y.reshape(-1,1)))
            yest = lm.predict([[x_new]])[0][0]
            return yest
        else:
            n = len(x_new)
            yest_test = np.zeros(n)
            #Looping through all x-points
            for i in range(n):
                if len(w.shape) == 2:
                    lm.fit(np.diag(w[i,:]).dot(x.reshape(-1,1)), np.diag(w[i,:]).dot(y.reshape(-1,1)))
                else:
                    lm.fit(np.diag(w[i,:][0]).dot(x),np.diag(w[i,:][0]).dot(y))
                pred = lm.predict(x_new[i].reshape(1,-1))
                yest_test[i] = pred.item()
            return yest_test
