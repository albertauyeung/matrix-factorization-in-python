import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
import numpy as np
import math
import skcuda.misc as misc



class MF():
    
    def __init__(self, R, K, alpha, beta, iterations, use_gpu):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        - use_gpu (boolean) : declares the GPU implementation
        """
        
        self.gpu = gpu
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))
        
        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j, regarding the value of boolean variable 'use_gpu' we do our calculation accordingly 
        """
        
        if(self.use_gpu):
            
            #initialization
            linalg.init()

            #make the appropriate format
            p_gpu = gpuarray.to_gpu(self.P)
            q_gpu = gpuarray.to_gpu(self.Q)

            prediction = self.b + self.b_u[i] + self.b_i[j] + linalg.dot(p_gpu[i, :],q_gpu[j, :],transb='T')
        
        
        else:
                    
            prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        
        
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        
        if(self.use_gpu):
            
            #initialization
            linalg.init()

            #make the appropriate format
            p_gpu = gpuarray.to_gpu(self.P)
            q_gpu = gpuarray.to_gpu(self.Q)

            #we denote as transb='T' that we take the second argument, matrix q_gpu as transpose
            fullMatrix = self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + linalg.dot(p_gpu, q_gpu, transb='T').get()
        
        else:
            
            fullMatrix = self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
            
        return fullMatrix
