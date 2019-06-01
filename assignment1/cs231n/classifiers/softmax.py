from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #set training number and classes number variables
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        #calculate scores and subtract max
        s = np.dot(X[i,:],W)
        s -= np.max(s) #shift to avoid overflow
        
        #take ratios of score exponential to sum of score exponentials
        s_exp = np.exp(s)
        exp_ratios = s_exp/np.sum(s_exp)
        #update loss
        loss += -np.log(exp_ratios[y[i]])
        
        #calculate gradient across all classes
        for j in range(num_classes):
            dW[:,j] += X[i,:]*(exp_ratios[j]-(j==y[i]))
        
    #normalize and regularizaion
    loss /= num_train
    loss += reg*np.sum(W*W)
    dW /= num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    #calculate scores, shift to avoid overflow
    s = np.dot(X,W)
    s -= np.max(s)
    
    #isolate scores for correct classifications
    s_true = s[np.arange(num_train),y]
    s_expsum = np.sum(np.exp(s),axis=1)
    
    #calculate loss
    loss = -np.mean(np.log(np.exp(s_true)/s_expsum))
    
    #calculate gradient
    s_exp = np.exp(s)
    #divide exponential by exponential sum using np.divide
    ratio = np.divide(s_exp,s_expsum.reshape(num_train,1))
    ratio[np.arange(num_train),y] = -(s_expsum-s_exp[np.arange(num_train),y])/s_expsum
    #dot product is unnormalized dW
    dW = X.T.dot(ratio)
    
    #normalize and regularizaio
    loss += reg*np.sum(W*W)
    dW /= num_train
    dW += 2*reg*W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
