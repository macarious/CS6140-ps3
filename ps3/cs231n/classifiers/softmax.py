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

    C = W.shape[1]  # Number of classes
    N = X.shape[0]  # Number of training examples

    for i in range(N):
        # softmax = P(yi|xi) = exp(Wj.T * xi) / sum(exp(Wk.T * xi)
        # loss = -log(P(yi|xi)) = -log(exp(Wj.T * xi) / sum(exp(Wk.T * xi)))
        scores = X[i].dot(W)

        # Ensure numeric stability (from https://cs231n.github.io/linear-classify/#softmax)
        # Shift score values so that the highest value is 0
        log_C = -1 * np.max(scores)
        scores += log_C
        exp_scores = np.exp(scores)

        softmax = exp_scores / np.sum(exp_scores)
        loss += -1 * np.log(softmax[y[i]])

        for j in range(C):
            # dW = (P(yi|xi) - 1) * xi
            dW[:, j] += (softmax[j] - (j == y[i])) * X[i]

    # (total loss) = (data loss) + (regularization loss)
    # (data loss) = loss / N
    # (regularization loss) = reg * sum(W^2)
    loss /= N
    loss += 0.5 * reg * np.sum(W * W)
    
    # (total gradient) = (data gradient) + (regularization gradient)
    # (data gradient) = dW / N
    # (regularization gradient) = reg * W
    dW /= N
    dW += reg * W

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

    N = X.shape[0]  # Number of training examples

    # Calculate the loss
    scores = X.dot(W)
    log_C = -1 * np.max(X.dot(W), axis=1, keepdims=True)
    scores += log_C
    exp_scores = np.exp(scores)
    softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    loss = np.sum(-1 * np.log(softmax[np.arange(N), y]))
    
    # (total loss) = (data loss) + (regularization loss)
    # (data loss) = loss / N
    # (regularization loss) = 0.5 * reg * sum(W^2)
    loss /= N
    loss += 0.5 * reg * np.sum(W * W)

    # Calculate the gradient
    softmax[np.arange(N), y] -= 1
    dW = X.T.dot(softmax)

    dW /= N
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
