# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np

def calculate_mse(y, tx, w):
    """Calculate the loss using MSE fro vector e"""
    e = y - tx.dot(w)
    return 1/(2*len(y))*np.dot(e,e)

def calculate_mae(y, tx, w):
    """Calculate the loss using MAE for vector e"""
    e = y - tx.dot(w)
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y -  tx.dot(w) #tx @ w

    
    return calculate_mse(e)

    #return calculate_mae(e)