# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    A = tx.T @ tx
    b = tx.T @ y
    
    w = np.linalg.solve(A, b)
    
    e = y - tx @ w
    mse = (1/(2*len(y))) * e**2
    
    return w, mse
