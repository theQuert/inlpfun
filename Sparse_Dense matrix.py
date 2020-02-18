#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:05:31 2019

@author: quert
"""

import numpy as np
from numpy import array
# Dense matrix -> Sparse matrix
from scipy.sparse import csr_matrix 

# Create dense matrix
A = array([[1, 0, 0, 1, 0, 0], [0, 0, 2, 0, 0, 1], [0, 0, 0, 2, 0, 0]])
print(A)

# Convert to sparse matrix (CSR_method)
S = csr_matrix(A)
print(S)

# Reconstrut dense matrix
B = S.todense()
print(B)

# Sparsity of A matrix
sparsity = 1.0 - np.count_nonzero(A) / A.size
print(sparsity)