# Implementation of Matrix Factorization in Python

The source code [mf.py](mf.py) is an implementation of the matrix factorization algorithm in Python, using stochastic gradient descent. An article with detailed explanation of the algorithm can be found at [http://www.albertauyeung.com/post/python-matrix-factorization/](http://www.albertauyeung.com/post/python-matrix-factorization/).

Below is an example of using the algorithm:

```python
import numpy as np
from mf import MF

# A rating matrix with ratings from 5 users on 4 items
# zero entries are unknown values
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

# Perform training and obtain the user and item matrices 
mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=20)
training_process = mf.train()
print(mf.P)
print(mf.Q)
print(mf.full_matrix())

# Prints the following:
'''
[[ 1.45345236  0.06946249]
 [ 1.12922538  0.2319001 ]
 [-1.21051208  0.94619099]
 [-0.93607816  0.43182699]
 [-0.6919936  -0.93611985]]

[[ 1.42787151 -0.20548935]
 [ 0.84792614  0.29530697]
 [ 0.18071811 -1.2672859 ]
 [-1.4211893   0.20465575]]
 
[[ 4.98407556  2.99856476  3.96309763  1.01351377]
 [ 3.99274702  2.27661831  3.20365416  1.0125506 ]
 [ 1.0064803   1.00498576  2.37696737  4.98530109]
 [ 1.00999456  0.59175173  2.58437035  3.99597255]
 [ 2.26471556  1.01985428  4.9871617   3.9942251 ]]
'''
```
