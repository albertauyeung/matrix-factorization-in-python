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


#because there are many dot products to calculate during the procedure, GPU can be used for this task
#use_gpu = False/True,   Set to True to use GPU in training and prediction



# Perform training and obtain the user and item matrices 
mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=20, use_gpu=False)
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
 
 
 
# Perform training and obtain the user and item matrices, using GPU for the multiplications
mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=20, use_gpu=True)


# Prints the following:
'''

[[-1.45063442 -0.61549057]
 [-1.18644282 -0.25861855]
 [ 0.98953876  0.41593886]
 [ 0.83830212 -0.55288053]
 [-0.50317654  1.14994057]]
 
[[-1.37430118  0.245039  ]
 [ 0.02231156 -1.12899481]
 [-0.98101813  0.33079953]
 [ 1.65238654  0.79608774]]

[[4.993465   2.98861347 5.3236645  1.01310464]
 [3.99121154 1.86497915 4.45591865 1.00713051]
 [1.01303675 0.99894327 2.39137093 4.98668646]
 [1.01104568 2.11692377 2.24681459 3.99308259]
 [4.1289681  1.02158625 4.98319103 3.98910547]]

'''
```
