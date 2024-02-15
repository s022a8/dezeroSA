if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
from dezero import Function
from dezero import Variable
from dezero.utils import plot_dot_graph

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001 # 学習率
iters = 10000 # 繰り返し回数

for i in range(iters):
    print(x0, x1)
    
    y = rosenbrock(x0, x1)
    
    x0.cleangrad()
    x1.cleangrad()
    y.backward()
    
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad