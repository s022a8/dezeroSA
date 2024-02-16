if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
from dezero import Function
from dezero import Variable
from dezero.utils import plot_dot_graph

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
# y = f(x)
# y.backward(create_graph=True)
# print(x.grad)

# plot_dot_graph(y, False, "y_x^4-2x^2.png", False)

# gx = x.grad
# x.cleangrad()
# gx.backward()
# print(x.grad)

# plot_dot_graph(gx, False, "gx_x^4-2x^2.png", False)

iters = 10

for i in range(iters):
    print(i, x)
    
    y = f(x)
    x.cleangrad()
    y.backward(create_graph=True)
    
    gx = x.grad
    x.cleangrad()
    gx.backward()
    gx2 = x.grad
    
    x.data -= gx.data / gx2.data
    