if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero.models import MLP
import numpy as np
from dezero import as_variable
from dezero import as_array
import dezero.functions as F
from dezero import Variable

model = MLP((10, 3))

x = np.array([[0.2, -0.4]])
y = model(x)
# print(y)

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

# p = softmax1d(y)
# print(p)

def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y

y2 = np.array([as_array(model(x)), as_array(model(x)), as_array(model(x))])
p2 = softmax_simple(y2)
print(p2)
