if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# x = Variable(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 124]]]))
print(x.shape)
# y = F.reshape(x, (6,))
# y.backward(retain_grad=True)
# print(x.grad)

# x = Variable(np.random.randn(1, 2, 3))
# print(x)
# y = x.reshape((2, 3))
# print(y)
# y = x.reshape(2, 3)
# print(y)

y = F.transpose(x, (1, 0))
print(y.shape)
print(y)
y.backward(retain_grad=True)
print(y.grad)
print(y.grad.shape)
print(x.grad)
print(x.grad.shape)