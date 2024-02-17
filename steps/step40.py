if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import sum_to
import dezero.functions as F

# x = np.array([[1, 2, 3], [4, 5, 6]])
# y = sum_to(x, (1, 3))
# print(y)

# y = sum_to(x, (2, 1))
# print(y)

# sum_toの動作確認
x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], 
            [[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 124]]])
print(x.shape)

y = sum_to(x, (4, 2, 1))
print(y.shape)
print("sum_to(x, (4, 2, 1)):\n {}".format(y))

y = sum_to(x, (1, 2, 3))
print(y.shape)
print("sum_to(x, (1, 2, 3)):\n {}".format(y))

y = sum_to(x, (4, 1, 3))
print(y.shape)
print("sum_to(x, (4, 1, 3)):\n {}".format(y))

y = sum_to(x, (4, 1, 1))
print(y.shape)
print("sum_to(x, (4, 1, 1)):\n {}".format(y))

y = sum_to(x, (4, 2))
print(y.shape)
print("sum_to(x, (4, 2)):\n {}".format(y))

y = sum_to(x, (4, 1))
print(y.shape)
print("sum_to(x, (4, 1)):\n {}".format(y))

# Add等の演算のブロードキャスト
x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)

y.backward()
print(x0.grad)
print(x1.grad)