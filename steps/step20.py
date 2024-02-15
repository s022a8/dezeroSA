import numpy as np
import sys
sys.path.append('.')
from src.main import *
import config.config as cfg

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))


# y = a * b + c
z = (a - c) / b  # div(sub(a, c), b)

# y.backward()
z.backward(True)

print(z.data)
print(a.grad)
print(b.grad)
print(c.grad)