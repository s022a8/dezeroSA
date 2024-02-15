import numpy as np
import sys
sys.path.append('.')
from src.main import *
import config.config as cfg

x = Variable(np.array(2.0))
y = x ** 3
y.backward(True)
print(y)
print(y.grad)
print(x.grad)