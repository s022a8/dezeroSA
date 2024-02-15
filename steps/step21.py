import numpy as np
import sys
sys.path.append('.')
from src.main import *
import config.config as cfg

# x = Variable(np.array(2.0))
# y = x + np.array(3.0)
# print(y)

x = Variable(np.array(2.0))
y = 3.0 * x + 1.0
print(y)

x = Variable(np.array([1.0]))
y = np.array([2.0]) + x
print(y)