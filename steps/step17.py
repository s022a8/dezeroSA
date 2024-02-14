import numpy as np
import sys
sys.path.append('.')
from src.main import *

for i in range(10):
    x = Variable(np.random.randn(10000))
    y = square(square(square(x)))
