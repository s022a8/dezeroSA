import numpy as np
import sys
sys.path.append('.')
from src.main import *

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)

