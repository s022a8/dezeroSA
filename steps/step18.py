import numpy as np
import sys
sys.path.append('.')
from src.main import *
import config.config as cfg

x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
y.backward()
print(cfg.Config.enable_backprop)

with cfg.no_grad():
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    print(cfg.Config.enable_backprop)

print(cfg.Config.enable_backprop)