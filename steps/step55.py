if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import utils

H, W = 4, 4
KH, KW = 3, 3
SH, SW = 1, 1
PH, PW = 1, 1

OH = utils.get_conv_outsize(H, KH, SH, PH)
OW = utils.get_conv_outsize(W, KW, SW, PW)
print(OH, OW)