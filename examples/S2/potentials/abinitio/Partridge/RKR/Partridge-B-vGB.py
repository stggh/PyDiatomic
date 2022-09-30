import numpy as np

def G(v):
    we = 434.0
    wexe = 2.54
    return we*(v+1/2) - wexe*(v+1/2)**2

def B(v):
    Be = 0.2234
    alphae = 0.00198
    return Be - alphae*(v+1/2)

for v in range(21):
    print("{:2d}  {:10.5f}   {:10.5f}".format(v, G(v), B(v)))
