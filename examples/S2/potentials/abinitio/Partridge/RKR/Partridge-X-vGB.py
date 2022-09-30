import numpy as np

def G(v):
    we = 719.3
    wexe = 2.81
    return we*(v+1/2) - wexe*(v+1/2)**2

def B(v):
    Be = 0.2927
    alphae = 0.00148
    return Be - alphae*(v+1/2)

for v in range(21):
    print("{:2d}  {:10.5f}   {:10.5f}".format(v, G(v), B(v)))
