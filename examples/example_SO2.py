# -*- coding: utf-8 -*-
#########################################################
# SO2
# adiabatic potential curves from diabatic interaction matrix
# based on Park, Jiang, and Field JCP 144, 144313 (2016)
#
# Stephen.Gibson@anu.edu.au

##########################################################

import numpy as np
import cse
import matplotlib.pyplot as plt


def Vab(q3, lamab):
    return lamab * q3


def Vbc(lambc):
    return lambc


def Ma(q3, w3):
    return w3*q3**2/2


def Mb(q3, w3, Dab):
    return Ma(q3, w3) + Dab


def Mc(q3, Dac, D0, l):
    return (Dac - D0)*np.exp(-np.abs(q3/l)) + D0


def SO2_interaction_matrix(q3, Dab=14760, lamab=3400):
    Dac = 18792
    D0 = 3152
    lambc = 500
    w3 = 1350
    l = 2
    ones = np.ones_like(q3)

    # diabatic interaction matrix for all q3 values
    VT = np.array([[Ma(q3, w3),     Vab(q3, lamab),             q3*0],
                   [Vab(q3, lamab), Mb(q3, w3, Dab), Vbc(lambc)*ones],
                   [q3*0,           Vbc(lambc)*ones, Mc(q3, Dac, D0, l)]])

    return VT

if __name__ == "__main__":
    # normal mode grid
    q3 = np.arange(-7, 7, 0.1)

    Dab = [14760, 14760/2, 0]
    lamab = [4000, 3400, 2800]
    nD = len(Dab)
    nl = len(lamab)

    fig = plt.figure(figsize=(nD*3, nl*3))
    k = 0
    for i in range(nl):    # each row a new lambda value
        for j in range(nD):   # each column a new Dab value

            VT = SO2_interaction_matrix(q3, Dab[j], lamab[i])

            # cse to store diabatic interaction matrix
            # fix me - dummy mass
            SO2 = cse.Cse('32S16O16O', R=q3, VT=VT)
            # diagonalize to give the adiabatic matrix
            SO2.diabatic2adiabatic()

            k += 1
            ax = plt.subplot(f'{nD:d}{nl:d}{k:1d}')
            if j % nD == 0:  # start of a new row
                plt.ylabel(fr"$\lambda_{{ab}}=$ {lamab[i]:.0f}", fontsize=12)
            plt.tick_params(axis='y', left=False, right=False,
                            labelleft=False)
            plt.tick_params(axis='x', labelbottom=False)
            plt.plot(q3, SO2.VT[0, 0], 'C0--', label='Diabats')
            plt.plot(q3, SO2.VT[1, 1], 'C0--')
            plt.plot(q3, SO2.VT[2, 2], 'C0--')

            plt.plot(q3, SO2.AT[0, 0], 'C1-', label='Adiabats')
            plt.plot(q3, SO2.AT[1, 1], 'C1-')
            plt.plot(q3, SO2.AT[2, 2], 'C1-')
            plt.title(fr"$D_{{ab}}$ = {Dab[j]:.0f}"
                      fr"{'             ' if i == 2 else ''}")
            if i == nl-1:  # bottom row
                plt.xlabel(r"$q_3$", fontsize=15)
                plt.tick_params(axis='x', labelbottom=True)
            plt.axis(xmin=-7.9, xmax=7.9, ymin=-5000, ymax=20000)
            ax.set_aspect(1.0/1500)

    leg = plt.legend(labelspacing=0.1, bbox_to_anchor=(1.05, 3.8), fontsize=12)
    col = ['C0', 'C1']
    for i, text in enumerate(leg.get_texts()):
        plt.setp(text, color=col[i])

    plt.suptitle(r"SO$_{2}$ vibronic coupling", fontsize=15)
    plt.savefig("output/example_SO2.png", dpi=75)
    plt.show()
