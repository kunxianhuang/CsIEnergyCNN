#!/bin/env python3
#-*-coding=utf-8-*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#draw histogram of energues and corrected energies
def drawhist(csvfn = "../test_result/CsIArray_testout_200.csv"):

    csi_pd = pd.read_table(csvfn, sep=",")

    print(csi_pd.columns)
    depos_energies = csi_pd.loc[:,'Sum']
    de_mean = depos_energies.mean()
    de_std = depos_energies.std()
    print("mean:{}, std:{}".format(de_mean, de_std))
    n, bins, patches = plt.hist(depos_energies, 500, range=(0.0,2500),facecolor='red', label="Deposited Energies")

    corre_energies = csi_pd.loc[:,'Ene_predict']
    co_mean = corre_energies.mean()
    co_std = corre_energies.std()
    print("mean:{}, std:{}".format(co_mean, co_std))
    n, bins, patches = plt.hist(corre_energies, 500, range=(0.0,2500),facecolor='green', label="Corrected Energies")

        
    plt.xlabel('Energies(MeV)')
    plt.ylabel('Entries')
    plt.title('Test data gamma-ray deposited energies')
    #plt.test(100, 60, r'$\mu='e_mean',\ \sigma='e_std'$')
    plt.axis([10.0, 2500, 0, 4400])
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
    

    return


def main():
    drawhist(csvfn = "../test_result/CsIArray_testout_200.csv")

if __name__=="__main__":

    main()
