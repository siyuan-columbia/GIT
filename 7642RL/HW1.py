# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:49:21 2019

@author: sli
"""
import sys
import numpy as np
def DieN():
    N=int(sys.argv[1])
    isBadList=(sys.argv[2].split(','))
    isBadList=np.array([int(i) for i in isBadList])
    die=np.arange(1,N+1)
    print(die*((1-isBadList)/N))
    bankroll=0
    P_bad=sum(isBadList)/N
    while -bankroll*P_bad+sum(die*((1-isBadList)/N))>0:#
        print(bankroll)
        bankroll=bankroll-bankroll*P_bad+sum(die*((1-isBadList)/N))
    
    
    print (bankroll)
    return

if __name__=="__main__":
    DieN()