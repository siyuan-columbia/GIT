# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:49:21 2019

@author: sli
"""
import sys
import numpy as np
N=int(sys.argv[1])
isBadList=(sys.argv[2].split(','))
isBadList=np.array([int(i) for i in isBadList])
die=np.arange(1,N+1)


def threshold(N,isBadList):
    P_bad=sum(isBadList)/N
    die=np.arange(1,N+1)
    return sum(die*((1-np.array(isBadList))/N))/P_bad

def DieN(N,isBadList,money_start):
    die=np.arange(1,N+1)
    dieList=die*(1-np.array(isBadList))
    stopSignal=threshold(N,isBadList)
    out=np.zeros(N)
    if money_start<=stopSignal:
        for i in range(len(dieList)):
            if dieList[i] ==0:
                out[i]=0
            else:
                money_now=money_start+dieList[i]
                out[i]=DieN(N,isBadList,money_now)
    else: 
        return money_start
    
    return np.mean(out)
    
    
    

if __name__=="__main__":
    print(DieN(N,isBadList,money_start=0))

