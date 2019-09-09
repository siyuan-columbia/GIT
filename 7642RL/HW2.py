import sys
import numpy as np
from sympy import *
P=float(sys.argv[1])
valueEst=sys.argv[2].split(',')
reward=sys.argv[3].split(',')
valueEst=np.array([float(i) for i in valueEst])
reward=np.array([float(i) for i in reward])
import ipdb

def TD(P,valueEst,reward):
    exp=np.zeros(len(valueEst)-2)
    exp[0]=valueEst[0]+P*(valueEst[1]+reward[0])+(1-P)*(valueEst[2]+reward[1])-valueEst[0]
    exp[1]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+valueEst[3]-valueEst[0]
    exp[2]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+reward[4]+valueEst[4]-valueEst[0]
    exp[3]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+reward[4]+reward[5]+valueEst[5]-valueEst[0]
    #exp[4]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+reward[4]+reward[5]+reward[6]+valueEst[6]-valueEst[0]
    exp[4]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+reward[4]+reward[5]+reward[6]+0-valueEst[0]

    #lam=0.403032
    #lam=0.6226326309908364
    #lam=0.622
#    lam=symbols('lam')
#    i=symbols('i')
#    left=Eq(exp[0]*(1-lam)+exp[1]*(1-lam)*lam+exp[2]*(1-lam)*lam**2+exp[3]*(1-lam)*lam**3+exp[4]*(1-lam)*lam**4+exp[4]*Sum((1-lam)*(lam**i),(i,5,oo)),exp[4])
#    #exp[0]*(1-lam)+exp[1]*(1-lam)*lam+exp[2]*(1-lam)*lam**2+exp[3]*(1-lam)*lam**3+exp[4]*(1-lam)*lam**4
#    #res=solve(left,exp[4])
#    #solve(left,lam)
#    equation=exp[0]*(1-lam)+exp[1]*(1-lam)*lam+exp[2]*(1-lam)*lam**2+exp[3]*(1-lam)*lam**3\
#            +exp[4]*(1-lam)*lam**4+exp[4]*(1-lam)*lam**5+exp[4]*(1-lam)*lam**6+exp[4]*(1-lam)*lam**7\
#            +exp[4]*(1-lam)*lam**8+exp[4]*(1-lam)*lam**9+exp[4]*(1-lam)*lam**10+exp[4]*(1-lam)*lam**11\
#            +exp[4]*(1-lam)*lam**12+exp[4]*(1-lam)*lam**13+exp[4]*(1-lam)*lam**14\
#            +exp[4]*(1-lam)*lam**15+exp[4]*(1-lam)*lam**16+exp[4]*(1-lam)*lam**17+exp[4]*(1-lam)*lam**18\
#            +exp[4]*(1-lam)*lam**19+exp[4]*(1-lam)*lam**20+exp[4]*(1-lam)*lam**21+exp[4]*(1-lam)*lam**22\
#            +exp[4]*(1-lam)*lam**23+exp[4]*(1-lam)*lam**24+exp[4]*(1-lam)*lam**25+exp[4]*(1-lam)*lam**26\
#            +exp[4]*(1-lam)*lam**27+exp[4]*(1-lam)*lam**28+exp[4]*(1-lam)*lam**29+exp[4]*(1-lam)*lam**30\
#            +exp[4]*(1-lam)*lam**31+exp[4]*(1-lam)*lam**32+exp[4]*(1-lam)*lam**33+exp[4]*(1-lam)*lam**34\
#            +exp[4]*(1-lam)*lam**35+exp[4]*(1-lam)*lam**36+exp[4]*(1-lam)*lam**37+exp[4]*(1-lam)*lam**38\
#            +exp[4]*(1-lam)*lam**39+exp[4]*(1-lam)*lam**40+exp[4]*(1-lam)*lam**41+exp[4]*(1-lam)*lam**42\
#            +exp[4]*(1-lam)*lam**43+exp[4]*(1-lam)*lam**44+exp[4]*(1-lam)*lam**45+exp[4]*(1-lam)*lam**46\
#            +exp[4]*(1-lam)*lam**47+exp[4]*(1-lam)*lam**48+exp[4]*(1-lam)*lam**49+exp[4]*(1-lam)*lam**50
    for i in np.arange(0,1,0.001):
        lam=i
        equation=exp[0]*(1-lam)+exp[1]*(1-lam)*lam+exp[2]*(1-lam)*lam**2+exp[3]*(1-lam)*lam**3\
            +exp[4]*(1-lam)*lam**4+exp[4]*(1-((1-lam)+(1-lam)*lam+(1-lam)*lam**2+(1-lam)*lam**3+(1-lam)*lam**4))
        lam=i+0.001
        equation_n=exp[0]*(1-lam)+exp[1]*(1-lam)*lam+exp[2]*(1-lam)*lam**2+exp[3]*(1-lam)*lam**3\
            +exp[4]*(1-lam)*lam**4+exp[4]*(1-lam)*lam**5+exp[4]*(1-lam)*lam**6+exp[4]*(1-lam)*lam**7\
            +exp[4]*(1-lam)*lam**8+exp[4]*(1-lam)*lam**9+exp[4]*(1-lam)*lam**10+exp[4]*(1-lam)*lam**11\
            +exp[4]*(1-lam)*lam**12+exp[4]*(1-lam)*lam**13+exp[4]*(1-lam)*lam**14\
            +exp[4]*(1-lam)*lam**15+exp[4]*(1-lam)*lam**16+exp[4]*(1-lam)*lam**17+exp[4]*(1-lam)*lam**18\
            +exp[4]*(1-lam)*lam**19+exp[4]*(1-lam)*lam**20+exp[4]*(1-lam)*lam**21+exp[4]*(1-lam)*lam**22\
            +exp[4]*(1-lam)*lam**23+exp[4]*(1-lam)*lam**24+exp[4]*(1-lam)*lam**25+exp[4]*(1-lam)*lam**26\
            +exp[4]*(1-lam)*lam**27+exp[4]*(1-lam)*lam**28+exp[4]*(1-lam)*lam**29+exp[4]*(1-lam)*lam**30\
            +exp[4]*(1-lam)*lam**31+exp[4]*(1-lam)*lam**32+exp[4]*(1-lam)*lam**33+exp[4]*(1-lam)*lam**34\
            +exp[4]*(1-lam)*lam**35+exp[4]*(1-lam)*lam**36+exp[4]*(1-lam)*lam**37+exp[4]*(1-lam)*lam**38\
            +exp[4]*(1-lam)*lam**39+exp[4]*(1-lam)*lam**40+exp[4]*(1-lam)*lam**41+exp[4]*(1-lam)*lam**42\
            +exp[4]*(1-lam)*lam**43+exp[4]*(1-lam)*lam**44+exp[4]*(1-lam)*lam**45+exp[4]*(1-lam)*lam**46\
            +exp[4]*(1-lam)*lam**47+exp[4]*(1-lam)*lam**48+exp[4]*(1-lam)*lam**49+exp[4]*(1-lam)*lam**50
        if (equation-exp[4])*(equation_n-exp[4])<0:
            res=i
            return res
    
    #solve(equation-exp[4],lam)
    
    
    #return exp[0]*(1-lam)+exp[1]*(1-lam)*lam+exp[2]*(1-lam)*lam**2+exp[3]*(1-lam)*lam**3+exp[4]*(1-lam)*lam**4

    


if __name__=="__main__":
    print(TD(P,valueEst,reward))
