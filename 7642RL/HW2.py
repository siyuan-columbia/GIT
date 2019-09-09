import sys
import numpy as np
P=float(sys.argv[1])
valueEst=sys.argv[2].split(',')
reward=sys.argv[3].split(',')
valueEst=np.array([float(i) for i in valueEst])
reward=np.array([float(i) for i in reward])

def TD(P,valueEst,reward):
    exp=np.zeros(len(valueEst)-2)
    exp[0]=valueEst[0]+P*(valueEst[1]+reward[0])+(1-P)*(valueEst[2]+reward[1])-valueEst[0]
    exp[1]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+valueEst[3]-valueEst[0]
    exp[2]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+reward[4]+valueEst[4]-valueEst[0]
    exp[3]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+reward[4]+reward[5]+valueEst[5]-valueEst[0]
    exp[4]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+reward[4]+reward[5]+reward[6]+valueEst[6]-valueEst[0]
    
    return exp 
    
    


if __name__=="__main__":
    print (TD(P,valueEst,reward))
