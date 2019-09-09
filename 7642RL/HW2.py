import sys
import numpy as np
P=float(sys.argv[1])
valueEst=sys.argv[2].split(',')
reward=sys.argv[3].split(',')
valueEst=np.array([float(i) for i in valueEst])
reward=np.array([float(i) for i in reward])


def TD(P,valueEst,reward):
    exp=np.zeros(len(valueEst)-1)
    exp[0]=valueEst[0]+P*(valueEst[1]+reward[0])+(1-P)*(valueEst[2]+reward[1])-valueEst[0]
    exp[1]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+valueEst[3]-valueEst[0]
    exp[2]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+reward[4]+valueEst[4]-valueEst[0]
    exp[3]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+reward[4]+reward[5]+valueEst[5]-valueEst[0]
    exp[4]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+reward[4]+reward[5]+reward[6]+valueEst[6]-valueEst[0]
    exp[5]=valueEst[0]+P*(reward[0]+reward[2])+(1-P)*(reward[1]+reward[3])+reward[4]+reward[5]+reward[6]+0-valueEst[0]


#    equation=exp[0]*(1-lam)+exp[1]*(1-lam)*lam+exp[2]*(1-lam)*lam**2+exp[3]*(1-lam)*lam**3\
#            +exp[4]*(1-lam)*lam**4+exp[5]*(1-((1-lam)+(1-lam)*lam+(1-lam)*lam**2+(1-lam)*lam**3+(1-lam)*lam**4))
#    equation=exp[0]+lam*(exp[1]-exp[0])+lam**2*(exp[2]-exp[1])+lam**3*(exp[3]-exp[2])+lam**4*(exp[4]-exp[3]) +lam**5*(exp[5]-exp[4])
#    let equation=Gt get :
#    exp[0]+lam*(exp[1]-exp[0])+lam**2*(exp[2]-exp[1])+lam**3*(exp[3]-exp[2])+lam**4*(exp[4]-exp[3]) +lam**5*(exp[5]-exp[4])

    
    #return 0 = exp[0]-exp[5]+lam*(exp[1]-exp[0])+lam**2*(exp[2]-exp[1])+lam**3*(exp[3]-exp[2])+lam**4*(exp[4]-exp[3]) +lam**5*(exp[5]-exp[4])
    return np.roots([exp[5]-exp[4],exp[4]-exp[3],exp[3]-exp[2],exp[2]-exp[1],exp[1]-exp[0],exp[0]-exp[5]])
    


if __name__=="__main__":
    print(TD(P,valueEst,reward))
