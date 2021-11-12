"""""
In this, a direct version of New Q-Newton's method Backtracking, applied for a square system of equations, is proposed.

Author: Tuyen Trung Truong, Mathematics department, University of Oslo. 

October 2021

The rough idea is as follows: 

Consider a system of n equations in m variables: F=(f_1,...,f_n)=0. We will think about this as finding global minima of f=f_1^+...+f_n^2

One compute the gradient of F: H=\nabla F. 

In case F is the gradient of a function g, then H is the Hessian of F, and hence is symmetric and hence is diagonable.

In general, H is not symmetric and not invertible. 

We will compare the performance of: 

Newton's method

Levenberg-Marquardt method with Armijo's Backtracking line search

New Q-Newton's method Backtracking 

New Q-Newton's method Backtracking for systems of equations


]

But experiments show that this way does very well !!

[Another way is to choose the "learning rate" randomly at each step. One can choose the "learning rate" to be < 1/||H||. In this implementation, we will follow this approach, which is simpler to run.]
"""""



import math
import numpy as np
import numdifftools as nd
from numpy import linalg as LA
from random import random
import params as pr
import scipy
import scipy.special as scisp
from scipy.optimize import fmin_bfgs
import algopy
from algopy import UTPM, exp
from utils import *
import time, datetime
import cubic_reg2, cubic_reg
from mpmath import *

"""Here list all cost functions reported in experiments. 
Details for each function, its gradient and Hessian, as well as eigenvalues of the Hessian
"""

#### Importing parameters for functions

gamma = 1/3
a = pr.a
b = pr.b
c6 = pr.c6
c9 = pr.c9
coef = pr.coef
sign1 = pr.sign1
sign2 = pr.sign2
ep = pr.ep
atol=pr.atol
rtol=pr.rtol
#D=pr.D

######## Find real solutions of systems of real functions


def SystemReal(z):
    
    ## Example 1
    #x,y=z
    #f1=x*x+y*y-1
    #f2=x+y+1
    #tmp=np.array([f1,f2])

    ## Example 2: from the paper "Newton homotopies for sampling stationary points of potential energy landscapes, arXiv:1412.3810"
    
    x,y=z
    f1=(29/16)*(x**3)-2*x*y
    f2=y-(x**2)
    tmp=np.array([f1,f2])
    
    return tmp


def JacobianR(z):
    SystemRealJ=nd.Gradient(SystemReal)
    tmp=SystemRealJ(z)

    return tmp

def CostFunctionR(z):
    tmp=0.5*sum(SystemReal(z)*SystemReal(z))
    return tmp

def CostFunctionRDer(z):
    CostFunctionRG=nd.Gradient(CostFunctionR)
    tmp=CostFunctionRG(z)
    return tmp

def CostFunctionRHessian(z):
    CostFunctionRH=nd.Hessian(CostFunctionR)
    tmp=CostFunctionRH(z)
    return tmp



#########

########### f49: find complex roots to sin(z)=0 #################
######## Use that sin(x+iy)=sin(x)cosh(y)+i cos(x)sinh(y)

def f49(z):
    x,y=z
    tmp=(np.sin(x)*np.cosh(y))*(np.sin(x)*np.cosh(y))+(np.cos(x)*np.sinh(y))*(np.cos(x)*np.sinh(y))
    return tmp


def f49Der(z):
    
    f49G=nd.Gradient(f49)
    tmp=f49G(z)
    return tmp

def f49Hessian(z):
    f49H=nd.Hessian(f49)
    tmp=f49H(z)
    #w,v=LA.eig(tmp)
    return tmp


#########

########### f50: find complex roots to a general analytic function in 1 complex variable #################
######## You can input the function in tmp= in Cf50

def Df50(w):
    
    tmp=(1-1.005*np.exp(-w)+0.525*np.exp(-2*w)-0.475*np.exp(-3*w)-0.045*np.exp(-4*w))/(2.27*np.exp(-w)-2.19*np.exp(-2*w)+1.86*np.exp(-3*w)-0.38*np.exp(-4*w))

    return tmp


def Cf50(z):
    x,y=z
    w=x+y*1j
    #EXAMPLE 1: formula (4.2) in the paper Delves and Lyness, ~1960
    #tmp=1250162561*(w**16)+385455882*(w**15)+845947696*(w**14)+240775148*(w**13)+247926664*(w**12)+64249356*(w**11)+41018752*(w**10)+9490840*(w**9)+4178260*(w**8)+837860*(w**7)+267232*(w**6)+44184*(w**5)+10416*(w**4)+1288*(w**3)+224*(w**2)+16*(w**1)+2
    
    #Example 2: Saddle point
    #tmp=(w**2)+1
    
    
    #Example 3: Derivative of formula 7.4 in the paper Delves and Lyness, ~1960
    #Df50G=nd.Gradient(Df50)
    #tmp=Df50G(w)
    
    #Example 4: Riemann zeta function. Numdiff ia not compatible with mpmath (used to define the Riemann zeta function)
    #tmp=zeta(w)
    
    #Example 5: Multiple roots
    #tmp=w*((w-1)**2)*((w-2)**3)*((w-5)**5)
    
    #Example 6: sin(z)
    #tmp=np.sin(w)
    
    
    #Example 7: Approximation of Riemann zeta function
    tmp=1
    for jj in range(1000):
        tmp=tmp+np.exp(-w*np.log(jj+2))
    
    #Example 8: Example from Delves and Lyness, ~1960, J1^2-J0J2
    #tmp=(scisp.jv(1,z))*(scisp.jv(1,z))-(scisp.jv(0,z))*(scisp.jv(2,z))
    
    
    return tmp

def f50(z):
    x,y=z
    u=Cf50(z).real
    v=Cf50(z).imag
    
    #For the Riemann zeta function
    #u=libmp.to_float(Cf50(z).real)
    #v=libmp.to_float(Cf50(z).imag)

    tmp=u*u+v*v
    return tmp


def f50Der(z):
    
    f50G=nd.Gradient(f50)
    tmp=f50G(z)
    return tmp

def f50Hessian(z):
    f50H=nd.Hessian(f50)
    tmp=f50H(z)
    #w,v=LA.eig(tmp)
    return tmp




################

#z00=np.array([-32+random()*64 for _ in range(D)])
#z00=np.array([23.49261912, -13.86471849])

#print(z00)
#print(f23(z00))
#print(f23Der(z00))
#print(f23Hessian(z00))

#print(f23(z00).dtype)
#print(type(f23(z00)))


#print(f23Der(z00).dtype)
#print(type(f23Der(z00)))






#Newton's method:
def NewtonMethod():
    print("0.Newton's method:")

    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    for m in range(NIterate):
        #print("Iteration =",m)
        tmp = fHessian(z00)
        Der=fDer(z00)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = Der * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = np.matmul(Der, HessInv)
    
        z00 = z00 - gn
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<atol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return

def LocalNewtonMethod():
    print("0.Local.Newton's method:")

    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    for m in range(NIterate):
        #print("Iteration =",m)
        tmp = fHessian(z00)
        Der=fDer(z00)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = Der * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = np.matmul(Der, HessInv)
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return


#Random Newton's method:

def RandomNewtonMethod():

    print("1.Random Newton's method:")

    z00=z00_old
    z00_1=z00_old
    #print(z00)

    minLR = 0
    maxLR = 2

    time0=time.time()
    for m in range(NIterate):

        delta = minLR + random() * (maxLR - minLR)
        Der=fDer(z00)
        tmp = fHessian(z00)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = delta * Der * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = delta * np.matmul(Der, HessInv)
    
        z00 = z00 - gn
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return
    
    
def LocalRandomNewtonMethod():

    print("1.Local.Random Newton's method:")

    z00=z00_old
    z00_1=z00_old
    #print(z00)

    minLR = 0
    maxLR = 2

    time0=time.time()
    for m in range(NIterate):

        delta = minLR + random() * (maxLR - minLR)
        Der=fDer(z00)
        tmp = fHessian(z00)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = delta * Der * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = delta * np.matmul(Der, HessInv)
    
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return
    


def NewQNewtonG2():
    
    print("16. New Q Newton's method G2:")
    
    z00=z00_old
    z00_1=z00_old
    delta=1
    time0=time.time()
    #print(z00)
    for m in range(NIterate):
        #print("Iteration=",m)
        tmp=fHessian(z00)
        Der=fDer(z00)
        if dimN == 1:
            HessTr = tmp
            #print(HessTr)
            if abs(HessTr)==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))
            #print(HessTr)
            
            HessInv = 1 / HessTr
            #print(HessInv)
            gn = Der * HessInv
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        elif dimN > 1:
            HessTr = tmp.transpose()
            #print(HessTr)
            #print(f26Der(z00))
            if abs(LA.det(HessTr))==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))*np.identity(dimN,dtype=float)
            #print(HessTr)
            #HessInv = LA.inv(HessTr)
            #print(HessInv)

            gn=0
            for i in range(dimN):
                gn=gn+(np.matmul(Der,Evect[i])/(math.sqrt(L2Norm2(np.matmul(HessTr,Evect[i]),dimN))))*Evect[i]
        z00 = z00 - gn
        if verbose:
            print(f(z00))
        #print(z00)
        
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return







#NewQNewtonPlusV1G2

def NewQNewtonPlusV1G2():
    
    print("18. New Q Newton's method Plus V1G2:")
    
    z00=z00_old
    z00_1=z00_old
    delta=1
    time0=time.time()
    #print(z00)
    for m in range(NIterate):
        #print("Iteration=",m)
        tmp=fHessian(z00)
        Der=fDer(z00)
        if dimN == 1:
            HessTr = tmp
            #print(HessTr)
            if abs(HessTr)==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))
            #print(HessTr)
            
            HessInv = 1 / HessTr
            #print(HessInv)
            gn = Der * HessInv
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        elif dimN > 1:
            HessTr = tmp.transpose()
            #print(HessTr)
            #print(f26Der(z00))
            if abs(LA.det(HessTr))==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))*np.identity(dimN,dtype=float)
            #print(HessTr)
            #HessInv = LA.inv(HessTr)
            #print(HessInv)
            gn=0
            for i in range(dimN):
                gn=gn+(np.matmul(Der,Evect[i])/(math.sqrt(L2Norm2(np.matmul(HessTr,Evect[i]),dimN))))*Evect[i]
    

        if (f(z00-gn)>f(z00)):
            gamgam=1
            while (f(z00-gamgam*gn)>f(z00)):
                gamgam=gamgam/2
            z00=z00-gamgam*gn
        else:
            z00 = z00 - gn
        if verbose:
            print(f(z00))
        #print(z00)
        
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return







def NewQNewtonPlusV2G2():
    
    print("20. New Q Newton's method Plus V2G2:")
    
    z00=z00_old
    z00_1=z00_old
    delta=1
    time0=time.time()
    #print(z00)
    for m in range(NIterate):
        #print("Iteration=",m)
        tmp=fHessian(z00)
        Der=fDer(z00)
        if dimN == 1:
            HessTr = tmp
            #print(HessTr)
            if abs(HessTr)==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))
            #print(HessTr)
            
            HessInv = 1 / HessTr
            #print(HessInv)
            gn = Der * HessInv
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        elif dimN > 1:
            HessTr = tmp.transpose()
            #print(HessTr)
            #print(f26Der(z00))
            if abs(LA.det(HessTr))==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))*np.identity(dimN,dtype=float)
            #print(HessTr)
            #HessInv = LA.inv(HessTr)
            #print(HessInv)
            gn=0
            for i in range(dimN):
                gn=gn+(np.matmul(Der,Evect[i])/(math.sqrt(L2Norm2(np.matmul(HessTr,Evect[i]),dimN))))*Evect[i]

        gamgam=1
        
        gn=gn/(1+math.sqrt(L2Norm2(gn,dimN)))
        gnNorm=np.dot(gn,Der)
        while (f(z00-gamgam*gn)>f(z00)-0.5*gamgam*gnNorm):
            gamgam=gamgam/2
        z00=z00-gamgam*gn
        
        if verbose:
            print(f(z00))
        #print(z00)

        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return


def NewQNewtonPlusV3G2():
    
    print("22. New Q Newton's method Plus V3G2:")
    
    z00=z00_old
    z00_1=z00_old
    delta=1
    time0=time.time()
    #print(z00)
    for m in range(NIterate):
        #print("Iteration=",m)
        tmp=fHessian(z00)
        Der=fDer(z00)
        if dimN == 1:
            HessTr = tmp
            #print(HessTr)
            if abs(HessTr)==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))
            #print(HessTr)
            
            HessInv = 1 / HessTr
            #print(HessInv)
            gn = Der * HessInv
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        elif dimN > 1:
            HessTr = tmp.transpose()
            #print(HessTr)
            #print(f26Der(z00))
            if abs(LA.det(HessTr))==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))*np.identity(dimN,dtype=float)
            #print(HessTr)
            #HessInv = LA.inv(HessTr)
            #print(HessInv)
            gn=0
            for i in range(dimN):
                gn=gn+(np.matmul(Der,Evect[i])/(math.sqrt(L2Norm2(np.matmul(HessTr,Evect[i]),dimN))))*Evect[i]

        gn=gn/(1+math.sqrt(L2Norm2(gn,dimN)))
        if (f(z00-gn)>f(z00)):
            gamgam=1
            while (f(z00-gamgam*gn)>f(z00)):
                gamgam=gamgam/2
            z00=z00-gamgam*gn
        else:
            z00 = z00 - gn
        if verbose:
            print(f(z00))
        #print(z00)
        
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return





def NewQNewtonPlusV4G2():
    
    print("23. New Q Newton's method Plus V4G2:")
    
    z00=z00_old
    z00_1=z00_old
    delta=1
    time0=time.time()
    #print(z00)
    for m in range(NIterate):
        #print("Iteration=",m)
        tmp=fHessian(z00)
        Der=fDer(z00)
        if dimN == 1:
            HessTr = tmp
            #print(HessTr)
            if abs(HessTr)==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))
            #print(HessTr)
            
            HessInv = 1 / HessTr
            #print(HessInv)
            gn = Der * HessInv
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        elif dimN > 1:
            HessTr = tmp.transpose()
            #print(HessTr)
            #print(f26Der(z00))
            if abs(LA.det(HessTr))==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))*np.identity(dimN,dtype=float)
            #print(HessTr)
            #HessInv = LA.inv(HessTr)
            #print(HessInv)
            gn=0
            for i in range(dimN):
                gn=gn+(np.matmul(Der,Evect[i])/(math.sqrt(L2Norm2(np.matmul(HessTr,Evect[i]),dimN))))*Evect[i]

        gamgam=1
        gnNorm=np.dot(gn,Der)
        while (f(z00-gamgam*gn)>f(z00)-0.5*gamgam*gnNorm):
            gamgam=gamgam/2
        z00=z00-gamgam*gn
        
        if verbose:
            print(f(z00))
        #print(z00)

        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
        #print(fDer(z00))
    return





#BFGS

def BFGS():

    print("4. BFGS")
    z00=z00_old
    #print(z00)
    time0=time.time()
    xopt=fmin_bfgs(f, z00,fprime=fDer,maxiter=NIterate,gtol=pr.rtol)
    #xopt=fmin_bfgs(f, z00,fprime=fDer,maxiter=NIterate,gtol=pr.rtol, retall=True)
    time1=time.time()
    print("time=",time1-time0)
    #print(xopt)
    if printLastPoint==True:
        print("The last point =",xopt)
    print("function value=",f(xopt))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(xopt),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    return
    
def LocalBFGS():

    print("4.Local. BFGS")
    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    
    #xopt=fmin_bfgs(f, z00,fprime=fDer,maxiter=NIterate,gtol=pr.rtol, retall=True)
    
    for m in range(NIterate):
        g_x=fDer(z00)
    
        xopt=fmin_bfgs(f, z00,fprime=fDer,maxiter=1,gtol=pr.rtol)
        kkapp=1
        while constraintChect(z00+kkapp*(xopt-z00))==False:
            kkapp=kkapp/2
        z00=z00+kkapp*(xopt-z00)
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    
    
    time1=time.time()
    print("time=",time1-time0)
    print("number of iterates=", m)
    
    if printLastPoint==True:
        print("The last point =",z00)
    print("function value=", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        tmp, w=fHessian(z00)
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
    
    
    
    
    return

#Adaptive cubic regularisation

def AdaptiveCubicRegularisation():

    print("7. Adaptive Cubic Regularisation")
    z00=z00_old
    #print(z00)
    time0=time.time()
    
    cr=cubic_reg.AdaptiveCubicReg(z00, f=f,gradient=None, hessian=None,hessian_update_method='exact', conv_tol=rtol)
    xopt, intermediate_points,  n_iter, flag=cr.adaptive_cubic_reg()
    
    #cr2=cubic_reg2.AdaptiveCubicReg(z00, f=f,gradient=None, hessian=None,hessian_update_method='exact', conv_tol=rtol)
    #xopt, intermediate_points, intermediate_function_values,  n_iter, flag=cr2.adaptive_cubic_reg()
    
    time1=time.time()
    print("time=",time1-time0)
    print("number of iterates=", n_iter)
    #print("optimal point=", xopt)
    if printLastPoint==True:
        print("The last point =",xopt)
    print("function value=", f(xopt))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(xopt),dimN)))
    #print("function values of intermediate points=", intermediate_function_values)
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)

    return


#Local Adaptive cubic regularisation

def LocalAdaptiveCubicRegularisation():

    print("7.Local. Adaptive Cubic Regularisation")
    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    
    for m in range(NIterate):
        g_x=fDer(z00)
    
        #cr=cubic_reg.AdaptiveCubicReg(z00, f=f,gradient=None, hessian=None,hessian_update_method='exact', conv_tol=rtol, maxiter=1)
        #xopt, intermediate_points,  n_iter, flag=cr.adaptive_cubic_reg()
        cr2=cubic_reg2.AdaptiveCubicReg(z00, f=f,gradient=None, hessian=None,hessian_update_method='exact', conv_tol=rtol,maxiter=1)
        xopt, intermediate_points, intermediate_function_values,  n_iter, flag=cr2.adaptive_cubic_reg()
        kkapp=1
        while constraintChect(z00+kkapp*(xopt-z00))==False:
            kkapp=kkapp/2
        z00=z00+kkapp*(xopt-z00)
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
        
    
    
    time1=time.time()
    print("time=",time1-time0)
    print("number of iterates=", n_iter)
    #print("optimal point=", xopt)
    if printLastPoint==True:
        print("The last point =",z00)
    print("function value=", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
    return



#Backtraking Gradient Descent

def BacktrackingGD():

    print("5. Backtracking Gradient Descent")

    z00=z00_old
    z00_1=z00_old
    #print(z00)

    time0=time.time()
    delta0=pr.delta0
    alpha=pr.alpha
    beta=pr.beta

    for m in range(NIterate):
        delta = delta0
        x = z00
        f_x = f(x)
        g_x = fDer(x)
        gx_norm = L2Norm2(g_x, dimN)
        #g_xx, w = fHessian(x)
        #gxx_norm = math.sqrt(L2Norm2(g_xx, dimN))
        x_check = x - delta * g_x
        check = f(x_check) - f_x + alpha * delta * gx_norm
        count = 0

        if check > 0:
            while check > 0:  # and delta > 1e-6:
                count += 1
                delta = delta * beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x_check) - f_x + alpha * delta * gx_norm
    
        gn = delta * fDer(z00)
        z00 = z00 - gn
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
    #print(fDer(z00))
    return

def LocalBacktrackingGD():

    print("5.Local. Backtracking Gradient Descent")

    z00=z00_old
    z00_1=z00_old
    #print(z00)

    time0=time.time()
    delta0=pr.delta0
    alpha=pr.alpha
    beta=pr.beta

    for m in range(NIterate):
        delta = delta0
        x = z00
        f_x = f(x)
        g_x = fDer(x)
        gx_norm = L2Norm2(g_x, dimN)
        #g_xx, w = fHessian(x)
        #gxx_norm = math.sqrt(L2Norm2(g_xx, dimN))
        x_check = x - delta * g_x
        check = f(x_check) - f_x + alpha * delta * gx_norm
        count = 0

        if check > 0:
            while check > 0:  # and delta > 1e-6:
                count += 1
                delta = delta * beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x_check) - f_x + alpha * delta * gx_norm
    
        gn = delta * fDer(z00)
        
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
    #print(fDer(z00))
    return



#Two-way Backtracking

def TwoWayBacktrackingGD():
    print("8. Two-way Backtracking GD")
    
    z00=z00_old
    z00_1=z00_old
    
    
    time0=time.time()
    delta0=pr.delta0
    alpha=pr.alpha
    beta=pr.beta
    
    
    
    lr_pure = [delta0]
    productGraph=[]
    

    print("Number of maximum iterates:", NIterate, "; But note that the number of actual iterates may be much smaller")

    for m in range(NIterate):
        delta = lr_pure[-1]
        #    delta_tmp = delta
        x = z00
        f_x = f(x)
        g_x = fDer(x)
        #gx_norm = L2Norm2(g_x, dimN)
        #g_xx, w=fHessian(x)
        gxx_norm=math.sqrt(L2Norm2(g_xx, dimN))
        #x_prev = x
        x_check = x - delta * g_x
        check = f(x_check) - f_x + alpha * delta * gx_norm
        count = 0
        
        #    if avoid == 0:
        if check > 0:
            while check > 0:  # and delta > 1e-6:
                count += 1;
                delta = delta * beta  # rescale delta
                #x_prev = x_check
                x_check = x - delta * g_x
                check = f(x_check) - f_x + alpha * delta * gx_norm
        elif check < 0:
            while check < 0 and delta <= delta0:
                count += 1
                delta = delta / beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x - delta * g_x) - f_x + alpha * delta * gx_norm
            delta = delta * beta

        lr_pure.append(delta)
        #productGraph.append(delta*gxx_norm)
        
        gn = delta * fDer(z00)
        z00 = z00 - gn
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
   
    #print(fDer(z00))
    return

def LocalTwoWayBacktrackingGD():
    print("8.Local. Two-way Backtracking GD")
    
    z00=z00_old
    z00_1=z00_old
    
    
    time0=time.time()
    delta0=pr.delta0
    alpha=pr.alpha
    beta=pr.beta
    
    
    
    lr_pure = [delta0]
    productGraph=[]
    

    print("Number of maximum iterates:", NIterate, "; But note that the number of actual iterates may be much smaller")

    for m in range(NIterate):
        delta = lr_pure[-1]
        #    delta_tmp = delta
        x = z00
        f_x = f(x)
        g_x = fDer(x)
        gx_norm = L2Norm2(g_x, dimN)
        #g_xx, w=fHessian(x)
        #gxx_norm=math.sqrt(L2Norm2(g_xx, dimN))
        #x_prev = x
        x_check = x - delta * g_x
        check = f(x_check) - f_x + alpha * delta * gx_norm
        count = 0
        
        #    if avoid == 0:
        if check > 0:
            while check > 0:  # and delta > 1e-6:
                count += 1;
                delta = delta * beta  # rescale delta
                #x_prev = x_check
                x_check = x - delta * g_x
                check = f(x_check) - f_x + alpha * delta * gx_norm
        elif check < 0:
            while check < 0 and delta <= delta0:
                count += 1
                delta = delta / beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x - delta * g_x) - f_x + alpha * delta * gx_norm
            delta = delta * beta

        lr_pure.append(delta)
        #productGraph.append(delta*gxx_norm)
        
        gn = delta * fDer(z00)
        
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    
   
    #print(fDer(z00))
    return



#Unbounded Two-way Backtracking GD

def UnboundedTwoWayBacktrackingGD():
    print("9. Unbounded Two-way Backtracking GD")
    
    z00=z00_old
    z00_1=z00_old
    
    
    time0=time.time()
    delta0=pr.delta0
    alpha=pr.alpha
    beta=pr.beta
    
    
    
    lr_pure = [delta0]
    productGraph=[]
    

    print("Number of maximum iterates:", NIterate, "; But note that the number of actual iterates may be much smaller")

    for m in range(NIterate):
        delta = lr_pure[-1]
        #    delta_tmp = delta
        x = z00
        f_x = f(x)
        g_x = fDer(x)
        gx_norm = L2Norm2(g_x, dimN)
        gx_normSquareRoot=math.sqrt(gx_norm)
        #g_xx, w=fHessian(x)
        #gxx_norm=math.sqrt(L2Norm2(g_xx, dimN))
        #x_prev = x
        x_check = x - delta * g_x
        check = f(x_check) - f_x + alpha * delta * gx_norm
        count = 0
        
        #    if avoid == 0:
        if check > 0:
            while check > 0:  # and delta > 1e-6:
                count += 1;
                delta = delta * beta  # rescale delta
                #x_prev = x_check
                x_check = x - delta * g_x
                check = f(x_check) - f_x + alpha * delta * gx_norm
        elif check < 0:
            while check < 0 and delta <= UnboundedLR(gx_normSquareRoot,1e+100, delta0):
                count += 1
                delta = delta / beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x - delta * g_x) - f_x + alpha * delta * gx_norm
            delta = delta * beta

        lr_pure.append(delta)
        #productGraph.append(delta*gxx_norm)
        
        gn = delta * fDer(z00)
        z00 = z00 - gn
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    
    #print(fDer(z00))
    flag=0
    for i in range(len(lr_pure)):
        if delta0<lr_pure[i]:
            flag+=1
    print("Number of learning rates bigger than \delta _0=", flag)
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    return


def LocalUnboundedTwoWayBacktrackingGD():
    print("9.Local. Unbounded Two-way Backtracking GD")
    
    z00=z00_old
    z00_1=z00_old
    
    
    time0=time.time()
    delta0=pr.delta0
    alpha=pr.alpha
    beta=pr.beta
    
    
    
    lr_pure = [delta0]
    productGraph=[]
    

    print("Number of maximum iterates:", NIterate, "; But note that the number of actual iterates may be much smaller")

    for m in range(NIterate):
        delta = lr_pure[-1]
        #    delta_tmp = delta
        x = z00
        f_x = f(x)
        g_x = fDer(x)
        gx_norm = L2Norm2(g_x, dimN)
        gx_normSquareRoot=math.sqrt(gx_norm)
        #g_xx, w=fHessian(x)
        #gxx_norm=math.sqrt(L2Norm2(g_xx, dimN))
        #x_prev = x
        x_check = x - delta * g_x
        check = f(x_check) - f_x + alpha * delta * gx_norm
        count = 0
        
        #    if avoid == 0:
        if check > 0:
            while check > 0:  # and delta > 1e-6:
                count += 1;
                delta = delta * beta  # rescale delta
                #x_prev = x_check
                x_check = x - delta * g_x
                check = f(x_check) - f_x + alpha * delta * gx_norm
        elif check < 0:
            while check < 0 and delta <= UnboundedLR(gx_normSquareRoot,1e+100, delta0):
                count += 1
                delta = delta / beta  # rescale delta
                x_check = x - delta * g_x
                check = f(x - delta * g_x) - f_x + alpha * delta * gx_norm
            delta = delta * beta

        lr_pure.append(delta)
        #productGraph.append(delta*gxx_norm)
        
        gn = delta * fDer(z00)
        
        kkapp=1
        while constraintChect(z00-kkapp*gn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 - kkapp*gn
        
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(g_x, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    
    #print(fDer(z00))
    flag=0
    for i in range(len(lr_pure)):
        if delta0<lr_pure[i]:
            flag+=1
    print("Number of learning rates bigger than \delta _0=", flag)
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    return


def InertialNewtonM():
    print("6. Inertial Newton's method")

    z00=z00_old
    z00_1=z00_old

    #print(z00)

    alpha1 = 0.5

    beta1 = 0.1

    lambda1 = (1/beta1) -alpha1



    psin=np.array([-1+random()*2 for _ in range(dimN)])



    time0=time.time()


    for m in range(NIterate):

        gn = fDer(z00)

        gamman=(m+1)**(-0.5)

    

        z00 = z00 + gamman * (lambda1*z00-(1/beta1)*psin-beta1*gn)

        psin=psin+gamman*(lambda1*z00-(1/beta1)*psin)
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(gn, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    return



#Local Inertial Newton's method

def LocalInertialNewtonM():
    print("6.Local. Inertial Newton's method")

    z00=z00_old
    z00_1=z00_old

    #print(z00)

    alpha1 = 0.5

    beta1 = 0.1

    lambda1 = (1/beta1) -alpha1



    psin=np.array([-1+random()*2 for _ in range(dimN)])



    time0=time.time()


    for m in range(NIterate):

        gn = fDer(z00)

        gamman=(m+1)**(-0.5)
        
        ggn=gamman * (lambda1*z00-(1/beta1)*psin-beta1*gn)
        
        kkapp=1
        while constraintChect(z00+kkapp*ggn)==False:
            kkapp=kkapp/2
            
    
        z00 = z00 + kkapp*ggn

        

        psin=psin+gamman*(lambda1*z00-(1/beta1)*psin)
        if verbose:
            print(f(z00))
            #print(z00)
            
        if stopCriterion == 0:
            if (L2Norm2(gn, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    return
    

## Newton's method for system of equations
def NewtonMethodSE():
    print("30.Newton's method for System of Equations:")
    
    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    for m in range(NIterate):
        #print("Iteration =",m)
        tmp = np.matmul(JacobianR(z00).transpose(),JacobianR(z00))
        Der=fDer(z00)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = Der * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = np.matmul(Der, HessInv)
        
        z00 = z00 - gn
        if verbose:
            print(f(z00))
        #print(z00)
        
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<atol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return


## Levenberg-Marquardt method
def LevenbergMarquardt():
    print("32. Levenberg-Marquardt method:")
    
    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    for m in range(NIterate):
        #print("Iteration =",m)
        tmp = np.matmul(JacobianR(z00).transpose(),JacobianR(z00))+cutoff(f(z00))*np.identity(dimN,dtype=float)
        Der=fDer(z00)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = Der * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = np.matmul(Der, HessInv)
        
    
        
        if verbose:
            print(f(z00))
            #print(z00)

        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<atol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return





## Levenberg-Marquardt method with Backtracking line search
def LevenbergMarquardtBacktracking():
    print("31. Levenberg-Marquardt method with Backtracking line search:")
    
    z00=z00_old
    z00_1=z00_old
    #print(z00)
    time0=time.time()
    for m in range(NIterate):
        #print("Iteration =",m)
        tmp = np.matmul(JacobianR(z00).transpose(),JacobianR(z00))+cutoff(f(z00))*np.identity(dimN,dtype=float)
        Der=fDer(z00)
        if dimN == 1:
            HessTr = tmp
            HessInv = 1 / HessTr
            gn = Der * HessInv
        elif dimN > 1:
            HessTr = tmp.transpose()
            HessInv = LA.inv(HessTr)
            gn = np.matmul(Der, HessInv)
        
        gamgam=1
        gnNorm=np.dot(gn,Der)
        while (f(z00-gamgam*gn)>f(z00)-0.5*gamgam*gnNorm):
            gamgam=gamgam/2
        z00=z00-gamgam*gn

        if verbose:
            print(f(z00))
        #print(z00)
        
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<atol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return

## New Q-Newton's Method Backtracking


def NewQNewtonPlus():
    
    print("33. New Q Newton's method Backtracking:")
    
    z00=z00_old
    z00_1=z00_old
    delta=1
    time0=time.time()
    #print(z00)
    for m in range(NIterate):
        #print("Iteration=",m)
        tmp=fHessian(z00)
        evals, evecs=LA.eig(tmp)
        evals=evals.real
        evecs=evecs.real
        Der=fDer(z00)
        if dimN == 1:
            HessTr = tmp
            #print(HessTr)
            if abs(HessTr)==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))
                evals=evals+delta*cutoff(L2Norm2(Der,dimN))
            #print(HessTr)
            
            HessInv = 1 / HessTr
            #print(HessInv)
            gn = Der * HessInv
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        elif dimN > 1:
            HessTr = tmp.transpose()
            #print(HessTr)
            #print(f26Der(z00))
            if abs(LA.det(HessTr))==0:
                HessTr=HessTr+delta*cutoff(L2Norm2(Der,dimN))*np.identity(dimN,dtype=float)
                for i in range(dimN):
                    evals[i]=evals[i]+delta*cutoff(L2Norm2(Der,dimN))
            #print(HessTr)
            #HessInv = LA.inv(HessTr)
            #print(HessInv)
            gn=0
            for i in range(dimN):
                gn=gn+(np.matmul(Der,evecs[:,i])/abs(evals[i]))*evecs[:,i]
    
        gamgam=1
        gnNorm=np.dot(gn,Der)
        while (f(z00-gamgam*gn)>f(z00)-0.5*gamgam*gnNorm):
            gamgam=gamgam/2
        z00=z00-gamgam*gn
    
        if verbose:
            print(f(z00))
            #print(z00)
        
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return



## New Q-Newton's Method Backtracking SE


def NewQNewtonPlusSE():
    
    print("33. New Q Newton's method Backtracking for Systems of Equations:")
    
    z00=z00_old
    z00_1=z00_old
    delta=1
    time0=time.time()
    #print(z00)
    for m in range(NIterate):
        #print("Iteration=",m)
        tmp=fHessian(z00)
        evals, evecs=LA.eig(tmp)
        evals=evals.real
        evecs=evecs.real
        Der=fDer(z00)
        if dimN == 1:
            HessTr = tmp
            #print(HessTr)
            if abs(HessTr)==0:
                HessTr=HessTr+delta*cutoff(f(z00))
            #print(HessTr)
            
            HessInv = 1 / HessTr
            #print(HessInv)
            gn = Der * HessInv
            gn=NegativeOrthogonalDecomposition(HessTr,gn,dimN)
        elif dimN > 1:
            HessTr = tmp.transpose()
            #print(HessTr)
            #print(f26Der(z00))
            if abs(LA.det(HessTr))==0:
                HessTr=HessTr+delta*cutoff(f(z00))*np.identity(dimN,dtype=float)
            #print(HessTr)
            #HessInv = LA.inv(HessTr)
            #print(HessInv)
            gn=0
            for i in range(dimN):
                gn=gn+(np.matmul(Der,evecs[:,i])/abs(evals[i]))*evecs[:,i]
    
        gamgam=1
        gnNorm=np.dot(gn,Der)
        while (f(z00-gamgam*gn)>f(z00)-0.5*gamgam*gnNorm):
            gamgam=gamgam/2
        z00=z00-gamgam*gn
    
        if verbose:
            print(f(z00))
            #print(z00)
        
        if stopCriterion == 0:
            if (L2Norm2(Der, dimN) < atol) :  # stop when meet the relative and absolute tolerances
                break
        elif stopCriterion == 2:
            if abs(f(z00)-f(zmin))<errtol:
                break
        if L2Norm2(z00-z00_1,dimN)<aatol:
            break
        z00_1=z00
    time1=time.time()
    print("time=",time1-time0)
    print("m=", m)
    #print(z00)
    if printLastPoint==True:
        print("The last point =",z00)
    if verbose==False:
        print("function value =", f(z00))
    print("norm of gradient=", math.sqrt(L2Norm2(fDer(z00),dimN)))
    if printHessianEigenvalues==True:
        w,v=LA.eig(fHessian(z00))
        #print(fDer(z00_old))
        print("Eigenvalues of Hessian=", w)
    #print(fDer(z00))
    return





######## Note: For the function f39 (taken from the paper by Rodomanov-Nesterov), cc is a mmxN matrix, entries randomly chosen in [-1,1], bb is a mm matrix, entries randomly chosen in [-1,1], the initial point z00_old is randomly chosen with length 1/N. The parameters for the function are bb, cc and ggam.




D=2
mm=1

dimN=D


bound=5

ggam=1




#verbose=True
#stopCriterion=1

printLastPoint=True

printHessianEigenvalues=True

verbose=False
stopCriterion=0
#stopCriterion=0
#stopCriterion=2

NIterate=10000
#NIterate=10000

print("atol=", atol)
aatol=1e-20

#z00_rand=np.array([-1+np.random.rand()*2*1 for _ in range(dimN)])
#z00_old=z00_rand/(math.sqrt(L2Norm2(z00_rand,dimN)*dimN))
#z00_old=z00_rand

#z00_old=np.array([np.random.uniform(-1,1)*math.pi for _ in range(dimN)])

#z00_old=np.array([3 for _ in range(dimN)])
z00_old=np.array([20*np.random.rand()-10 for _ in range(dimN)])
#z00_old=np.array([9.76536427, -4.15647151])

Evect=np.zeros((dimN,dimN))
for i in range(dimN):
    Evect[i][i]=1



bb=np.array([-1+np.random.rand()*2*1 for _ in range(mm)])

cc1=np.random.rand(mm,dimN)
cc2=cc1*2
cc=cc2-1

rrtol=1e-10


print("the function is", "CostFunctionR")
f=CostFunctionR
fDer=CostFunctionRDer
fHessian=CostFunctionRHessian



    
#z00_old, xi=f46Initialization()
#xi=np.array([1,1,1,-1,1])
#print("Matrix xi=",xi)
#tmp=f46(z00_old)
#print("f46 value=",tmp)



#z00InPaper=np.array([0,0.29723*math.pi,0.33306*math.pi,0.62176*math.pi,0])
#z00InPaper=np.array([0,0.61866*math.pi,0])
#print("Optimum according to the paper=", f46(z00InPaper))

#x1=np.random.rand()*2*math.pi
#x2=0
#x3=np.random.rand()*2*math.pi
#z00_old=np.array([x1,x2,x3])


#print("angleMatrix=",angleMatrix)

#print("angleMatrixCos=", angleMatrixCos)

#print("angleMatrixSin= ",angleMatrixSin)

#z00_old=np.array([-2.903534+0.3, -2.903534-0.8])

# For the function f40, bb takes values in {-1,1}, and bb and cc and z00_old are taken from real datasets: #https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

#z00_old=np.array([-2.903534+0.3, -2.903534-0.8])

#z00_old=np.array([0.26010457, -10.91803423, 2.98112261, -15.95313456,  -2.78250859, -0.77467653,  -2.02113182,   9.10887908, -10.45035903,  11.94967756, -1.24926898,  -2.13950642,   7.20804014,   1.0291962,    0.06391697, 2.71562242, -11.41484204,  10.59539405,  12.95776531,  11.13258434,
 #  8.16230421, -17.21206152,  -4.0493811,  -19.69634293,  14.25263482, 3.19319406,  11.45059677,  18.89542157,  19.44495031,  -3.66913821])

#z00_old=np.array([-0.15359941, -0.59005902, 0.45366905, -0.94873933,  0.52152264, -0.02738085,0.17599868,  0.36736119,  0.30861332,  0.90622707,  0.10472251, -0.74494753, 0.67337336, -0.21703503, -0.17819413, -0.14024491, -0.93297061,  0.63585997, -0.34774991, -0.02915787, -0.17318147, -0.04669807,  0.03478713, -0.21959983,
 # 0.54296245,  0.71978214, -0.50010954, -0.69673303,  0.583932,   -0.38138978, -0.85625076,   0.20134663, -0.71309977, -0.61278167,  0.86638939,  0.45731164, -0.32956812,  0.64553452, -0.89968231,  0.79641384,  0.44785232,  0.38489415, -0.51330669,  0.81273771, -0.54611157, -0.87101225, -0.72997209, -0.16185048, 0.38042508, -0.63330049,  0.71930612, -0.33714448, -0.24835364, -0.78859559,
 #-0.07531072,  0.19087508, -0.95964552, -0.72759281,  0.13079216,  0.6982817, 0.54827214,  0.70860856, -0.51314115, -0.54742142,  0.73180924, -0.28666226, 0.89588517,  0.35797497, -0.21406766, -0.05558283,  0.89932563, -0.16479757, -0.29753867,  0.5090385,   0.95156811,  0.8701501,   0.62499125, -0.22215331, 0.8355082,  -0.83695582, -0.96214862, -0.22495384, -0.30823426,  0.55635375,
 # 0.38262606, -0.60688932, -0.04303575,  0.59260985,  0.5887739,  -0.00570958, -0.502354, 0.50740011, -0.08916369,  0.62672251,  0.13993309, -0.92816931, 0.50047918,  0.856543,    0.99560466, -0.44254687])

print("Number of iterates=", NIterate)

print("initial point=", z00_old)
#print("The matrix xi=",xi)

## For some problems, such as f39 and f40, we also use the minimum point, for another stopping criterion
zmin=np.zeros(D)
#errtol=rrtol*abs(f(z00_old)-f(zmin))
#print("errtol=",errtol)

print("function value at initial point=", f(z00_old))

print("Value of the system of equations at initial point", SystemReal(z00_old))

print("Value of the Jacobian of the system of equations at initial point", JacobianR(z00_old))

print("Value of the gradient of the cost function at initial point", fDer(z00_old))

print("Value of Jacobian multiplied with the system at initial point ", np.matmul(JacobianR(z00_old).transpose(),SystemReal(z00_old)))

print("Value of the Hessian of the cost function at initial point", fHessian(z00_old))


#print("derivative at the initial point=", fDer(z00_old))

#printHessianEigenvalues=True
#tmp=fHessian(z00_old)
#w,v=LA.eig(fHessian(z00_old))
#print("Eigenvalues of the Hessian at the initial point=", w)

#print(type(z00_old))
#print(len(z00_old))

#NewtonMethod()
#NewtonMethodSE()
LevenbergMarquardtBacktracking()
LevenbergMarquardt()
NewQNewtonPlus()
NewQNewtonPlusSE()


#RandomNewtonMethod()

NewQNewtonG2()
NewQNewtonPlusV1G2()
NewQNewtonPlusV2G2()
NewQNewtonPlusV3G2()
NewQNewtonPlusV4G2()

#BFGS()

#UnboundedTwoWayBacktrackingGD()
#BacktrackingGD()
#InertialNewtonM()
#AdaptiveCubicRegularisation()
#TwoWayBacktrackingGD()



#LocalNewtonMethod()
#LocalRandomNewQNewton()
#LocalRandomNewtonMethod()
#LocalNewQNewton()

#LocalInertialNewtonM()
#LocalBacktrackingGD()

#LocalTwoWayBacktrackingGD()
#LocalUnboundedTwoWayBacktrackingGD()
#LocalBFGS()
#LocalAdaptiveCubicRegularisation()
#AdaptiveCubicRegularisation()

#zshift=np.full(D,0,dtype=np.float128)
#zsquare=np.full(D,0,dtype=np.float128)
#for i in range(D-1):
#    zshift[i]=i+1
#for i in range(D-1):
#    zsquare[i]=i*i
#print(zshift)
#print(zsquare)

#f23G=nd.Gradient(f24)
#f23H=nd.Hessian(f24)
#print(f23G(z00))
#print(f23H(z00))

#z00=np.array([1,1,1])
#print(z00)
#print(f26(z00))
#print(f26Der(z00))
#print(f26Hessian(z00))

