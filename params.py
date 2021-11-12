import numpy as np
from random import random

"""Default parameters for all gradient methods"""
rtol = 1e-10  # error tolerance
atol = rtol * rtol  # square of error tolerance
NIterate = 100000  # Number of iterates
alpha = 0.5  # alpha for Backtracking
beta = 0.7  # beta for Backtracking

n1 = 200  # Number of epochs in Hybrid Backtracking GD
n2 = NIterate // n1  # Number of iterations in each epoch for Hybrid Backtracking GD
p = 5  # Number of initial checks of Armijo's condition for Hybrid Backtracking GD in each epoch

delta0 = 1  # Initial learning rate for Backtraking GD
delta0N = 0.45  # Learning rate for Standard GD and the like

deltaMin = delta0N * 1.2  # maxLR for Cyclic Learning Rates
deltaMax = delta0N * 0.6  # baseLR for Cyclic Learning Rates
nn = 11  # nn-1 is the size of one cycle in the Cyclic Learning Rates. We will use a linear update scheme for learning rates, starting from baseLR.

gamma0 = 0.9  #
verbose = False  # If True, print detailed outputs when errors occur, where we will reduce NIterate to monitor. Used together with putting stopCriterion=2 and reducing NIterate.

choiceAdam = 1e-3  # the maximum value of delta0 in Adam
delta=1 # used in New Q-Newton's method

# Stopping Criteria.
# One is the usual one with both ||nabla f(z_n)|| and |f(z_n)-f(z_{n+1})| smaller than a threshold value.
# But for Rosenbrock functions, we use only abs(f(z_n)) as the sole criterion, to be compatible with other methods mentioned on Wikipedia.
# There are times when we want to check the real convergence behaviour, we do not put any criterion, but just let the concerned method to run NIterate iterations
# Use stopCriterion =0 if want to stop when ||nabla f|| and f(z_{n+1}-f(z_n)) small.
# Use stopCriterion=1 if want to stop when f(z_n) is small, for Rosenbruck function. Use otherwise if just want to run until NIterate.
# Value =0 if we want to stop when ||x_{n+1}-x_n||, ||\nabla f(x_n)|| and |f(x_{n+1})-f(x_n)| are all small.
# Value =1 if we want to stop when |f(x_n)| is small.(Usually use the later option for Rosenbruck function.)
# Value =2 if we don't want to put any stopping condition. Use together with putting verbose=True, and reducing NIterate.
stopCriterion = 0

#### Coefficients for applied functions
a = 1  # parameter for Rosenbrock function
b = 100  # parameter for Rosenbrock function
c6 = 1  # parameter for function f6
c9 = 1  # parameter for function f9
gamma =1/3  # parameter for cost functions f1 and f2
coef = 0  # parameter for function f11
sign1 = 1  # parameter for function f12
sign2 = 0  # parameter for function f12
ep = 1e-5  # parameter for functions f11, f12

thr=1e-20


#### Initial value for v0 in NAG
v0D1 = 1.385685437141149e-05  # 1 dimensional functions
v0D2 = np.array([1.385685437141149e-05, -0.00001188e-06])  # 2 dimensional functions
v0D3 = np.array([1.385685437141149e-05, -0.00001188e-06, 0.52268996e-06])  # 3 dimensional functions
v0D4 = np.array([1.385685437141149e-05, 0, 0.52268996e-06,
                 -0.81010244e-05])  # 4 dimensional functions

##### Initial points
z0f1 = 1.00001188
z0f2 = 1.00001188
z0f3 = 3
#z0f4 = 1.00001188
z0f4 = 0.55134554
#z0f4 = 0.75134554

z0f5 = 1.00001188
#z0f6 = np.array([-0.99998925, 2.00001188])
#z0f6 = np.array([0.55134554, 0.75134554])
z0f6 = np.array([4, -5])
#z0f7 = np.array([-0.99998925, 2.00001188])
z0f7 = np.array([0.55134554, 0.75134554])
z0f8C1 = 0.6
z0f8C2 = 0.8
z0f8C3 = 0.9
z0f9 = np.array([-0.99998925, 2.00001188])
z0f10 = np.array([-0.7020, 0.5342, -2.0101, 2.002])
z0f11 = 1.00100012
z0f12 = np.array([5.00001188, 1.00001188, 2.00001188])
#z0f13=np.array([-0.99998925, -2.00001188])
#z0f13=np.array([-0.99998925, 2.00001188])
#z0f13=np.array([-0.001, 0.1])
#z0f13=np.array([0.55134554, 0.75134554])
z0f13=np.array([0.55134554, -0.75134554])
z0f14=np.array([-0.99998925, 2.00001188])
z0f15 = 1.00001188
z0f16C1=2.35287527
z0f16C2=2.35284172
z0f16C3=2.35283735
z0f16C4=2.352836327
z0f16C5=-2.352836323
z0f17=0
z0f18=1.00001188
z0f19=np.array([1, 2])
z0f20=np.array([0.55134554, 0.75134554])
z0f21=np.array([0.55134554, 0.75134554])
z0f22 = np.array([0.00001188, 0.00002188, 0.00003188])
#z0f23=np.array([-0.99998925, -2.00001188])
#z0f23=np.array([-0.99998925, 2.00001188])
#z0f23=np.array([-0.001, 0.1])
#z0f23=np.array([0.55134554, 0.75134554])
#z0f23=np.array([0.55134554, -0.75134554])

z0f23=np.array([random() for _ in range(2)])

####################### Parameters for higher dimensional examples
#D=10
#v0D=np.array([random() for _ in range(D)])

#z0f24=np.array([-32+random()*64 for _ in range(D)])

D=3
v0D=np.array([random() for _ in range(D)])

bound=1

z0f24=np.array([-bound+random()*2*bound for _ in range(D)])



#### Different methods. Can use # to deactivate certain methods.
METHODS = [
            "NEWTON METHOD",
            "NEW Q NEWTON METHOD",
            "RANDOM NEW Q NEWTON METHOD",
           "RANDOM NEWTON METHOD",
           "INERTIAL NEWTON METHOD",
           "BFGS",
           #"LOCAL NEW Q NEWTON METHOD",
           #"LOCAL NEWTON METHOD"
           ]

#### The next three command lines are just for your convenience when you want to monitor should errors occur. Just delete the "#" and rerun the main.py file.

NIterate =3
#stopCriterion =2
verbose = True
