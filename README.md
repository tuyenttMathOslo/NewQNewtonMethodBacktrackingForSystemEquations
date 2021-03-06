# NewQNewtonMethodBacktrackingForSystemEquations
Python code for the more direct use of New Q-Newton's method Backtracking for systems of equations.
This contains Python codes for the paper "Tuyen Trung Truong, A more direct and better variant of New Q-Newton's method Backtracking for m equations in m variables, arXiv:2110.07403".
I simplified the implementation, when choosing only 2 values for \delta (here, with values 0 and 1), while there should be m+1 such \delta when you are working with dimension m.
Also, I checked only the condition that the concerned matrix is invertible, while theoretical results require that we should check that the absolute values of the eigenvalues of the matrix are all "sufficiently large", see the paper for details.
However, so far the implementation works fine.
But if for some examples the implementation has singular matrix error (for the New Q-Newton's method Backtracking and variants) then you should have m+1 \delta's.
Another way to fix, should such an error rises, is to choose at each iterate a random value for the \delta. Algorithms to be compared include the Backtracking line search version of Levenberg-Marquardt method (see the paper for details), and other methods. 
The implementation also works for the more general problem of "least square fit" in statistics, which is to minimise a cost function of the form f=f_1^2+...+f_N^2.
The main file to run is "functionsDirectRun9.py".
