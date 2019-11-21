#!/usr/bin/env python
# coding: utf-8

# # A Specific Integer Quadratic Programming with [$\text{Branch-and-Bound}$](https://en.wikipedia.org/wiki/Branch_and_bound) Method
# 
# ### *The core idea is to solve relaxed continuous quadratic programming problems with repeatedly tightened variable boundaries until integer solutions reached. The optimal solution is the integer solution producing the best optimal value.*
# 
# ### Goal: Find the optimal rounded values that is closest to a vector $c$
# ### Specific Integer Quadratic Problem:
# 
# $\quad \min_{x} \ \ x^{T}x - 2c^{T}x$ <br>
# $\quad s.t. \quad Ax <= b$ <br>
# $\quad \quad\quad \ \ x^{T}I=1$<br>
# $\quad \quad\quad \ \ x^{T}x-0.02x>=\mathbf{0}$<br>
# $\quad \quad \quad \ \ lowerbound <= x <= upperbound$ <br>
# $\quad and \quad x_{j} \in Z,j = 1,2,3,\cdots,n$ <br>
# $\quad where \ x,c,lowerbound,upperbound \in{R^{n\times{1}}}, A \in{R^{m\times{n}}}, b \in{R^{m\times{1}}},Z \ is \ some \ discrete \ set \ with \ equal \ step \ size $

# ### Key Mathematical Operations:
# * $N^{i}:=\{c^{i},A^{i},b^{i},lowerbound^{i},upperbound^{i}\}$: $\text{A Specific Form Problem mentioned above}$
# * $\mathbb{S}^{i} := \{ x|x^{T}I=1 ,A^{i}x<=b^{i},lowerbound^{i}<=x<=upperbound^{i}\}$: $\text{continuous feasible region for } N^{i}$
# * $QP(x,c,A,b,lowerbound,upperbound)$: $\text{solver of Specific Form Integer Quadratic Programming without integer constraint by}$ [$\text{SLSQP}$](https://en.wikipedia.org/wiki/Sequential_quadratic_programming)
# * $x^{i} := QP(N^{i})$: $\text{optimal solution of} \ N^{i} \text{without integer constraints}$
# * $z^{i} := (x^{i})^{T}x^{i} - 2(c^{i})^{T}x^{i} \ where \ c^{i} \in N^{i}$ :$\text{optimal value}$
# * $J^{i} := \{ j|x_{j}^{i} \notin{Z} \} \ where \ x_{j}^{i} \in{x^{i}}$
# * $x^{i+} := \{ x_{j}^{i+}|j \in J^{i},x_{j}^{i} \in{x^{i}},x_{j}^{i+}=\min{(\{z|z>=x_{j}^{i},z \in{Z} \}})$
# * $x^{i-} := \{ x_{j}^{i-}|j \in J^{i},x_{j}^{i} \in{x^{i}},x_{j}^{i-}=\max{(\{z|z<=x_{j}^{i},z \in{Z} \}})$

# ### Algorithm:
# $Given: \ c,A,b,lowerbound,upperbound$<br>
# $N^{0}=\{ c^{0}=c,A^{0}=A,b^{0}=b,lowerbound^{0}=lowerbound,upperbound^{0}=upperbound \}$<br>
# $\mathcal{L}=\{N^{0}\}$<br>
# $z^{*} = inf, x^{*}=None$<br>
# $while \ |\mathcal{L}| > 0:$<br>
# $\quad Select \ and \ delete \ the \ last \ Node ,N^{i}, from \ \mathcal{L}$<br>
# $\quad if \ x^{i} \notin{\mathbb{S}^{i}} \ or \ z^{i}>z^{*}:$<br>
# $\quad \quad next$<br>
# $\quad else:$<br>
# $\quad \quad if \ x_{j}^{i} \in{Z} \ and \ x^{i} \in{\mathbb{S}^{0}}:$<br>
# $\quad \quad \quad z^{*}=z^{i}$<br>
# $\quad \quad \quad x^{*}=x^{i}$<br>
# $\quad \quad \quad \mathcal{L}=\{ N^{n}|z^{n}<z^{*},all \ Node^{n} \ in \ \mathcal{L} \}$<br>
# $\quad \quad else:$<br>
# $\quad \quad \quad Select \ j,j \in{J^{i}}$<br>
# $\quad \quad \quad lowerbound^{updated} = lowerbound^{i}$<br>
# $\quad \quad \quad lowerbound^{updated}_{j} = x_{j}^{i+}$<br>
# $\quad \quad \quad upperbound^{updated} = upperbound^{i}$<br>
# $\quad \quad \quad upperbound^{updated}_{j} = x_{j}^{i-}$<br>
# $\quad \quad \quad N_{i+}=\{ c_{i+}=c^{i},A_{i+}=A^{i},b_{i+}=b^{i},lowerbound_{i+}=lowerbound_{updated},upperbound_{i+}=upperbound^{i} \}$<br>
# $\quad \quad \quad N_{i-}=\{ c_{i-}=c^{i},A_{i-}=A^{i},b_{i-}=b^{i},lowerbound_{i-}=lowerbound^{i},upperbound_{i-}=upperbound_{updated} \}$<br>
# $\quad \quad \quad Add \ N_{i+},N_{i-} \ to \ \mathcal{L}$<br>
# $return \ z^{*},x^{*}$

# ### <span style="color:red">$\text{The algorithm introduced here should be easily modified to solve the general integer programming problem}$</span>
# 
# $\quad \min_{x} \ \ x^{T}Sx - 2c^{T}x$ <br>
# $\quad s.t. \quad Ax <= b$ <br>
# $\quad \quad \quad \ \ Ex = f$ <br>
# $\quad \quad \quad \ \ x^{T}Qx - p^{T}x >= r$ <br>
# $\quad \quad \quad \ \ lowerbound <= x <= upperbound$ <br>
# $\quad and \quad x_{j} \in{Z} , j = 1,2,3,\cdots,n$

# In[1]:


import numpy as np
from scipy import optimize as opt
import random
import math


# In[2]:


class computing:
    '''
    This class is mainly to construct the solver for continuous quadratic programming problem with linear constraints. 
    '''
    
    def L2_norm_diff(x,origin):
        '''
        Compute the L2-norm of the difference between two vectors
        
        Parameters:
        -----------
        x: numpy array, one-dimensional
        origin: numpy array, one-dimensional
        *** x and origin share same size
        
        Returns:
        --------
        float: L2-norm
        '''
        return np.linalg.norm(x-origin,ord=2)

    def L1_norm_diff(x,origin):
        '''
        Compute the L1-norm of the difference between two vectors
        
        Parameters:
        -----------
        x: numpy array, one-dimensional
        origin: numpy array, one-dimensional
        *** x and origin share same size
        
        Returns:
        --------
        float: L1-norm
        '''
        return np.linalg.norm(x-origin,ord=1)
    
    def quad_obj_func(x,c,A=None):
        '''
        Compute the result of a typical quadratic objective function in matrix form
        f(x,c,A) = transpose(x) * A * x - 2 * transpose(c) * x
        
        Parameters:
        -----------
        x: numpy array, one-dimensional
        c: numpy array, one-dimensional
        A: numpy array/matrix, a valid positive semidefinite matrix with number of rows same as x/c 
                               A is identity matrix if A is not provided
        Returns:
        --------
        float: f(x,c,A)
        '''
        if A is None:
            A = np.eye(x.size)
        return x.dot(A).dot(x) - 2*c.dot(x)
    
    @classmethod
    def qp_solver(cls,origin,A=None,b=None,upperbound=None,lowerbound=None):
        '''
        This is the solver for continuous quadratic programming problem with constraints.
        
        min f(x,c,A) = transpose(x) * x - 2 * transpose(c) * x
        st  A * x <= b
            x(x-0.02)>=0
            lowerbound <= x <= upperbound
            transpose(x)*I = 1
        where c = origin, and I is identity vector
        
        
        Parameters:
        -----------
        origin: numpy array, one-dimensional, initial guess
        A: numpy array, two-dimensional with number of columns equal to size of origin, constraints matrix
        b: numpy array, one-dimensional with number of element equal to rows of A, upperbound of constraints
        upperbound: numpy array, one-dimensional, same size as origin, upperbound for decision variables
        lowerbound: numpy array, one-dimensional, same size as origin, lowerbound for decision variables
        
        Returns:
        --------
        float: optimal solution
        
        '''

        if A is None and b is None:
            constraints = []
        else:
            constraints = [{'type': 'ineq', 'fun': lambda x:  b - np.matmul(A,x)}]

        constraints.append({'type':'eq','fun': lambda x: np.sum(x)-1})
        constraints.append({'type':'ineq','fun': lambda x: x*(x-0.02)})

        if upperbound is None:
            upperbound = 1
        if lowerbound is None:
            lowerbound = 0

        bounds = opt.Bounds(lowerbound,upperbound)

        result = opt.minimize(cls.quad_obj_func,origin.copy(),args=(origin,),bounds=bounds,constraints=constraints,method='SLSQP')['x']
            
        return result


# In[3]:


class prune_check:
    '''
    This class is to define functions checking if the following three conditions met for a given one-dimensional array, x
    1. Element in x is consistent with the required style
    2. x is within the feasible region, which means linear constraints respected and x within its boundaries
    3. objective value derived from x is larger than a given value
    '''

    style = None
    tol = None
    
    @classmethod
    def settol(cls,tol):
        '''
        tol: None or float,tolerance for accuracy
        '''
        cls.tol = tol
    
    @classmethod
    def setstyle(cls,style):
        '''
        style: str, '{int}d{int}', number of decimal places and last digit divisor
        '''
        cls.style = style
        dp,unit = [int(i) for i in style.split('d')]
        if cls.tol is None:
            cls.tol = min(10**-(dp+3),1e-5)
    
    @classmethod
    def getstyle(cls):
        if cls.style is None:
            print('A style of solution should be given')
        return cls.style
    
    @classmethod
    def floor_base(cls,number):
        style = cls.style
        dp,unit = [int(i) for i in style.split('d')]
        tmp = math.floor(number*10**dp)

        number = tmp/10**dp
        last_digit = int(str(tmp)[-1])

        if last_digit%unit == 0:
            pass
        else:
            number = number - (last_digit%unit)/10**dp
        return round(number,dp)
        
    @classmethod
    def ceil_base(cls,number):
        '''
        Return the smallest float with 2 decimal places bigger than the given number
        '''
        style = cls.style
        dp,unit = [int(i) for i in style.split('d')]
        increment = unit*10**-dp
        return round(cls.floor_base(number)+increment,dp) 
    
    @classmethod
    def integrality_base(cls,number):
        '''
        Return True if number is with only 2 decimal places else False
        '''
        return any([number - cls.floor_base(number)<cls.tol,cls.ceil_base(number) - number<cls.tol])
    
    @classmethod
    def floor(cls,array):
        '''
        Vectorized floor_base
        floor_base: Return the largest float with 2 decimal places smaller than the given number
        '''
        return np.vectorize(cls.floor_base)(array)
    @classmethod
    def ceil(cls,array):
        '''
        Vectorized ceil_base
        ceil_base: Return the smallest float with 2 decimal places bigger than the given number
        '''
        return np.vectorize(cls.ceil_base)(array)
    @classmethod  
    def integrality(cls,array):
        '''
        Vectorized integrality_base
        Return True if number is with only 2 decimal places else False
        '''
        return np.vectorize(cls.integrality_base)(array)
    
    @classmethod
    def infeasiblity(cls,array,A,b,upperbound,lowerbound):
        '''
        Check if linear constaints are respected and solution is within boundaries.
        A * array <= b
        lowerbound <= x <= upperbound
        
        Paramters:
        ----------
        array: numpy array, one-dimensional, solution vector
        A: numpy array, constraint matrix
        b: numpy array, constraint upperbound
        upperbound: numpy array, upperbound for decision variables
        lowerbound: numpy array, lowerbound for decision variables
        
        Returns:
        --------
        boolean
        
        '''
        
        lb_check = np.all(array-lowerbound >= -cls.tol) # x >= lowerbound 
        ub_check = np.all(upperbound-array >= -cls.tol) # x <= upperbound 
        if A is None and b is None:
            ineq_check = True # special case when no constraints given
        else:
            ineq_check = np.all(np.matmul(A,array)-b <= cls.tol) # Ax <= b
        eq_check = abs(np.sum(array)-1) < cls.tol # sum(x) == 1
        if all([lb_check,ub_check,ineq_check,eq_check]): # all conditions met --> not infeasible;else --> infeasible
            return False
        else:
            return True
    
    def bounds(obj_val,best_obj_val):
        '''
        Check if the objective value is bigger than the given best-found objective value
        '''
        return obj_val > best_obj_val


# In[4]:


class Node:
    '''
    Each Node represents a continuous quadratic programming problem with linear constraints.
    min f(x,c,A) = transpose(x) * x - 2 * transpose(c) * x
        st  A * x <= b
            lowerbound <= x <= upperbound
            transpose(x)*I = 1
        where c = origin, and I is identity vector
    '''
    
    def __init__(self,origin,A,b,lowerbound,upperbound,best_obj_val):
        '''
        Parameters:
        -----------
        origin: numpy array, one-dimensional, initial guess
        A: numpy array, two-dimensional with number of columns equal to size of origin, constraints matrix
        b: numpy array, one-dimensional with number of element equal to rows of A, upperbound of constraints
        upperbound: numpy array, one-dimensional, same size as origin, upperbound for decision variables
        lowerbound: numpy array, one-dimensional, same size as origin, lowerbound for decision variables
        best_obj_val: the so-far-best-found objective value(inf at the beginning)
        '''
        self.origin = origin
        self.A = A
        self.b = b
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.best_obj_val = best_obj_val
        
        self.sol = computing.qp_solver(self.origin,self.A,self.b,self.upperbound,self.lowerbound)
        self.obj_val = computing.quad_obj_func(self.sol,self.origin)
        
        #if np.any(np.isnan(self.sol)):
        #    self.is_infeasible = True
        #    self.is_bounded = False
        #    self.is_integral = False
        #else:
        self.is_infeasible = self.infeasiblity_check()
        self.is_bounded = self.bounds_check()
        self.is_integral = all(self.integral_check())
        
        self.is_end_node = self.check_end_node()
    
    def check_end_node(self):
        '''
        Check if one of the three ending conditions met
        '''
        return any([self.is_infeasible,self.is_bounded,self.is_integral])
    
    def infeasiblity_check(self):
        '''
        Check if the solution is infeasible
        '''
        return prune_check.infeasiblity(self.sol,self.A,self.b,self.upperbound,self.lowerbound)
    
    def bounds_check(self):
        '''
        Check if the objective value is worse than the so-far-best-found one
        '''
        return prune_check.bounds(self.obj_val,self.best_obj_val)
    
    def integral_check(self):
        '''
        Check if the solution is only with 2-decimal-place element
        '''
        return prune_check.integrality(self.sol)


# In[5]:


def optimal_rounding(origin,A,b,lowerbound,upperbound,style,tol=None,max_iter=10000,disp=False):
    '''
    Find the optimal 2-decimal-place/integer solution for the quadratic programming with linear constraints
    using Branch-and-Bound method through tightening boundaries for decision variables repeatedly to find solution with only
    integer elements until the global optima found or maximum iteration reached. 
    
    Parameters:
    -----------
    origin: numpy array, one-dimensional, initial guess
    A: numpy array, two-dimensional with number of columns equal to size of origin, constraints matrix
    b: numpy array, one-dimensional with number of element equal to rows of A, upperbound of constraints
    lowerbound: numpy array, one-dimensional, same size as origin, lowerbound for decision variables
    upperbound: numpy array, one-dimensional, same size as origin, upperbound for decision variables
    style: str, '{int}d{int}', number of decimal places and last digit divisor
    tol: float, accuracy tolerance. if not provided, then it would be inferred from style 
    max_iter: int, maximum iteration number
    disp: boolean, print out the subproblem for each loop if True
    
    Return:
    -------
    best_sol: numpy array, optimal solution
    best_obj_val: float, optimal value
    
    '''
    prune_check.settol(tol)
    prune_check.setstyle(style)
    root_Node = Node(origin,A,b,lowerbound.copy(),upperbound.copy(),np.inf) # initial Node
    Node_list = [root_Node] # add initial Node to the tracking list

    best_sol = None # initial so-far-best-found solution
    best_obj_val = np.inf # initial so-far-best-found optimal value
    
    Node_num = len(Node_list) # number of element in the tracking list
    num_iter = 0 # number of iterations
    while Node_num > 0 and num_iter < max_iter: # loop if tracking list is not empty and maximum iteration number not reached
        current_Node = Node_list.pop() # delete the last Node and explore this Node
        # End Node if no feasible solution found or the optimal value is worse than the so-far-best-found
        if current_Node.is_bounded or current_Node.is_infeasible:
            pass 
        else:
            # End Node if the optimal solution is feasible and contain only integer element
            if current_Node.is_integral and not prune_check.infeasiblity(current_Node.sol,A,b,upperbound,lowerbound):
                best_obj_val = current_Node.obj_val
                best_sol = current_Node.sol
                Node_list = [n for n in Node_list if n.obj_val < best_obj_val]
            # Split one decision variable's boundaries into two parts and construct two new continuous QP problems
            else:
                not_int_var = np.where(~current_Node.integral_check())[0] # identify non-integer decision variables
                i = random.choice(not_int_var) # randomly pick up one non-integer decision variable to further explore

                ceil = prune_check.ceil(current_Node.sol) # smallest number bigger than the optimal solution
                floor = prune_check.floor(current_Node.sol) # biggest number smaller than the optimal solution
                
                
                current_upperbound = current_Node.upperbound.copy() 
                current_lowerbound = current_Node.lowerbound.copy()
                #current_sol = current_Node.sol.copy()
                
                
                # update upperbound for the selected decision variable
                left_upperbound = current_upperbound.copy()
                left_upperbound[i] = floor[i]
                left_lowerbound = current_lowerbound.copy()
                
                # update lowerbound for the selected decision variable
                right_upperbound = current_upperbound.copy()
                right_lowerbound = current_lowerbound.copy()
                right_lowerbound[i] = ceil[i]
                
                if disp: # print updated information
                    print('var: ',i)
                    print('current sol: ',current_Node.sol)
                    print('floor[i]: ',floor[i])
                    print('ceil[i]: ',ceil[i])
                    print('current bound: ',[current_lowerbound,current_upperbound])
                    print('left bound: ',[left_lowerbound,left_upperbound])
                    print('right bound: ',[right_lowerbound,right_upperbound])

                # check if decision variables have feasible boundaries
                if all(left_lowerbound <= left_upperbound): 
                    left_Node = Node(current_Node.origin,A,b,left_lowerbound,left_upperbound,best_obj_val) # new Node
                    Node_list.append(left_Node) # add new Node to tracking list 
                if all(right_lowerbound <= right_upperbound):
                    right_Node = Node(current_Node.origin,A,b,right_lowerbound,right_upperbound,best_obj_val)
                    Node_list.append(right_Node)

        Node_num = len(Node_list)
        num_iter += 1
        if num_iter == max_iter:
            print('Max Iteration Exceed')
    return best_obj_val,best_sol




