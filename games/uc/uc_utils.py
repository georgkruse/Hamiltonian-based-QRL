import os
import pennylane as qml
import numpy as np
import pandas as pd
from pyqubo import Binary

def kazarlis_uc(n, path):
    '''
    Function that reads the dataset from the kazarlis file and a few parameters.
    
    Input:
    n                   - Number of Units
    
    Output:
    p_min               - List of Minimum Power Outputs of the units
    p_max               - List of Maximum Power Outputs of the units
    A                   -   List of Binary Costs of Units producing power
    B                   -   List of Linear Costs of Units producing power
    C                   -   List of Quadratic Costs of units producing power
    '''
    
    dataset = path + '/games/uc/data/kazarlis_units_10.csv'
    df = pd.read_csv(dataset,sep=',')
    p_min = df['min_output'].values[:n]
    p_max = df['max_output'].values[:n]
    A = df['a'].values[:n]
    B = df['b'].values[:n]
    C = df['c'].values[:n]

    return p_min, p_max, A, B, C

def objective_uc_qubo(y,A, B, C, L, lambda_1, n, constants, eq_bits= None):
    '''
    Function that formulates the binary unit commitment problem in an qubo problem.
    It returns the cost function consisting of the calculation of the cost and the penalty
    for producing too much or too little power.
    
    Input:
    y                   -   Parameters, that need to be optimized (e.g. binary variables)
    A                   -   Fixed Cost of unit producing power
    B                   -   Linear cost of unit producting a certain ammount of power
    C                   -   Quadratic cost of unit producting a certain ammount of power
    labmda_1            -   Factor for weighting the penalty of over and underproduction
    n                   -   number of units (without reduction)
    constants           -   constants with fixed power output of Units
    eq_bits             -   Pandas dict containing the bits that get reduced by recursive QAOA;
                            use empty list for normal behavior
    
    Ouput:
    QUBO                -   QUOB formulation of the cost function
    '''
    if type(eq_bits) == type(None):
        eq_bits = pd.DataFrame(columns=["org", "project", "factor"])

    p = constants[:n]
    list_obj = []
    list_const = []
    for i in range(n):
        if i in eq_bits.org.values:
            row_vals = eq_bits.loc[eq_bits['org'] == i].values[0]
            idx = int(row_vals[1])
            if row_vals[2] == -1:
                factor = 1
            else:
                factor = 0 
        else:
            idx = i
            factor = 0
        # Generating Bit-Flips with the term (y-1)**2
        # list_obj.append((A[i] + B[i]*p[i] + C[i]*p[i]**2)*(y[idx]-factor)**2)
        # list_obj.append((A[i] + B[i]*p[i])*(y[idx]-factor)**2)
        list_const.append(p[i]*(y[idx]-factor)**2)
    objective = 0 #sum(list_obj)
    constraint_1 = lambda_1*((sum(list_const) - L)**2)
    QUBO = objective + constraint_1
    return QUBO