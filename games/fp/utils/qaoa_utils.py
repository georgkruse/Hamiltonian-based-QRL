import numpy as np
from itertools import permutations
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import Aer, BasicAer, transpile
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import QAOA, VQE, NumPyMinimumEigensolver
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.converters import circuit_to_gate
from qiskit_optimization.algorithms import CplexOptimizer, GurobiOptimizer
from docplex.mp.model import Model
from qiskit.algorithms.optimizers import COBYLA, SLSQP

from scipy.optimize import minimize


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# self-made histogram function (includes possibilities to only plot a certain number of results)
def generate_histogram(counts, figsize=(15, 6), dpi=100, color='midnightblue', mode='probs', num_to_keep = 5):
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        counts = dict(sorted(counts.items(), key=lambda x:-x[1]))
        sum_heights = np.sum(list(counts.values()))
        keys = list(counts.keys())[:min(len(counts), num_to_keep)]
        if len(keys)>10:
            r = 70
        else:
            r = 0
        heights = np.array(list(counts.values()), dtype=float)[:min(len(keys), num_to_keep)]
        if mode=='probs':
            heights = heights/sum_heights
            f = '{:.3f}'
            ylabel = 'probability'
        elif mode=='counts':
            f = '{:.0f}'
            ylabel = 'counts'
        ax.bar(keys, heights, color=color, align='center')
        fs = 6
        if min(num_to_keep, len(heights))<=5:
            fs = 'x-small'
        
        for x,y in zip(keys,heights):
            label = f.format(y)

            plt.annotate(label, # this is the value which we want to label (text)
                         (x,y), # x and y is the points location where we have to label
                         textcoords="offset points",
                         xytext=(0,3), # this for the distance between the points
                         # and the text label
                         ha='center', fontsize=fs, rotation=r)#,
                         #arrowprops=dict(arrowstyle=".", color='black'))
        

def get_problem(problem_number):
    if problem_number==1:
        n = 2
        C = [[0,4],[2,1]]
        T = [[1,3],[2,1]]
        
    elif problem_number==2:
        n = 3
        C = [[2,1,5],[1,3,2],[1,2,7]]
        T = [[1,3,1],[4,0,3],[1,3,4]]

    elif problem_number==3:
        n = 2
        C = [[1,2],[4,1]]
        T = [[5,3],[7,1]]

    elif problem_number==4:
        n = 2
        C = [[4,1],[4,8]]
        T = [[5,6],[7,2]]
    
    return n, C, T

    # calculate the objective/cost function of a bitsring
def calc_obj(bitstring, n, C, T, A):
    bitstring = bitstring[::-1] # result bitstring needs to be reversed (first qubit corrseponds to last bit in bitstring)
    bs = np.fromiter(map(int, bitstring), dtype=int)
    obj_sum = 0
    
    # cost function
    for q in range(n):
        for p in range(n):
            for j in range(n):
                for i in range(n):
                    Cij = C[i][j]
                    Tpq = T[p][q]
                    #obj_sum += Cij*Tpq*((1-bs[n*p+i])*(1-bs[n*q+j])/4)
                    obj_sum += Cij*Tpq*bs[n*p+i]*bs[n*q+j]
    # now the constraints
    for i in range(n):
        p_sum = 0
        for p in range(n):
            p_sum += bs[n*p+i]
        obj_sum += (A* (1-p_sum)**2)
            
    
    for p in range(n):
        i_sum = 0
        for i in range(n):
            i_sum += bs[n*p+i]
        obj_sum += (A* (1-i_sum)**2)
        
    return obj_sum
# initial (equal superposition) state which is gorund state of H_B
def prepare_initial_state(n):
    qc = QuantumCircuit(n**2)
    for i in range(n**2):
        qc.h(i)
    return circuit_to_gate(qc, label='$\psi(0)$')

# mixer hamiltonian: exp(-i beta H_B)
def create_mixing_hamiltonian(n):
    beta = Parameter('$\\beta$')
    qc = QuantumCircuit(n**2, name='$exp(-i\\beta H_B)$')
    for qubit in range(n**2):
        qc.rx(2*beta, qubit)
    return qc

# cost hamiltonian: exp(-i gamma H_C)
def create_cost_hamiltonian(n, C, T, A):
    def qubit_num(p, i):
        return n*p+i
    gamma = Parameter('$\\gamma$')
    qc = QuantumCircuit(n**2, name='$exp(-i\\gamma H_C)$') #need n^2 qubits to represent the x_ij=0/1 variables
    # all the terms commute so we can write the sum in the Hamiltonian as a simple product in the unitary
    for q in range(n):
        for p in range(n):
            for j in range(n):
                for i in range(n):
                    Cij = C[i][j]
                    Tpq = T[p][q]
                    qc.rz(-0.5*gamma*Cij*Tpq, qubit_num(p, i))
                    qc.rz(-0.5*gamma*Cij*Tpq, qubit_num(q, j))
                    if not qubit_num(p, i)==qubit_num(q, j):
                        qc.rzz(0.5*gamma*Cij*Tpq, qubit_num(p, i), qubit_num(q, j))
    # now the constraints
    for p in range(n):
        for i in range(n):
            qc.rz(-2*gamma*A*(n-2), qubit_num(p, i))
    for i in range(n):
        for p in range(n):
            for q in range(p):
                qc.rzz(gamma*A, qubit_num(p, i), qubit_num(q, i))
    for p in range(n):
        for i in range(n):
            for j in range(i):
                qc.rzz(gamma*A, qubit_num(p, i), qubit_num(p, j))
    return qc

    
def generate_qaoa_circuit(n, C, T, A, p):
    num_qubits = n**2
    qc = QuantumCircuit(num_qubits)
     # initial state
    init_state_gate = prepare_initial_state(n)
    qc.append(init_state_gate, range(num_qubits))
    
    gammas = ParameterVector('$\\gamma_i$', length=p)
    betas = ParameterVector('$\\beta_i$', length=p)
    # layers
    cfc = create_cost_hamiltonian(n, C, T, A)
    mc = create_mixing_hamiltonian(n)
    for rep in range(p):
        
        cfc2 = cfc.assign_parameters([gammas[rep]])
        mc2 = mc.assign_parameters([betas[rep]])
        qc.append(cfc2, range(num_qubits))
        qc.append(mc2, range(num_qubits))
        
    qc.measure_all()
    
    return qc

def compute_cf(x, n, C, T):
    obj_sum = 0
    for q in range(n):
        for p in range(n):
            for j in range(n):
                for i in range(n):
                    Cij = C[i][j]
                    Tpq = T[p][q]
                    obj_sum += Cij*Tpq*x[p][i]*x[q][j]
    return obj_sum

def get_minimum(n, C, T):
    all_valid_values = []
    all_valid_vectors = []
    current_minimum=np.inf
    current_x = None
    for perm in permutations(range(n)):
        x = np.zeros((n, n))
        for e in range(n):
            x[e][perm[e]]=1
        objsum = compute_cf(x, n, C, T)
        all_valid_values.append(objsum)
        all_valid_vectors.append(x)
        if objsum <current_minimum:
            current_minimum = objsum
            current_x = x
    return current_minimum, current_x

from collections import OrderedDict
from qiskit.utils import algorithm_globals
from qiskit.algorithms import QAOA
from qiskit.opflow import StateFn

def sample_most_likely(state_vector):
    """Compute the most likely binary string from state vector.
    Args:
        state_vector (numpy.ndarray or dict): state vector or counts.
    Returns:
        numpy.ndarray: binary string as numpy.ndarray of ints.
    """
    if isinstance(state_vector, (OrderedDict, dict)):
        # get the binary string with the largest count
        binary_string = sorted(state_vector.items(), key=lambda kv: kv[1])[-1][0]
        x = np.asarray([int(y) for y in reversed(list(binary_string))])
        return x
    elif isinstance(state_vector, StateFn):
        binary_string = list(state_vector.sample().keys())[0]
        x = np.asarray([int(y) for y in reversed(list(binary_string))])
        return x
    else:
        n = int(np.log2(state_vector.shape[0]))
        k = np.argmax(np.abs(state_vector))
        x = np.zeros(n)
        for i in range(n):
            x[i] = k % 2
            k >>= 1
        return x

# calculate the objective/cost function of a bitsring
def calculate_objective_quantum(bitstring, n, N, P_min, P_max, A, B, C, L):

    bitstring = bitstring[::-1] # result bitstring needs to be reversed (first qubit corrseponds to last bit in bitstring)
    bs = np.fromiter(map(int, bitstring), dtype=int)
    mdl = Model('docplex model')
    v = bs[:n]
    z = bs[n:]
    z = np.reshape(z, (n,N+1))
     # Definition of the variables to solve for.
    h = [(P_max[i] - P_min[i])/N for i in range(n)]
    p = [[] for _ in range(n)]
        
    for i in range(n):
        p[i] = mdl.sum((P_min[i] + (k-1)*h[i])*z[i][k] for k in range(N+1))
        z_ik = [z[i][k] for k in range(N+1)]
        mdl.add_constraint(v[i] + mdl.sum(z_ik) == 1)  

    mdl.add_constraint(mdl.sum(p) == L)         
    mdl.minimize(mdl.sum(A[i]*(1-v[i]) + B[i]*p[i] + C[i]*p[i]**2 for i in range(n)))
    return mdl.solve()


# calculate the objective/cost function of a bitsring
def calculate_objective_hybrid(bitstring, n, N, P_min, P_max, A, B, C, L, p):

    bitstring = bitstring[::-1] # result bitstring needs to be reversed (first qubit corrseponds to last bit in bitstring)
    bs = np.fromiter(map(int, bitstring), dtype=int)
    # print(bs)
    # print(bitstring)
    obj_sum = 0
    mdl = Model('docplex model')
    h = [(P_max[i] - P_min[i])/N for i in range(n)]
    v = bs[:n]
    z = bs[n:]
    # p = [[] for _ in range(n)]

    # for i in range(n):
    #     p[i] = mdl.sum((P_min[i] + (k-1)*h[i])*z[i][k] for k in range(N+1))
    mdl.add_constraint(mdl.sum(bs[i]*p[i] for i in range(n)) >= L)
    mdl.minimize(mdl.sum(A[i]*(bs[i]) + B[i]*p[i] + C[i]*p[i]**2 for i in range(n)))
    return mdl.solve()

