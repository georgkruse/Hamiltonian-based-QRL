"""
author:rodrigo.coelho@iisb.fraunhofer.de
"""

"""
This script uses OpenQAOA to train QAOA models for each problem instance in a dataset of KP problem instances.
"""


from docplex.mp.model import Model
from openqaoa.problems import FromDocplex2IsingModel
from collections import defaultdict
import pennylane as qml
import numpy as np
from openqaoa import QAOA
import matplotlib.pyplot as plt

def KP(values: list, weights: list, maximum_weight: int):
    """
    Crete a Docplex model of the Knapsack problem:
    
    args:
        values - list containing the values of the items
        weights - list containing the weights of the items
        maximum_weight - maximum weight allowed in the "backpack"
    """

    mdl = Model("Knapsack")
    num_items = len(values)
    x = mdl.binary_var_list(range(num_items), name = "x")
    cost = -mdl.sum(x[i] * values[i] for i in range(num_items))
    mdl.minimize(cost)
    mdl.add_constraint(
        mdl.sum(x[i] * weights[i] for i in range(num_items)) <= maximum_weight
    )
    return mdl

def qaoa_results(penalty, mdl, sol_str, p=1, t=0.1, maxiter=100, unbalanced=False):
    """
    Solving a docplex model using QAOA in OpenQAOA
    
    penalty: The penalty terms for the inequality constraints.
    
    """
    ising_hamiltonian = FromDocplex2IsingModel(mdl,
                                   unbalanced_const=unbalanced,
                                   strength_ineq=[penalty[0],penalty[1]]).ising_model

    qaoa = QAOA() # OpenQAOA QAOA model
    qaoa.set_circuit_properties(p=p, init_type="ramp", linear_ramp_time=t) # initialization betas and gammas with a ramp technique
    qaoa.set_classical_optimizer(maxiter=maxiter) 
    qaoa.compile(ising_hamiltonian)
    qaoa.optimize()
    nq = ising_hamiltonian.n
    results = qaoa.result.lowest_cost_bitstrings(2**nq)
    results["openqaoa"] = qaoa.result
    for nn, string in enumerate(results["solutions_bitstrings"]):
        if string[:len(sol_str)] == sol_str:
            pos = nn
            break
    results["opt_pos"] = pos # Optimal position in the sorted Hamiltonian
    results["probability"] = results["probabilities"][results["opt_pos"]] # Probability of the optimal solution
    results["classical_sol"] = sol_str # classical solution
    results["CoP"] = results["probability"] * 2 ** nq # Coefficient of Performance https://arxiv.org/abs/2211.13914
    results["Emin"] = np.min(results["bitstrings_energies"]) # minimum eigenvalue
    results["Emax"] = np.max(results["bitstrings_energies"]) # maximum eigenvalue
    results["Ef"] = np.min(qaoa.result.optimized["cost"]) # current average energy
    results["r"] = (results["Ef"] - results["Emax"])/(results["Emin"] - results["Emax"]) # Approximation ratio
    results["opt_angles"] = qaoa.result.optimized["angles"] # Optimal betas and gammas 
    results["num_qubits"] = nq
    results["ising_hamiltonian"] = ising_hamiltonian
    results["cx"] = 2 * len([t for t in ising_hamiltonian.terms if len(t) == 2]) * p
    return results

def solve_knapsack(mdl):
    docplex_sol = mdl.solve()
    solution = ""
    for ii in mdl.iter_binary_vars():
        solution += str(int(np.round(docplex_sol.get_value(ii),1)))
    return solution


if __name__ == "__main__":
    # The size of the instances to be solved
    cases = [3,4,5,6,7,8,9,10]

    results_qaoa = {"unbalanced":{}, "slack":{}}

    #The lambdas for the unbalanced penalization
    lambdas = [0.96,0.0371]

    path_to_dataset = "/home/users/coelho/quantum-computing/QRL/games/knapsack/KP_dataset.npy"

    dataset = np.load(path_to_dataset, allow_pickle=True).item()
    dataset_size = 100

    for problem_size in cases:

        values = []
        weights = []
        maximum_weight = []

        results_qaoa["unbalanced"][problem_size] = {}
        results_qaoa["slack"][problem_size] = {}

        for i in range(dataset_size):
            values.append(dataset[f"{problem_size}"]["validation"][f"{i}"]["values"])
            weights.append(dataset[f"{problem_size}"]["validation"][f"{i}"]["weights"])
            maximum_weight.append(dataset[f"{problem_size}"]["validation"][f"{i}"]["maximum_weight"])


        mdls = [KP(values[i], weights[i], maximum_weight[i]) for i in range(dataset_size)]
        sol_strs = [solve_knapsack(mdls[i]) for i in range(dataset_size)]

        for i in range(dataset_size):
            results_qaoa["unbalanced"][problem_size][f"{i}"] = {}
            results_qaoa["slack"][problem_size][f"{i}"] = {}
            results_qaoa["unbalanced"][problem_size] = qaoa_results(lambdas,mdls[i],sol_strs[i],p = 3,maxiter = 100,unbalanced = True)
            results_qaoa["slack"][problem_size] = qaoa_results(lambdas,mdls[i],sol_strs[i],p = 3,maxiter = 100,unbalanced = False)
    
    np.save("qaoa_dataset.npy", results_qaoa)