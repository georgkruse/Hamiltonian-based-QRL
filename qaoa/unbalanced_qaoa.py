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

problems = {}

# Problem instance with 3 nodes
problems["3"] = {}
problems['3']["values"] = [5,3,19]
problems['3']["weights"] = [3,7,25]
problems['3']["maximum_weight"] = 11

#Problem instance with 4 nodes
problems["4"] = {}
problems['4']["values"] = [3,5,2,10]
problems['4']["weights"] = [10,3,7,9]
problems['4']["maximum_weight"] = 20

# Problem instance with 5 nodes
problems["5"] = {}
problems['5']["values"] = [1,10,5,20,17]
problems['5']["weights"] = [5,9,10,15,3]
problems['5']["maximum_weight"] = 21

# Problem instance with 6 nodes
problems["6"] = {}
problems['6']["values"] = [20,16,5,9,10,11]
problems['6']["weights"] = [5,4,10,16,1,17]
problems['6']["maximum_weight"] = 32

# Problem instance with 7 nodes
problems["7"] = {}
problems['7']["values"] = [20,10,4,18,5,9,18]
problems['7']["weights"] = [15,10,3,17,9,10,17]
problems['7']["maximum_weight"] = 37

# Problem instance with 8 nodes
problems["8"] = {}
problems['8']["values"] = [3,5,10,12,22,11,4,19]
problems['8']["weights"] = [20,15,10,11,3,7,9,16]
problems['8']["maximum_weight"] = 30

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
    cases = np.arange(3,9)

    results_qaoa = {"unbalanced":{}, "slack":{}}
    lambdas = [0.96,0.0371]

    for problem_size in cases:
        values = problems[f"{problem_size}"]["values"]
        weights = problems[f"{problem_size}"]["weights"]
        maximum_weight = problems[f"{problem_size}"]["maximum_weight"]

        #results_qaoa["unbalanced"][problem_size] = {}
        #results_qaoa["slack"][problem_size] = {}


        mdl = KP(values, weights, maximum_weight)
        sol_str = solve_knapsack(mdl)

        results_qaoa["unbalanced"][problem_size] = qaoa_results(lambdas,mdl,sol_str,p = 3,maxiter = 100,unbalanced = True)
        results_qaoa["slack"][problem_size] = qaoa_results(lambdas,mdl,sol_str,p = 3,maxiter = 100,unbalanced = False)
    
    np.save("results_qaoa_comparing.npy", results_qaoa)

    prop = "probability"
    prop_dict = {"slack":{}, "unbalanced":{}}
    for problem_size in cases:
        prop_dict["slack"][problem_size] = []
        prop_dict["unbalanced"][problem_size] = []
        prop_dict["slack"][problem_size] = results_qaoa["slack"][problem_size][prop]
        prop_dict["unbalanced"][problem_size] = results_qaoa["unbalanced"][problem_size][prop]
    
    fig, ax = plt.subplots(1,1, figsize=(6,3.5))

    ax.plot(range(3,9), list(prop_dict["unbalanced"].values()), label="unbalanced", marker="o", markeredgecolor="black", markersize=8)
    ax.plot(range(3,9), list(prop_dict["slack"].values()), label="slack", marker="^", markeredgecolor="black", markersize=8)
    
    ax.set_xlabel("Problem Size")
    ax.set_xticks(range(3,9))
    ax.grid()
    ax.set_ylabel(f"{prop}")
    ax.legend(loc='upper center')
    ax.set_yscale("log")
    fig.savefig("unbalanced_vs_slack.png",dpi=500, bbox_inches="tight")