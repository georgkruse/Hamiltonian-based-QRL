import pennylane as qml
import numpy as np
from itertools import product

dev = qml.device("lightning.qubit", wires = 4)

@qml.qnode(dev)
def quantum_circuit(state,params_input,quadratic_gate,linear_gate):
    for i in range(4):
        qml.Hadamard(wires = i)

    x = 0
    num_layers = 5

    for layer in range(num_layers):

        for idx in range(state[f'quadratic'].shape[0]):
            if quadratic_gate == "ZZ":
                qml.CNOT(wires=[state['quadratic'][idx,0], state["quadratic"][idx,1]])
                qml.RZ(state[f'quadratic'][idx,2]*params_input[x], wires=state["quadratic"][idx,1])
                qml.CNOT(wires=[state['quadratic'][idx,0], state["quadratic"][idx,1]])
            elif quadratic_gate == "XX":
                qml.IsingXX(state[f'quadratic'][idx,2]*params_input[x],wires = [state['quadratic'][idx,0], state["quadratic"][idx,1]])
            elif quadratic_gate == "YY":
                qml.IsingYY(state[f'quadratic'][idx,2]*params_input[x],wires = [state['quadratic'][idx,0], state["quadratic"][idx,1]])
            else:
                pass
        x += 1

        for idx in range(state[f'a'].shape[0]):
            if linear_gate == "Z":
                qml.RZ(state[f'a'][idx,1]*params_input[x], wires=state["a"][idx,0])
            elif linear_gate == "X":
                qml.RX(state[f'a'][idx,1]*params_input[x], wires=state["a"][idx,0])
            elif linear_gate == "Y":
                qml.RY(state[f'a'][idx,1]*params_input[x], wires=state["a"][idx,0])
        x += 1
    
    return qml.expval(qml.PauliZ(0))

state = {}
state["quadratic"] = np.array([[0,1,2],[0,3,2],[1,2,2],[2,3,2]])
state["a"] = np.array([[0,np.pi],[1,np.pi],[2,np.pi],[3,np.pi]])
state["linear"] = np.array([[0,2],[1,1],[2,0.5],[3,3]])

#quadratic_gate = "None"
#linear_gate = "Y"
#results = []
#for i in range(1000):
#    params = np.random.uniform(0, 2*np.pi, 10)
#    results.append(quantum_circuit(state,params, quadratic_gate, linear_gate))
#print(np.var(results))


quadratic_gates = ["ZZ","XX","YY"]
linear_gates = ["Z","X","Y"]
combinations = list(product(quadratic_gates, linear_gates))

results = [[] for _ in range(len(combinations))]


for i in range(200):
    for i,combination in enumerate(combinations):
        params = np.random.randn((10))
        results[i].append(quantum_circuit(state,params,combination[0],combination[1]))

for i, combination in enumerate(combinations):
    print(f"Quadratic Gates: {combination[0]}, Linear Gate: {combination[1]}, Expectation Value Variance: {np.var(results[i])}")



