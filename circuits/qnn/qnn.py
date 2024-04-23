import pennylane as qml

def circuit_qnn(params, layer, wires, linear, quadratic):
    '''
    Simple QNN circuit to benchmark against QAOA, VQE or QRL on supervised learning task.
    '''
    x = 0
    for w in range(wires):
        qml.Hadamard(wires=w)

    for _ in range(layer):
        for key, value in quadratic.items():
            qml.CNOT(wires=[int(key[0]), int(key[1])])
            qml.RZ(value*params[x], wires=int(key[1]))
            qml.CNOT(wires=[int(key[0]), int(key[1])])
            x += 1
        
        for key, value in linear.items():
            qml.RZ(value*params[x], wires=int(key))
            x += 1
        
        for idx in range(wires):
            qml.RX(params[x], wires=idx)
            x += 1

    return [qml.expval(qml.PauliZ(i)) for i in range(wires)]



            