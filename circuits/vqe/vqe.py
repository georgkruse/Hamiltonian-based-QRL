import pennylane as qml

def circuit_vqe(params, layer, wires, H, type='exp'):
    '''
    Baseline VQE circuit.
    '''
    for i in range(wires):
        qml.Hadamard(wires=i)
    x = 0
    for l in range(layer):
        for i in range(wires):
            qml.RX(params[x], wires=i)
            x +=1
        for i in range(wires):
            qml.RZ(params[x], wires=i)
            x +=1
        for i in range(wires):
            if i < wires-1:
                qml.CNOT(wires=[i,i+1])
    if type=='exp':
        return [qml.expval(H)]
    elif type=='probs':
        return qml.probs(wires=range(wires))

