import pennylane as qml
from pennylane import qaoa

def circuit_qaoa(params, layer, wires, H, H_mixer, type='exp'):
    '''
    Baseline QAOA circuit.
    '''
    for w in range(wires):
        qml.Hadamard(wires=w)
    idx = 0
    for _ in range(layer):
        qaoa.cost_layer(params[idx], H)
        qaoa.mixer_layer(params[idx+1], H_mixer)
        idx += 2
    if type=='exp':
        return qml.expval(H)
    elif type=='probs':
        return qml.probs(wires=range(wires))
