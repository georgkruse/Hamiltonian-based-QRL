import pennylane as qml

def pooling_block(theta, weights, type, blocks, wires):

    for idx in blocks:
        z = 0
        for i in range(wires):
            qml.RY(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
            op = qml.RZ(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
            z += 2

        z = 0
        for i in range(wires):                
            qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
            qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
            z += 2

        for i in range(wires):
            x = i
            xx = x+1
            if i == wires - 1:
                xx = 0
            qml.CNOT(wires=[x,xx])

def pooling_lunar_01(theta, weights, config, type, **kwargs):
    for i in range(config['num_qubits']):
        qml.Hadamard(wires=i)        
    if type == 'actor':
        pooling_block(theta, weights, type, config['blocks'][0][0], config['num_qubits'])
        pooling_block(theta, weights, type, config['blocks'][0][1], config['num_qubits']-1)
        pooling_block(theta, weights, type, config['blocks'][0][2], config['num_qubits']-2)
        return [qml.expval(qml.PauliZ(i)) for i in range(config['num_qubits']-2)]
    elif type == 'critic':
        pooling_block(theta, weights, type, config['blocks'][1][0], config['num_qubits'])
        pooling_block(theta, weights, type, config['blocks'][1][1], config['num_qubits']-1)
        pooling_block(theta, weights, type, config['blocks'][1][2], config['num_qubits']-2)
        return [qml.expval(qml.PauliZ(i)) for i in range(config['num_qubits']-2)]
    
def pooling_lunar_02(theta, weights, config, type, **kwargs):
    for i in range(config['num_qubits']):
        qml.Hadamard(wires=i)        
    if type == 'actor':
        pooling_block(theta, weights, type, config['blocks'][0][0], config['num_qubits'])
        pooling_block(theta, weights, type, config['blocks'][0][1], config['num_qubits']-1)
        pooling_block(theta, weights, type, config['blocks'][0][2], config['num_qubits']-2)
        return [qml.expval(qml.PauliZ(i)) for i in range(config['num_qubits']-2)]
    elif type == 'critic':
        pooling_block(theta, weights, type, config['blocks'][1][0], config['num_qubits'])
        pooling_block(theta, weights, type, config['blocks'][1][1], config['num_qubits']-2)
        pooling_block(theta, weights, type, config['blocks'][1][2], config['num_qubits']-4)
        return [qml.expval(qml.PauliZ(i)) for i in range(config['num_qubits']-4)]
    
        
def pooling_pendulum_triple(theta, weights, config, type, **kwargs):

    for i in range(config['num_qubits']):
        qml.Hadamard(wires=i)        

    for idx in config['blocks'][0][0]:
        z = 0
        for i in range(config['num_qubits']):
            qml.RY(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
            qml.RZ(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
            z += 2

        z = 0
        for i in range(config['num_qubits']):                
            qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
            qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
            z += 2

        for i in range(config['num_qubits']):
            x = i
            xx = x+1
            if i == config['num_qubits'] - 1:
                xx = 0
            qml.CNOT(wires=[x,xx])
    
    for idx in config['blocks'][0][1]:
        z = 0
        for i in range(6):
            qml.RY(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
            qml.RZ(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
            z += 2

        z = 0
        for i in range(6):                
            qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
            qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
            z += 2
    
        for i in range(6):
            x = i
            xx = x+1
            if i == 6 - 1:
                xx = 0
            qml.CNOT(wires=[x,xx])
    
    for idx in config['blocks'][0][2]:
        z = 0
        for i in range(3):
            qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
            qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
            z += 2

        z = 0
        for i in range(3):                
            qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
            qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
            z += 2
    
        for i in range(3):
            x = i
            xx = x+1
            if i == 3 - 1:
                xx = 0
            qml.CNOT(wires=[x,xx])
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

def pooling_pendulum_double(theta, weights, config, type, **kwargs):

    for i in range(config['num_qubits']):
        qml.Hadamard(wires=i)        

    for idx in config['blocks'][0][0]:
        z = 0
        for i in range(config['num_qubits']):
            qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
            qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
            z += 2

        z = 0
        for i in range(config['num_qubits']):                
            qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
            qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
            z += 2

        for i in range(config['num_qubits']):
            x = i
            xx = x+1
            if i == config['num_qubits'] - 1:
                xx = 0
            qml.CNOT(wires=[x,xx])
    
    
    for idx in config['blocks'][0][1]:
        z = 0
        for i in range(3):
            qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
            qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
            z += 2

        z = 0
        for i in range(3):                
            qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
            qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
            z += 2
    
        for i in range(3):
            x = i
            xx = x+1
            if i == 3 - 1:
                xx = 0
            qml.CNOT(wires=[x,xx])
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]
    

      
    

    


