import pennylane as qml

# Specify the ciruit here or import it
def custom_circuit(theta, weights, config, H):
    for i in range(10):
        qml.Hadamard(wires=i)
    h_num = 0
    x = 0
    for _ in range(1):
        for idx in range(theta[f'quadratic_{h_num}'].shape[1]):
            qml.CNOT(wires=[int(theta[f'quadratic_{h_num}'][0,idx,0]), int(theta[f'quadratic_{h_num}'][0,idx,1])])
            qml.RZ(theta[f'quadratic_{h_num}'][0,idx,2]*weights[x], wires=int(theta[f'quadratic_{h_num}'][0,idx,1]))
            # qml.RZ(theta[f'quadratic_{h_num}'][0,idx,2]*2.0, wires=int(theta[f'quadratic_{h_num}'][0,idx,1]))
            # qml.RZ(theta[f'quadratic_{h_num}'][0,idx,2], wires=int(theta[f'quadratic_{h_num}'][0,idx,1]))
            qml.CNOT(wires=[int(theta[f'quadratic_{h_num}'][0,idx,0]), int(theta[f'quadratic_{h_num}'][0,idx,1])])
        x += 1
        for idx in range(theta[f'linear_{h_num}'].shape[1]):
            # qml.RZ(theta[f'linear_{h_num}'][0,idx,1]*weights[x], wires=int(theta[f'linear_{h_num}'][0,idx,0]))
            qml.RZ(theta[f'linear_{h_num}'][0,idx,1], wires=int(theta[f'linear_{h_num}'][0,idx,0]))
        # x += 1
        
        # for idx in range(theta[f'linear_{h_num}'].shape[1]):
        #     qml.RX(weights[x], wires=int(theta[f'linear_{h_num}'][0,idx,0]))
        for idx in range(theta[f'linear_{h_num}'].shape[1]):
            qml.RX(0.01, wires=int(theta[f'linear_{h_num}'][0,idx,0]))
        # x += 1          

    return [qml.expval(qml.PauliZ(i)) for i in range(5)]
    # return [qml.expval(H)]