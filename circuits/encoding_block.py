import torch
import pennylane as qml

from circuits.graph.graph_circuits import graph_encoding_block

def encoding_block(config, theta, weights, layer, type=None):
    '''
    Creates the encoding block for the VQC.
    '''
    qubit_idx = 0
    
    if (type == 'actor' or type == 'critic') :
        params = weights[f'input_scaling_{type}'][layer]
    elif (type == 'es' or type == 'ga'):
        idx = int(config['num_layers']*config['num_qubits'])
        params = weights[2*idx:4*idx]
        params = params[layer*config['num_qubits']*config['num_scaling_params']:(layer+1)*config['num_qubits']*config['num_scaling_params']]
    
    if config['noise']['depolarizing'][0]:
        if config['encoding_type'] == 'angular_classical':
            if config['use_input_scaling']:
                for i in range(theta.shape[1]):
                    qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                    qml.DepolarizingChannel(config['noise']['depolarizing'][1], wires=i)
                    qml.RZ(theta[:,i]*params[qubit_idx+1], wires=i)
                    qml.DepolarizingChannel(config['noise']['depolarizing'][1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(theta.shape[1]):
                    qml.RY(theta[:,i], wires=i)
                    qml.DepolarizingChannel(config['noise']['depolarizing'][1], wires=i)
                    qml.RZ(theta[:,i], wires=i)
                    qml.DepolarizingChannel(config['noise']['depolarizing'][1], wires=i)
        else:
            print('ERROR: No encoding block selected.')

    else:
        if config['encoding_type'] == 'custom':
            if config['use_input_scaling']:
                qubit_idx = 0
                if layer % 2 == 0:
                    for i in range(5):
                        qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                        qml.RZ(theta[:,i]*params[qubit_idx+1], wires=i)
                        qubit_idx += 2
                else:
                    for i in range(5):
                        qml.RY(theta[:,i+5]*params[qubit_idx], wires=i)
                        qml.RZ(theta[:,i+5]*params[qubit_idx+1], wires=i)
                        qubit_idx += 2

        elif config['encoding_type'] == 'graph_encoding':
            graph_encoding_block(config, theta, weights, layer, type)

        elif config['encoding_type'] == 'angular_classical':
            if config['use_input_scaling']:
                for i in range(theta.shape[1]):
                    qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                    qml.RZ(theta[:,i]*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(theta.shape[1]):
                    qml.RY(theta[:,i], wires=i)
                    qml.RZ(theta[:,i], wires=i)       

        elif config['encoding_type'] == 'angular_classical_qubit':
            if config['use_input_scaling']:
                for i in range(config['num_qubits']):
                    qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                    qml.RZ(theta[:,i]*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(theta[:,i], wires=i)
                    qml.RZ(theta[:,i], wires=i)

        elif config['encoding_type'] == 'angular_times_2':
            if config['use_input_scaling']:
                for i in range(config['num_qubits']):
                    qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                    qml.RZ(2*theta[:,i]*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(theta[:,i], wires=i)
                    qml.RZ(2*theta[:,i], wires=i)
        
        elif config['encoding_type'] == 'angle_encoding_RX':
            if config['use_input_scaling']:
                for i in range(theta.shape[1]):
                    qml.RX(theta[:,i]*params[i],wires=i)
            else:
                for i in range(theta.shape[1]):
                    qml.RX(theta[:,i],wires=i)

        elif config['encoding_type'] == 'angle_encoding_RX_atan':
            if config['use_input_scaling']:
                for i in range(theta.shape[1]):
                    qml.RX(torch.arctan(theta[:,i]*params[i]),wires=i)
            else:
                for i in range(theta.shape[1]):
                    qml.RX(torch.arctan(theta[:,i]),wires=i)

        elif config['encoding_type'] == 'angular_arctan':
            if config['use_input_scaling']:
                for i in range(config['num_qubits']):
                    qml.RY(torch.arctan(theta[:,i]*params[qubit_idx]), wires=i)
                    qml.RZ(torch.arctan(theta[:,i]*params[qubit_idx+1]), wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(torch.arctan(theta[:,i]), wires=i)
                    qml.RZ(torch.arctan(theta[:,i]), wires=i)
        
        elif config['encoding_type'] == 'angular_arctan_ext':
            if config['use_input_scaling']:
                for i in range(config['num_qubits']):
                    qml.RY(torch.arctan(theta[:,i])*params[qubit_idx], wires=i)
                    qml.RZ(torch.arctan(theta[:,i])*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(torch.arctan(theta[:,i]), wires=i)
                    qml.RZ(torch.arctan(theta[:,i]), wires=i)

        elif config['encoding_type'] == 'angular_sigmoid':
            if config['use_input_scaling']:
                for i in range(config['num_qubits']):
                    qml.RY(torch.sigmoid(theta[:,i]*params[qubit_idx]), wires=i)
                    qml.RZ(torch.sigmoid(theta[:,i]*params[qubit_idx+1]), wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(torch.sigmoid(theta[:,i]), wires=i)
                    qml.RZ(torch.sigmoid(theta[:,i]), wires=i)

        elif config['encoding_type'] == 'angular_sigmoid_ext':
            if config['use_input_scaling']:
                for i in range(config['num_qubits']):
                    qml.RY(torch.sigmoid(theta[:,i])*params[qubit_idx], wires=i)
                    qml.RZ(torch.sigmoid(theta[:,i])*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(torch.sigmoid(theta[:,i]), wires=i)
                    qml.RZ(torch.sigmoid(theta[:,i]), wires=i)

        elif config['encoding_type'] == 'layerwise_arctan':
            if layer % 2 == 0:
                for i in range(config['num_qubits']):
                    qml.RY(torch.arctan(theta[:,i])*params[qubit_idx], wires=i)
                    qml.RZ(torch.arctan(theta[:,i])*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                    qml.RZ(theta[:,i]*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
        
        elif config['encoding_type'] == 'layerwise_sigmoid':
            if layer % 2 == 0:
                for i in range(config['num_qubits']):
                    qml.RY(torch.sigmoid(theta[:,i])*params[qubit_idx], wires=i)
                    qml.RZ(torch.sigmoid(theta[:,i])*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                    qml.RZ(theta[:,i]*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
        
        elif config['encoding_type'] == 'layerwise_sigmoid_arctan':
            if layer % 2 == 0:
                for i in range(config['num_qubits']):
                    qml.RY(torch.sigmoid(theta[:,i])*params[qubit_idx], wires=i)
                    qml.RZ(torch.sigmoid(theta[:,i])*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            elif layer % 3 == 0:
                for i in range(config['num_qubits']):
                    qml.RY(torch.arctan(theta[:,i])*params[qubit_idx], wires=i)
                    qml.RZ(torch.arctan(theta[:,i])*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                    qml.RZ(theta[:,i]*params[qubit_idx+1], wires=i)
                    qubit_idx += 2

        elif config['encoding_type'] == 'layerwise_arctan_sigmoid':
            if layer % 2 == 0:
                for i in range(config['num_qubits']):
                    qml.RY(torch.arctan(theta[:,i])*params[qubit_idx], wires=i)
                    qml.RZ(torch.arctan(theta[:,i])*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            elif layer % 3 == 0:
                for i in range(config['num_qubits']):
                    qml.RY(torch.sigmoid(theta[:,i])*params[qubit_idx], wires=i)
                    qml.RZ(torch.sigmoid(theta[:,i])*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                    qml.RZ(theta[:,i]*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
    
        else:
            print('ERROR: No encoding block selected.')