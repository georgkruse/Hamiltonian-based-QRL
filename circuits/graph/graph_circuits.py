import torch
import numpy as np
import pennylane as qml
import torch


def graph_encoding_block(config, theta, weights, layer, type=None):

    if (type == 'actor' or type == 'critic'):
        params_input = weights[f'input_scaling_{type}'][layer]
        params_var = weights[f'weights_{type}'][layer]

    if isinstance(theta['quadratic_0'], torch.Tensor):
        indexing_quadratic = theta['quadratic_0'][:,:,:2].detach().numpy()
        indexing_linear = theta['linear_0'][:,:,0].detach().numpy()
    else:
        indexing_quadratic = theta['quadratic_0'][:,:,:2]
        indexing_linear = theta['linear_0'][:,:,0]

    quadratic_gates = {'zz': qml.IsingZZ, 'xx': qml.IsingXX, 'yy':qml.IsingYY}
    linear_gates = {'rz': qml.RZ, "rx":qml.RX, 'ry':qml.RY}
    linear_gate = linear_gates[config["linear_gate"].lower()]
    quadratic_gate = quadratic_gates[config["quadratic_gate"].lower()]
    annotations_gate = linear_gates[config["annotations_gate"].lower()]
        
    if not indexing_quadratic[0,0,0] + indexing_quadratic[0,0,1] == 0:

        x, z = 0, 0
        for h_num in range(1):
            if config['graph_encoding_type'] == 's-ppgl':
                for idx in range(theta[f'quadratic_{h_num}'].shape[1]):
                    quadratic_gate(theta[f'quadratic_{h_num}'][:,idx,2]*params_input[x], wires=[indexing_quadratic[0,idx,0],indexing_quadratic[0,idx,1]])
                
                for idx in range(theta[f'linear_{h_num}'].shape[1]):
                    linear_gate(theta[f'linear_{h_num}'][:,idx,1]*params_input[x], wires=indexing_linear[0,idx])

                x += 1
                if 'annotations' in theta.keys():
                    for idx in range(theta[f'linear_{h_num}'].shape[1]):
                        annotations_gate(theta['annotations'][:,idx,1]*params_var[z], wires=indexing_linear[0,idx])
                else:
                    for idx in range(theta[f'linear_{h_num}'].shape[1]):
                        annotations_gate(params_var[z], wires=indexing_linear[0,idx])
                z += 1
            

            elif config['graph_encoding_type'] == 's-ppgl-linear':
                for idx in range(theta[f'linear_{h_num}'].shape[1]):
                    linear_gate(theta[f'linear_{h_num}'][:,idx,1]*params_input[x], wires=indexing_linear[0,idx])
                x += 1

                if "annotations" in theta.keys():
                    for idx in range(theta[f'linear_{h_num}'].shape[1]):
                        annotations_gate(theta['annotations'][:,idx,1]*params_var[z], wires=indexing_linear[0,idx])
                else:
                    for idx in range(theta[f'linear_{h_num}'].shape[1]):
                        annotations_gate(params_var[z], wires=indexing_linear[0,idx])
                z += 1


            elif config['graph_encoding_type'] == 's-ppgl-quadratic':
                for idx in range(theta[f'quadratic_{h_num}'].shape[1]):
                    quadratic_gate(theta[f'quadratic_{h_num}'][:,idx,2]*params_input[x], wires=[indexing_quadratic[0,idx,0],indexing_quadratic[0,idx,1]])
                x += 1
                
                for idx in range(theta[f'linear_{h_num}'].shape[1]):
                    linear_gate(params_var[z], wires=indexing_linear[0,idx])
                z += 1


            elif config['graph_encoding_type'] == 'm-ppgl':
                for idx in range(theta[f'quadratic_{h_num}'].shape[1]):
                    quadratic_gate(theta[f'quadratic_{h_num}'][:,idx,2]*params_input[x], wires=[indexing_quadratic[0,idx,0],indexing_quadratic[0,idx,1]])
                    x += 1
                
                for idx in range(theta[f'linear_{h_num}'].shape[1]):
                    linear_gate(theta[f'linear_{h_num}'][:,idx,1]*params_input[x], wires=indexing_linear[0,idx])
                    x += 1
                
                if "annotations" in theta.keys():
                    for idx in range(theta[f'linear_{h_num}'].shape[1]):
                        annotations_gate(theta['annotations'][:,idx,1]*params_var[z], wires=indexing_linear[0,idx])
                        z += 1
                else:
                    for idx in range(theta[f'linear_{h_num}'].shape[1]):
                        annotations_gate(params_var[z], wires=indexing_linear[0,idx])
                        z += 1

            elif config['graph_encoding_type'] == 'm-ppgl-linear':  
                for idx in range(theta[f'linear_{h_num}'].shape[1]):
                    linear_gate(theta[f'linear_{h_num}'][:,idx,1]*params_input[x], wires=indexing_linear[0,idx])
                    x += 1
                
                if "annotations" in theta.keys():
                    for idx in range(theta[f'linear_{h_num}'].shape[1]):
                        annotations_gate(theta['annotations'][:,idx,1]*params_var[z], wires=indexing_linear[0,idx])
                        z += 1
                else:
                    for idx in range(theta[f'linear_{h_num}'].shape[1]):
                        annotations_gate(params_var[z], wires=indexing_linear[0,idx])
                        z += 1


            elif config['graph_encoding_type'] == 'm-ppgl-quadratic':
                for idx in range(theta[f'quadratic_{h_num}'].shape[1]):
                    quadratic_gate(theta[f'quadratic_{h_num}'][:,idx,2]*params_input[x], wires=[indexing_quadratic[0,idx,0],indexing_quadratic[0,idx,1]])
                    x += 1

                if "annotations" in theta.keys():
                    for idx in range(theta[f'linear_{h_num}'].shape[1]):
                        annotations_gate(theta['annotations'][:,idx,1]*params_var[z], wires=indexing_linear[0,idx])
                        z += 1
                else:
                    for idx in range(theta[f'linear_{h_num}'].shape[1]):
                        annotations_gate(params_var[z], wires=indexing_linear[0,idx])
                        z += 1


            elif config['graph_encoding_type'] == 'h-ppgl':
                for idx in range(theta[f'quadratic_{h_num}'].shape[1]):
                    quadratic_gate(theta[f'quadratic_{h_num}'][:,idx,2]*params_input[x], wires=[indexing_quadratic[0,idx,0],indexing_quadratic[0,idx,1]])
                    x += 1
                
                for idx in range(theta[f'linear_{h_num}'].shape[1]):
                    linear_gate(theta[f'linear_{h_num}'][:,idx,1]*params_input[x], wires=indexing_linear[0,idx])
                    x += 1
                
                if "annotations" in theta.keys():
                    for idx in range(theta[f'linear_{h_num}'].shape[1]):
                        annotations_gate(theta['annotations'][:,idx,1]*params_var[z], wires=indexing_linear[0,idx])
                else:
                    for idx in range(theta[f'linear_{h_num}'].shape[1]):
                        annotations_gate(params_var[z], wires=indexing_linear[0,idx])
                z += 1

            elif config['graph_encoding_type'] == 'hamiltonian-hwe':
                for idx in range(theta[f'quadratic_{h_num}'].shape[1]):
                    quadratic_gate(theta[f'quadratic_{h_num}'][:,idx,2], wires=[indexing_quadratic[0,idx,0],indexing_quadratic[0,idx,1]])
                
                for idx in range(theta[f'linear_{h_num}'].shape[1]):
                    linear_gate(theta[f'linear_{h_num}'][:,idx,1], wires=indexing_linear[0,idx])


            elif config['graph_encoding_type'] in ['angular', 'angular-hwe']:

                for idx in range(theta[f'linear_{h_num}'].shape[1]):
                    
                    qml.RY(theta[f'linear_{h_num}'][:,idx,1]*params_input[x], wires=indexing_linear[0,idx])
                    if "annotations" in theta.keys():
                        qml.RZ(theta[f'annotations'][:,idx,1]*params_input[x+1], wires=indexing_linear[0,idx])
                    else:
                        qml.RZ(params_input[x+1], wires=indexing_linear[0,idx])

                    x += 2
                
                # for idx in range(theta[f'linear_{h_num}'].shape[1]):
                #     if 'a_0' in theta.keys():
                #         qml.RX(theta['a_0'][:,idx,1], wires=indexing_linear[0,idx])
    else:
        # Do this operation during ray test to enforce batch dimension
        qml.RY(theta[f'linear_0'][:,0,1], wires=0)


