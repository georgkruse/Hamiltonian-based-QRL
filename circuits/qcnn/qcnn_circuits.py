import pennylane as qml
import numpy as np

def special_unitary(theta, weights, config, type, **kwargs):
    for i in range(config['num_qubits']):
        qml.Hadamard(wires=i)        
    
    idx = 0
    z = 0
    for i in range(config['num_qubits']):
        qml.RY(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
        qml.RZ(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
        z += 2

    z = 0
    for i in range(0, config['num_qubits'],2): 
        qml.SpecialUnitary(weights[f'weights_{type}_{idx}'][z:z+15], wires=[i, i+1])               
        z += 15
    
    for i in range(0, config['num_qubits'],2): 
        if i == 0:
            qml.SpecialUnitary(weights[f'weights_{type}_{idx}'][z:z+15], wires=[7, i])               
        else:
            qml.SpecialUnitary(weights[f'weights_{type}_{idx}'][z:z+15], wires=[i-1, i])               
        z += 15
    
    idx = 1
    z = 0
    for i in range(config['num_qubits']):
        qml.RY(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
        qml.RZ(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
        z += 2

    z = 0
     
    qml.SpecialUnitary(weights[f'weights_{type}_{idx}'][z:z+15], wires=[0, 2])               
    z += 15
    qml.SpecialUnitary(weights[f'weights_{type}_{idx}'][z:z+15], wires=[4, 6])               
    z += 15
    qml.SpecialUnitary(weights[f'weights_{type}_{idx}'][z:z+15], wires=[2, 4])               
    z += 15
       
    return [qml.expval(qml.PauliZ(i)) for i in range(0, config['num_qubits'], 2)]


    
def qcnn_pendulum_01(theta, weights, config, type, **kwargs):
    for i in range(config['num_qubits']):
        qml.Hadamard(wires=i)        
    
    if type == 'critic':
        for idx in range(2):
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
            if idx < 1:
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])

        m_0 = qml.measure(0)
        qml.cond(m_0, qml.PauliX)(wires=1)
        qml.CNOT(wires=[1,2])

        idx = 2
        z = 0
        for i in range(1,3):
            qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
            qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
            z += 2
        z = 0
        for i in range(1,3):                
            qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
            qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
            z += 2
        
        m_1 = qml.measure(1)
        qml.cond(m_1, qml.PauliX)(wires=2)
        idx = 3
        z = 0
        qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][-2], wires=2)
        qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][-1], wires=2)
        z = 0
        qml.RZ(weights[f'weights_{type}_{idx}'][-2], wires=2)
        qml.RY(weights[f'weights_{type}_{idx}'][-1], wires=2)
        
        return [qml.expval(qml.PauliZ(2))]


    else:
        for idx in range(3):
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

        return [qml.expval(qml.PauliZ(i)) for i in range(config['num_qubits'])]


def qcnn_lunar_01(theta, weights, config, type, **kwargs):

    for i in range(config['num_qubits']):
        qml.Hadamard(wires=i)        
    
    if type == 'actor':
        for idx in range(4):
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
        # https://discuss.pennylane.ai/t/mid-circuit-measurement-in-pennylane/3187
        # for i in range(0, config['num_qubits'], 2):
        #     m_c = qml.measure(i)
        #     qml.cond(m_c, qml.PauliX)(wires=i+1)
            
        # for idx in range(3,6):
        #     z = 0
        #     for i in range(1,config['num_qubits'],2):
        #         qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
        #         qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
        #         z += 2

        #     z = 0
        #     for i in range(1,config['num_qubits'],2):                
        #         qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
        #         qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
        #         z += 2
        
        #     qml.CNOT(wires=[1,3])
        #     qml.CNOT(wires=[3,5])
        #     qml.CNOT(wires=[5,7])
        #     qml.CNOT(wires=[7,1])
        # # m_1 = qml.measure(1)
        # m_3 = qml.measure(3)
        # m_5 = qml.measure(5)
        # m_7 = qml.measure(7)
        return [qml.expval(qml.PauliZ(i)) for i in range(4)] #[qml.expval(qml.PauliZ(i)) for i in range(1, config['num_qubits'],2)]
    
    elif type == 'critic':

        for idx in range(2):
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
                
            if idx < 1:
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])

        for i in range(0, config['num_qubits'], 2):
            m_c = qml.measure(i)
            qml.cond(m_c, qml.PauliX)(wires=i+1)
            
        idx = 2
        #############################
        z = 0
        for i in range(1,config['num_qubits'],2):
            qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
            qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
            z += 2

        z = 0
        for i in range(1,config['num_qubits'],2):                
            qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
            qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
            z += 2

        # qubits left: 1, 3, 5, 7
        
        m_1 = qml.measure(1)
        qml.cond(m_1, qml.PauliX)(wires=3)

        m_5 = qml.measure(5)
        qml.cond(m_5, qml.PauliX)(wires=7)
        
        # qubits left: 3, 7
        idx = 3
        #############################
        z = 0
        for i in [3,7]:
            qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
            qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
            z += 2

        z = 0
        for i in [3,7]:               
            qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
            qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
            z += 2

        idx = 4
        # qubits left: 3, 7
        m_3 = qml.measure(3)
        qml.cond(m_3, qml.PauliX)(wires=7)
        
        #############################
        z = 0
        i = 7
        qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
        qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)

        z = 0
        i = 7
                        
        qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
        qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
        
        return [qml.expval(qml.PauliZ(7))]
    
def qcnn_lunar_02(theta, weights, config, type, activations=None, **kwargs):

    for i in range(config['num_qubits']):
        qml.Hadamard(wires=i)        
    
    if type == 'activations_actor':
        type = 'actor'
        for idx in range(3):
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

            if idx < 2:    
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])

        return [qml.expval(qml.PauliZ(i)) for i in range(0, config['num_qubits'],2)]
        # https://discuss.pennylane.ai/t/mid-circuit-measurement-in-pennylane/3187
        # for i in range(0, config['num_qubits'], 2):
        #     m_c = qml.measure(i)
        #     qml.cond(m_c, qml.PauliX)(wires=i+1)
    elif type == 'actor':

        for idx in range(3):
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

            if idx < 2:    
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])
        w = 1
        for i in range(4):
            if activations[i] >= 0:
                qml.PauliX(wires=w)
            w += 2

        for idx in range(3,6):
            z = 0
            for i in range(1,config['num_qubits'],2):
                qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
                qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
                z += 2

            z = 0
            for i in range(1,config['num_qubits'],2):                
                qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
                qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
                z += 2
        
            qml.CNOT(wires=[1,3])
            qml.CNOT(wires=[3,5])
            qml.CNOT(wires=[5,7])
            qml.CNOT(wires=[7,1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(1, config['num_qubits'],2)]
    
    elif type == 'critic':

        for idx in range(3):
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
                
            if idx < 2:
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])

        for i in range(0, config['num_qubits'], 2):
            m_c = qml.measure(i)
            qml.cond(m_c, qml.PauliX)(wires=i+1)
            
        idx = 3
        #############################
        z = 0
        for i in range(1,config['num_qubits'],2):
            qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
            qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
            z += 2

        z = 0
        for i in range(1,config['num_qubits'],2):                
            qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
            qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
            z += 2

        # qubits left: 1, 3, 5, 7
        
        m_1 = qml.measure(1)
        qml.cond(m_1, qml.PauliX)(wires=3)

        m_5 = qml.measure(5)
        qml.cond(m_5, qml.PauliX)(wires=7)
        
        # qubits left: 3, 7
        idx = 4
        #############################
        z = 0
        for i in [3,7]:
            qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
            qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
            z += 2

        z = 0
        for i in [3,7]:               
            qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
            qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
            z += 2

        idx = 5
        # qubits left: 3, 7
        m_3 = qml.measure(3)
        qml.cond(m_3, qml.PauliX)(wires=7)
        
        #############################
        z = 0
        i = 7
        qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
        qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)

        z = 0
        i = 7
                        
        qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
        qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
        
        return [qml.expval(qml.PauliZ(7))]
           
def qcnn_lunar_03(theta, weights, config, type, activations=None):

    for i in range(config['num_qubits']):
        qml.Hadamard(wires=i)        
    
    if type == 'activations_actor':
        type = 'actor'
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

            if idx < config['blocks'][0][0][-1]:    
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])

        return [qml.expval(qml.PauliZ(i)) for i in range(0, config['num_qubits'],2)]
        # https://discuss.pennylane.ai/t/mid-circuit-measurement-in-pennylane/3187
        # for i in range(0, config['num_qubits'], 2):
        #     m_c = qml.measure(i)
        #     qml.cond(m_c, qml.PauliX)(wires=i+1)
    elif type == 'actor':

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

            if idx < config['blocks'][0][0][-1]:    
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])
        w = 1
        activations = np.where(activations > 0, np.pi, 0)
        for i in range(4):
            qml.RX(activations[:,i], wires=w)
            w += 2

        for idx in config['blocks'][0][1]:
            z = 0
            for i in range(1,config['num_qubits'],2):
                qml.RY(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
                qml.RZ(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
                z += 2

            z = 0
            for i in range(1,config['num_qubits'],2):                
                qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
                qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
                z += 2
        
            qml.CNOT(wires=[1,3])
            qml.CNOT(wires=[3,5])
            qml.CNOT(wires=[5,7])
            qml.CNOT(wires=[7,1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(1, config['num_qubits'],2)]
    
    elif type == 'critic':
        
        # Block 0
        for idx in config['blocks'][1][0]:
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
                
            if idx < config['blocks'][1][0][-1]:
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])

        for i in range(0, config['num_qubits'], 2):
            m_c = qml.measure(i)
            qml.cond(m_c, qml.PauliX)(wires=i+1)
            
        for idx in config['blocks'][1][1]:
            #############################
            z = 0
            for i in range(1,config['num_qubits'],2):
                qml.RY(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
                qml.RZ(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
                z += 2

            z = 0
            for i in range(1,config['num_qubits'],2):                
                qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
                qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
                z += 2
            if idx < config['blocks'][1][1][-1]:
                qml.CNOT(wires=[1,3])
                qml.CNOT(wires=[3,5])
                qml.CNOT(wires=[5,7])
                qml.CNOT(wires=[7,1])
        
        # qubits left: 1, 3, 5, 7
        m_1 = qml.measure(1)
        qml.cond(m_1, qml.PauliX)(wires=3)

        m_5 = qml.measure(5)
        qml.cond(m_5, qml.PauliX)(wires=7)
        
        # qubits left: 3, 7
        for idx in config['blocks'][1][2]:
        #############################
            z = 0
            for i in [3,7]:
                qml.RY(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
                qml.RZ(theta[:,i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
                z += 2

            z = 0
            for i in [3,7]:               
                qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
                qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
                z += 2
            if idx < config['blocks'][1][2][-1]:
                qml.CNOT(wires=[3,7])
                qml.CNOT(wires=[7,3])
        
        
        # qubits left: 3, 7
        m_7 = qml.measure(7)
        qml.cond(m_7, qml.PauliX)(wires=3)
       
        
        return [qml.expval(qml.PauliZ(3))]


def qcnn_pendulum_double(theta, weights, config, type, activations=None, **kwargs):

    for i in range(config['num_qubits']):
        qml.Hadamard(wires=i)        
    
    if type == 'activations_actor':
        type = 'actor'
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

            if idx < config['blocks'][0][0][-1]:    
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])

        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    elif type == 'actor':

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

            if idx < config['blocks'][0][0][-1]:    
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])
        w = 3
        for i in range(3):
            if activations[i] >= 0:
                qml.PauliX(wires=w)
            w += 1

        for idx in config['blocks'][0][1]:
            z = 0
            for i in range(3,6):
                qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
                qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
                z += 2

            z = 0
            for i in range(3,6):                
                qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
                qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
                z += 2
        
            qml.CNOT(wires=[3,4])
            qml.CNOT(wires=[4,5])
            qml.CNOT(wires=[5,3])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(3,6)]
    
    elif type == 'critic':
        
        # Block 0
        for idx in config['blocks'][1][0]:
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
                
            if idx < config['blocks'][1][0][-1]:
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])
        w = 3
        for i in range(0, 3):
            m_c = qml.measure(i)
            qml.cond(m_c, qml.PauliX)(wires=w)
            w += 1

        for idx in config['blocks'][1][1]:
            #############################
            z = 0
            for i in range(3,6):
                qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
                qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
                z += 2

            z = 0
            for i in range(3,6):                
                qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
                qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
                z += 2
            if idx < config['blocks'][1][1][-1]:
                qml.CNOT(wires=[3,4])
                qml.CNOT(wires=[4,5])
                qml.CNOT(wires=[5,3])
        
        # qubits left: 1, 3, 5, 7
        m_3 = qml.measure(3)
        qml.cond(m_3, qml.PauliX)(wires=4)
        
        # qubits left: 3, 7
        for idx in config['blocks'][1][2]:
        #############################
            z = 0
            for i in [4,5]:
                qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
                qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
                z += 2

            z = 0
            for i in [4,5]:               
                qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
                qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
                z += 2
            if idx < config['blocks'][1][2][-1]:
                qml.CNOT(wires=[4,5])
                qml.CNOT(wires=[5,4])
        
        # qubits left: 3, 7
        m_4 = qml.measure(4)
        qml.cond(m_4, qml.PauliX)(wires=5)
        
        
        return [qml.expval(qml.PauliZ(5))]

def qcnn_pendulum_triple(theta, weights, config, type, activations=None, **kwargs):

    for i in range(config['num_qubits']):
        qml.Hadamard(wires=i)        
    
    if type == 'activations_actor':
        type = 'actor'
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

            if idx < config['blocks'][0][0][-1]:    
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])

        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    elif type == 'actor':

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

            if idx < config['blocks'][0][0][-1]:    
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])
        w = 6
        for i in range(3):
            if activations[i] >= 0:
                qml.PauliX(wires=w)
            w += 1

        for idx in config['blocks'][0][1]:
            z = 0
            for i in range(6,9):
                qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
                qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
                z += 2

            z = 0
            for i in range(6,9):                
                qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
                qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
                z += 2
        
            qml.CNOT(wires=[6,7])
            qml.CNOT(wires=[7,8])
            qml.CNOT(wires=[8,6])
        
        # w = 6
        # for i in range(3,6):
        #     if activations[i] >= 0:
        #         qml.PauliX(wires=w)
        #     w += 1
        
        return [qml.expval(qml.PauliZ(i)) for i in range(6,9)]
    
    elif type == 'critic':
        
        # Block 0
        for idx in config['blocks'][1][0]:
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
                
            if idx < config['blocks'][1][0][-1]:
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])
        w = 3
        for i in range(0, 3):
            m_c = qml.measure(i)
            qml.cond(m_c, qml.PauliX)(wires=w)
            qml.cond(m_c, qml.PauliX)(wires=w+3)
            w += 1

        for idx in config['blocks'][1][1]:
            #############################
            z = 0
            for i in range(3,9):
                qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
                qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
                z += 2

            z = 0
            for i in range(3,9):                
                qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
                qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
                z += 2
            if idx < config['blocks'][1][1][-1]:
                for i in range(3, config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 3
                    qml.CNOT(wires=[x,xx])
        

        w = 6
        for i in range(3, 6):
            m_c = qml.measure(i)
            qml.cond(m_c, qml.PauliX)(wires=w)
            w += 1
        
        # qubits left: 3, 7
        for idx in config['blocks'][1][2]:
        #############################
            z = 0
            for i in range(6,9):
                qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
                qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
                z += 2

            z = 0
            for i in range(6,9):               
                qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
                qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
                z += 2

            for i in range(6, config['num_qubits']):
                x = i
                xx = x+1
                if i == config['num_qubits'] - 1:
                    xx = 6
                qml.CNOT(wires=[x,xx])
        
        
        return [qml.expval(qml.PauliZ(i)) for i in range(6,9)]
      
def qcnn_lunar_short(theta, weights, config, type, activations=None, **kwargs):

    for i in range(config['num_qubits']):
        qml.Hadamard(wires=i)        
    
    if type == 'actor':
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

        return [qml.expval(qml.PauliZ(i)) for i in range(config['num_qubits'])]

    elif type == 'critic':
        
        # Block 0
        for idx in config['blocks'][1][0]:
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
                
            if idx < config['blocks'][1][0][-1]:
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    qml.CNOT(wires=[x,xx])

        for i in range(0, config['num_qubits'], 2):
            m_c = qml.measure(i)
            qml.cond(m_c, qml.PauliX)(wires=i+1)
            
        for idx in config['blocks'][1][1]:
            #############################
            z = 0
            for i in range(1,config['num_qubits'],2):
                qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
                qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
                z += 2

            z = 0
            for i in range(1,config['num_qubits'],2):                
                qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
                qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
                z += 2
            if idx < config['blocks'][1][1][-1]:
                qml.CNOT(wires=[1,3])
                qml.CNOT(wires=[3,5])
                qml.CNOT(wires=[5,1])

        
        # qubits left: 1, 3, 5
        m_1 = qml.measure(1)
        qml.cond(m_1, qml.PauliX)(wires=3)
        qml.cond(m_1, qml.PauliX)(wires=5)
        
        # qubits left: 3, 5
        for idx in config['blocks'][1][2]:
        #############################
            z = 0
            for i in [3,5]:
                qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
                qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)
                z += 2

            z = 0
            for i in [3,5]:               
                qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
                qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
                z += 2
            if idx < config['blocks'][1][2][-1]:
                qml.CNOT(wires=[3,5])
                qml.CNOT(wires=[5,3])
        
        
        # qubits left: 3, 5
        m_5 = qml.measure(5)
        qml.cond(m_5, qml.PauliX)(wires=3)
        for idx in config['blocks'][1][3]:
            #############################
            z = 0
            i = 3
            qml.RY(theta[i]*weights[f'input_scaling_{type}_{idx}'][z], wires=i)
            qml.RZ(theta[i]*weights[f'input_scaling_{type}_{idx}'][z+1], wires=i)

            z = 0
            i = 3
            qml.RZ(weights[f'weights_{type}_{idx}'][z], wires=i)
            qml.RY(weights[f'weights_{type}_{idx}'][z+1], wires=i)
        
        return [qml.expval(qml.PauliZ(3))]

