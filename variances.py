import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
from functools import partial
import argparse

def sppgl(params,linear_inputs,quadratic_inputs, annotations, num_qubits,type_measurement):
    for qubit in range(num_qubits):
        qml.Hadamard(qubit)
    num_layers = params.shape[0]
    for layer in range(num_layers):
        y = 0
        for i in range(len(quadratic_inputs)):
            qml.IsingZZ(quadratic_inputs[i][2]*params[layer][y], wires=[quadratic_inputs[i][0], quadratic_inputs[i][1]])

        for i in range(num_qubits):
            qml.RZ(linear_inputs[i][1]*params[layer][y],wires = linear_inputs[i][0])

        y += 1

        for i in range(num_qubits):
            qml.RX(annotations[i][1]*params[layer][y],wires=annotations[i][0])
        y += 1
    if type_measurement == "single":
        return qml.expval(qml.PauliZ(wires = 0))
    else:
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

def mppgl(params,linear_inputs,quadratic_inputs, annotations, num_qubits,type_measurement):
    [qml.Hadamard(qubit) for qubit in range(num_qubits)]
    num_layers = params.shape[0]
    for layer in range(num_layers):
        y = 0
        for i in range(len(quadratic_inputs)):
            qml.IsingZZ(quadratic_inputs[i][2]*params[layer][y], wires=[quadratic_inputs[i][0], quadratic_inputs[i][1]])
            y += 1

        for i in range(num_qubits):
            qml.RZ(linear_inputs[i][1]*params[layer][y],wires = linear_inputs[i][0])
            y += 1

        for i in range(num_qubits):
            qml.RX(annotations[i][1]*params[layer][y],wires=annotations[i][0])
            y += 1
    if type_measurement == "single":
        return qml.expval(qml.PauliZ(0))
    else:
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

def hppgl(params,linear_inputs,quadratic_inputs, annotations, num_qubits,type_measurement):
    [qml.Hadamard(qubit) for qubit in range(num_qubits)]
    num_layers = params.shape[0]
    for layer in range(num_layers):
        y = 0
        for i in range(len(quadratic_inputs)):
            qml.IsingZZ(quadratic_inputs[i][2]*params[layer][y], wires=[quadratic_inputs[i][0], quadratic_inputs[i][1]])
            y += 1

        for i in range(num_qubits):
            qml.RZ(linear_inputs[i][1]*params[layer][y],wires = linear_inputs[i][0])
            y += 1

        for i in range(num_qubits):
            qml.RX(annotations[i][1]*params[layer][y],wires=annotations[i][0])
        y += 1
    if type_measurement == "single":
        return qml.expval(qml.PauliZ(0))
    else:
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

def hwe_skolik(params,linear_inputs,quadratic_inputs,annotations, num_qubits,type_measurement):
    num_layers = params.shape[0]
    for layer in range(num_layers):
        y = 0
        for i in range(num_qubits):
            qml.RX(params[layer][y],wires = i)
            y += 1

        for i in range(len(quadratic_inputs)):
            qml.IsingZZ(quadratic_inputs[i][2], wires=[quadratic_inputs[i][0], quadratic_inputs[i][1]])

        for i in range(num_qubits):
            qml.RY(params[layer][y],wires = i)
            y += 1
        
        for i in range(0,num_qubits-1):
            qml.CZ(wires = [i,i+1])

    if type_measurement == "single":
        return qml.expval(qml.PauliZ(0))
    else:
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

def sppgl_hwe(params,linear_inputs,quadratic_inputs,annotations, num_qubits,type_measurement):
    num_layers = params.shape[0]
    for layer in range(num_layers):
        y = 0

        for i in range(len(quadratic_inputs)):
            qml.IsingZZ(quadratic_inputs[i][2]*params[layer][y], wires=[quadratic_inputs[i][0], quadratic_inputs[i][1]])

        for i in range(num_qubits):
            qml.RZ(linear_inputs[i][1]*params[layer][y],wires = linear_inputs[i][0])
        y += 1
        
        for i in range(num_qubits):
            qml.RY(params[layer][y],wires = i)
            y += 1

        for i in range(num_qubits):
            qml.RZ(params[layer][y],wires = i)
            y += 1

        for i in range(0,num_qubits-1):
            qml.CZ(wires = [i,i+1])
        qml.CZ(wires = [num_qubits-1,0])

    if type_measurement == "single":
        return qml.expval(qml.PauliZ(0))
    else:
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))
    
def sppgl_hwe_static(params,linear_inputs,quadratic_inputs,annotations, num_qubits,type_measurement):
    num_layers = params.shape[0]
    for layer in range(num_layers):
        y = 0

        for i in range(len(quadratic_inputs)):
            qml.IsingZZ(quadratic_inputs[i][2], wires=[quadratic_inputs[i][0], quadratic_inputs[i][1]])

        for i in range(num_qubits):
            qml.RZ(linear_inputs[i][1],wires = linear_inputs[i][0])
        
        
        for i in range(num_qubits):
            qml.RY(params[layer][y],wires = i)
            y += 1

        for i in range(num_qubits):
            qml.RZ(params[layer][y],wires = i)
            y += 1

        for i in range(0,num_qubits-1):
            qml.CZ(wires = [i,i+1])
        qml.CZ(wires = [num_qubits-1,0])

    if type_measurement == "single":
        return qml.expval(qml.PauliZ(0))
    else:
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))
    
def tfim(params,linear_inputs,quadratic_inputs,annotations, num_qubits,type_measurement):
    num_layers = params.shape[0]
    for layer in range(num_layers):
        y = 0

        for i in range(num_qubits):
            if i < num_qubits -1:
                qml.IsingZZ(quadratic_inputs[i][2]*params[layer][y], wires=[i,i+1])

        y += 1

        for i in range(num_qubits):
            qml.RX(linear_inputs[i][1]*params[layer][y],wires = linear_inputs[i][0])

        y += 1


    if type_measurement == "single":
        return qml.expval(qml.PauliZ(0))
    else:
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

def ltfim(params,linear_inputs,quadratic_inputs,annotations, num_qubits,type_measurement):
    num_layers = params.shape[0]
    for layer in range(num_layers):
        y = 0

        for i in range(num_qubits):
            if i < num_qubits -1:
                qml.IsingZZ(quadratic_inputs[i][2]*params[layer][y], wires=[i,i+1])

        y += 1

        for i in range(num_qubits):
            qml.RX(linear_inputs[i][1]*params[layer][y],wires = linear_inputs[i][0])

        y += 1

        for i in range(num_qubits):
            qml.RZ(linear_inputs[i][1]*params[layer][y],wires = linear_inputs[i][0])

        y += 1
    if type_measurement == "single":
        return qml.expval(qml.PauliZ(0))
    else:
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

def create_array(n):
    column_1 = np.arange(n)
    column_2 = np.random.uniform(0, 2*np.pi, n)
    result_array = np.column_stack((column_1, column_2))
    return result_array

def create_quadratic_array(n):
    # Generate upper triangular indices to represent unique combinations
    indices = np.triu_indices(n, k=1)

    # Extract row and column indices
    row_indices, col_indices = indices

    # Generate random values for the third column
    column_3 = np.random.uniform(0, 2*np.pi, len(row_indices))

    # Stack the columns horizontally to create the final array
    result_array = np.column_stack((row_indices, col_indices, column_3))
    return result_array

def process_circuit(circuit_type, qubit, num_samples, num_layers, type_gradient,type_measurement):
    dev = qml.device("lightning.qubit", wires=qubit)
    
    if circuit_type == "sppgl":
        circuit = qml.QNode(sppgl, dev, diff_method="adjoint")
    elif circuit_type == "mppgl":
        circuit = qml.QNode(mppgl, dev, diff_method="adjoint")
    elif circuit_type == "hppgl":
        circuit = qml.QNode(hppgl, dev, diff_method="adjoint")
    elif circuit_type == "hwe":
        circuit = qml.QNode(hwe_skolik, dev, diff_method="adjoint")
    elif circuit_type == "sppgl-hwe":
        circuit = qml.QNode(sppgl_hwe, dev, diff_method="adjoint")
    elif circuit_type == "sppgl-hwe-static":
        circuit = qml.QNode(sppgl_hwe_static, dev, diff_method="adjoint")
    elif circuit_type == "tfim":
        circuit = qml.QNode(tfim, dev, diff_method="adjoint")
    elif circuit_type == "ltfim":
        circuit = qml.QNode(ltfim, dev, diff_method="adjoint")
    else:
        raise ValueError("Circuit Type not recognized")
    
    grad_fn = qml.grad(circuit, argnum=0)
    
    grad_vals_last, grad_vals_middle, grad_vals_all = [], [], []
    for _ in range(num_samples):
        linear_inputs = create_array(qubit)
        annotations = create_array(qubit)
        quadratic_inputs = create_quadratic_array(qubit)
        
        if circuit_type == "sppgl":
            params = np.random.uniform(0, 2*np.pi, size=(num_layers, 2))
        elif circuit_type == "mppgl":
            params = np.random.uniform(0, 2*np.pi, size=(num_layers, 2*qubit + len(quadratic_inputs)))
        elif circuit_type == "hppgl":
            params = np.random.uniform(0, 2*np.pi, size=(num_layers, 1 + qubit + len(quadratic_inputs)))
        elif circuit_type == "hwe":
            params = np.random.uniform(0, 2*np.pi, size=(num_layers, 2*qubit))
        elif circuit_type == "sppgl-hwe":
            params = np.random.uniform(0, 2*np.pi, size=(num_layers, 2*qubit+2))    
        elif circuit_type == "sppgl-hwe-static":
            params = np.random.uniform(0, 2*np.pi, size=(num_layers, 2*qubit))     
        elif circuit_type == "tfim":
            params = np.random.uniform(0, 2*np.pi, size=(num_layers, 2))
        elif circuit_type == "ltfim":
            params = np.random.uniform(0, 2*np.pi, size=(num_layers, 3))
        else:
            raise ValueError("Circuit Type not recognized")
        
        gradient = grad_fn(params, linear_inputs, quadratic_inputs, annotations, qubit,type_measurement)
        
        grad_vals_last.append(gradient[-1])
        grad_vals_all.append(np.mean(gradient))
        grad_vals_middle.append(gradient[len(gradient)//2])
    
    return [np.var(grad_vals_last), np.var(grad_vals_middle), np.var(grad_vals_all)]

if __name__ == "__main__":
    start = time.time()
    num_qubits = [9,10,11,12,13,14,15] #[2,3,4,5,6,7,8] #9,10] #[9, 10 11,12,13,14,15] #
    num_layers = 50
    num_samples = 1000

    parser = argparse.ArgumentParser(description= \
                                         "This is the runfile for the open source baseline repo.")
    parser.add_argument("--type_gradient", default= "last",type=str)
    parser.add_argument("--type_measurement", default="double",type=str)
    args = parser.parse_args()
    
    variances = {"sppgl": [], "mppgl": [], "hppgl": [], "hwe": [],"sppgl-hwe": [], "sppgl-hwe-static": [], "tfim": [], "ltfim": []}
    num_cpus = multiprocessing.cpu_count()
    print('cpus:', num_cpus)
    # with multiprocessing.Pool(processes=num_cpus) as pool:
    #     for qubit in num_qubits:
    #         start_qubit = time.time()
    #         circuit_types = ["sppgl", "mppgl", "hppgl", "hwe", "sppgl-hwe", "sppgl-hwe-static", "tfim", "ltfim"]
    #         process_func = partial(process_circuit, qubit=qubit, num_samples=num_samples,
    #                                num_layers=num_layers, type_gradient=args.type_gradient, type_measurement=args.type_measurement)
    #         results = pool.map(process_func, circuit_types)
            
    #         for circuit_type, variance in zip(circuit_types, results):
    #             variances[circuit_type].append(variance)
            
    #         end_qubit = time.time() - start_qubit
    
            
    #         print(f"Variance for qubit {qubit} calculated, took {end_qubit} seconds")
    #     np.save(f'variance_{args.type_gradient}_{args.type_measurement}_{num_layers}_layers_1000_samples_9_15', variances)
        
total_time = time.time() - start

print(total_time)

fig, ax_5 = plt.subplots(1, figsize=(6, 5))

label_names = [
    '$sge-sgv$',
    '$mge-mgv$',
    '$mge-sgv$',
    '$sge-sgv+hea$',
    '$encoding+hea$',
    # '$skolik+hea$',
    '$tfim$',
    # '$ltfim$'
]

index_1 = 1
index_2 = 0
colors = ['darkcyan', 'indianred', 'slategray',  'peru', 'mediumseagreen', 'cornflowerblue']

num_qubits = [2,3,4,5,6,7,8]
variances = np.load('variance_last_double_5_layers_1000_samples_2_8.npy', allow_pickle=True).item()

for key, value in variances.items():
    value = np.array(value)
    if key == "sppgl":
        ax_5.plot(num_qubits, value[:,index_1], "o", label=label_names[0], color=colors[0])
    elif key == "mppgl":
        ax_5.plot(num_qubits, value[:,index_1], "o", label=label_names[1], color=colors[1])
    elif key == "hppgl":
        ax_5.plot(num_qubits, value[:,index_1], "o", label=label_names[2], color=colors[2])
    elif key == "sppgl-hwe":
        ax_5.plot(num_qubits, value[:,index_1], "o", label=label_names[3], color=colors[3])
    elif key == "hwe":
        ax_5.plot(num_qubits, value[:,index_1], "o", label=label_names[4], color=colors[4])
    # elif key == "tfim":
    #     ax_5.plot(num_qubits, value[:,index_1], "o", label=label_names[5], color=colors[5])


num_qubits = [9,10,11,12,13,14,15]
variances = np.load('variance_last_double_5_layers_1000_samples_9_15.npy', allow_pickle=True).item()

for key, value in variances.items():
    value = np.array(value)
    if key == "sppgl":
        ax_5.plot(num_qubits, value[:,index_1], "o", color=colors[0])
    elif key == "mppgl":
        ax_5.plot(num_qubits, value[:,index_1], "o", color=colors[1])
    elif key == "hppgl":
        ax_5.plot(num_qubits, value[:,index_1], "o", color=colors[2])
    elif key == "sppgl-hwe":
        ax_5.plot(num_qubits, value[:,index_1], "o", color=colors[3])
    elif key == "sppgl-hwe-static":
        ax_5.plot(num_qubits, value[:,index_1], "o", color=colors[4])
    # elif key == "tfim":
    #     ax_5.plot(num_qubits, value[:,index_1], "o", label=label_names[5], color=colors[5])

# axis.set_ylim(-30, 3)
ax_5.set_xlabel("$qubits$", fontsize=15)
ax_5.set_ylabel('$variance$', fontsize=15)
ax_5.set_yscale('log')
# ax_5.set_ylim(1e-6, 0.5)
ax_5.set_title('$5 \, layer$',fontsize=15)

ax_5.minorticks_on()
ax_5.grid(which='both', alpha=0.4,  ls="-")

handles, labels = ax_5.get_legend_handles_labels()
order = [0,2,1,4,3]
ax_5.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=12, loc='lower left')
# num_qubits = [2,3,4,5,6,7,8]
# variances = np.load('variance_last_double_5_layers_1000_samples_2_8.npy', allow_pickle=True).item()

# for key, value in variances.items():
#     value = np.array(value)
#     if key == "sppgl":
#         ax_50.plot(num_qubits, value[:,index_2], "o", label=label_names[0], color=colors[0])
#     elif key == "mppgl":
#         ax_50.plot(num_qubits, value[:,index_2], "o", label=label_names[1], color=colors[1])
#     elif key == "hppgl":
#         ax_50.plot(num_qubits, value[:,index_2], "o", label=label_names[2], color=colors[2])
#     elif key == "sppgl-hwe":
#         ax_50.plot(num_qubits, value[:,index_2], "o", label=label_names[3], color=colors[3])
#     elif key == "sppgl-hwe-static":
#         ax_50.plot(num_qubits, value[:,index_2], "o", label=label_names[4], color=colors[4])


# num_qubits = [9,10,11,12,13,14,15]
# variances = np.load('variance_last_double_5_layers_1000_samples_9_15.npy', allow_pickle=True).item()

# for key, value in variances.items():
#     value = np.array(value)
#     if key == "sppgl":
#         ax_50.plot(num_qubits, value[:,index_2], "o", color=colors[0])
#     elif key == "mppgl":
#         ax_50.plot(num_qubits, value[:,index_2], "o", color=colors[1])
#     elif key == "hppgl":
#         ax_50.plot(num_qubits, value[:,index_2], "o", color=colors[2])
#     elif key == "sppgl-hwe":
#         ax_50.plot(num_qubits, value[:,index_2], "o", color=colors[3])
#     elif key == "hwe":
#         ax_50.plot(num_qubits, value[:,index_2], "o", color=colors[4])

# # axis.set_ylim(-30, 3)
# ax_50.set_xlabel("$qubits$", fontsize=15)
# ax_50.set_ylabel('$variance$', fontsize=15)
# ax_50.set_yscale('log')
# # ax_50.set_ylim(1e-6, 0.5)
# ax_50.set_title('$5 \, layer$',fontsize=15)
# ax_50.legend(fontsize=12, loc='lower left')
# ax_50.minorticks_on()
# ax_50.grid(which='major', alpha=0.4)

# variances = np.load('variance_last_double_50_layers_6.npy', allow_pickle=True).item()

# for key, value in variances.items():
#     ax_50.plot(num_qubits, value, "o", label=key)
# # axis.set_ylim(-30, 3)
# ax_50.set_xlabel("$qubits$", fontsize=15)
# # ax_edge.set_ylabel('$approximation ratio$', fontsize=15)
# ax_50.set_yscale('log')
# # ax_50.set_ylim(1e-6, 0.5)
# ax_50.set_title('$50 \, layer$',fontsize=15)
# ax_50.legend(fontsize=12, loc='lower left')
# ax_50.minorticks_on()
# ax_50.grid(which='major', alpha=0.4)

fig.tight_layout()
plt.savefig(f"Variance {args.type_gradient} {args.type_measurement}_5_layers_1000_samples.pdf")

# plt.semilogy(num_qubits, variances["sppgl"], "o", label = "sppgl")
# plt.semilogy(num_qubits, variances["mppgl"], "o",label = "mppgl")
# plt.semilogy(num_qubits, variances["hppgl"], "o", label = "hppgl")
# plt.semilogy(num_qubits, variances["sppgl-hwe"], "o", label = "sppgl-hwe")
# plt.semilogy(num_qubits, variances["hwe"], "o", label = "hwe")
# plt.semilogy(num_qubits, variances["tfim"], "o", label = "tfim")
# plt.semilogy(num_qubits, variances["ltfim"], "o", label = "ltfim")
# #plt.semilogy(qubit, np.exp(p[0] * qubit + p[1]), "o-.", label="Slope {:3.2f}".format(p[0]))
# plt.xlabel(r"N Qubits")
# plt.ylabel(r"Variance")
# plt.legend()
# plt.savefig(f"Variance {args.type_gradient} {args.type_measurement}_50_layers.png")