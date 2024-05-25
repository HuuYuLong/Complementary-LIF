import numpy as np
import torch.nn


# A. Operational cost  (Synapse Op)
def synaptic_op(input_size, output_size, kernel_size, neuron: str = None, T=None, fr_in=None, fr_out=None, stride=1):
    # MAC = 0.
    # ACC = 0.

    if isinstance(input_size, int):
        """ fully_connect"""
        if neuron == None:
            # ANN calculating
            MAC = input_size * output_size  # Weight
            ACC = output_size  # bias
        elif neuron == "LIF":
            # SNN LIF calculation
            spike_num_in = input_size * fr_in
            spike_num_out = output_size * fr_out

            MAC = T * output_size  # membrane potential leaky

            ACC_w = spike_num_in * input_size * output_size  # synapse weight
            ACC_b = T * spike_num_out

            ACC = ACC_w + ACC_b
        elif neuron == "CLIF":
            # SNN CLIF calculation
            spike_num_in = input_size * fr_in
            spike_num_out = output_size * fr_out

            MAC = T * output_size * 3  # membrane potential leaky

            ACC_w = spike_num_in * input_size * output_size  # synapse weight
            ACC_b = T * spike_num_out * 3

            ACC = ACC_w + ACC_b
        else:
            raise NotImplementedError

    else:
        """convolution"""
        C_in, H_in, W_in = input_size
        C_out, H_out, W_out = output_size
        _, _, H_kernel, W_kernel = kernel_size
        # stride = kerner.stride

        if neuron == None:
            # ANN calculating
            MAC = C_out * H_out * W_out * C_in * H_kernel * W_kernel  # Weight
            ACC = C_out * H_out * W_out  # bias
        elif neuron == "LIF":
            # SNN LIF calculation
            spike_num_in = (C_in * H_in * W_in) * fr_in
            spike_num_out = (C_out * H_out * W_out) * fr_out

            MAC = T * C_out * H_out * W_out  # membrane potential leaky

            ACC_w = spike_num_in * (H_kernel // stride) * (W_kernel // stride) * C_out  # Weight op
            ACC_b = T * C_out * H_out * W_out + spike_num_out  # Bias op

            ACC = ACC_w + ACC_b

        elif neuron == "CLIF":
            # SNN LIF calculation
            spike_num_in = (C_in * H_in * W_in) * fr_in
            spike_num_out = (C_out * H_out * W_out) * fr_out

            MAC = T * C_out * H_out * W_out * 3  # membrane potential leaky
            ACC_w = spike_num_in * (H_kernel // stride) * (W_kernel // stride) * C_out  # Weight op
            ACC_b = T * C_out * H_out * W_out + spike_num_out * 3  # Bias op

            ACC = ACC_w + ACC_b
        else:
            raise NotImplementedError

    return MAC, ACC


# B. Memory cost
def memory_cost(input_size, output_size, kernel_size=None, neuron: str = None, T=None, fr_in=None, fr_out=None,
                stride=1):
    # read_in = 0.
    # read_params = 0.
    # read_potential = 0.
    # write_out = 0.
    # write_potential = 0.

    if isinstance(input_size, int):
        """ fully_connect"""

        if neuron == None:
            read_in = input_size
            read_params = (input_size + 1) * output_size
            read_potential = 0.
            write_out = output_size
            write_potential = 0.
        elif neuron == "LIF":
            spike_num_in = input_size * fr_in
            spike_num_out = output_size * fr_out

            read_in = spike_num_in
            read_params = spike_num_in * output_size + output_size
            read_potential = (spike_num_in + 1) * spike_num_out
            write_out = spike_num_out
            write_potential = spike_num_in * output_size + output_size
        elif neuron == "CLIF":
            spike_num_in = input_size * fr_in
            spike_num_out = output_size * fr_out

            read_in = spike_num_in
            read_params = spike_num_in * output_size + output_size
            read_potential = (spike_num_in + 1) * spike_num_out * 4
            write_out = spike_num_out
            write_potential = (spike_num_in * output_size + output_size) * 2
        else:
            raise NotImplementedError


    else:
        """convolution"""
        C_in, H_in, W_in = input_size
        C_out, H_out, W_out = output_size
        _, _, H_kernel, W_kernel = kernel_size

        if neuron == None:
            read_in = C_out * C_in * H_out * W_out * H_kernel * W_kernel
            read_params = (C_in * W_kernel * H_kernel + 1) * C_out * W_out * H_out
            read_potential = 0.
            write_out = C_out * W_out * H_out
            write_potential = 0.
        elif neuron == "LIF":
            spike_num_in = (C_in * H_in * W_in) * fr_in
            spike_num_out = (C_out * H_out * W_out) * fr_out

            read_in = spike_num_in
            read_params = spike_num_in * C_out * W_kernel * H_kernel + C_out * W_out * H_out
            read_potential = spike_num_in * C_out * W_kernel * H_kernel + C_out * W_out * H_out
            write_out = spike_num_out
            write_potential = spike_num_in * C_out * W_kernel * H_kernel + C_out * W_out * H_out
        elif neuron == "CLIF":
            spike_num_in = (C_in * H_in * W_in) * fr_in
            spike_num_out = (C_out * H_out * W_out) * fr_out

            read_in = spike_num_in
            read_params = spike_num_in * C_out * W_kernel * H_kernel + C_out * W_out * H_out
            read_potential = (spike_num_in * C_out * W_kernel * H_kernel + C_out * W_out * H_out) * 4
            write_out = spike_num_out
            write_potential = (spike_num_in * C_out * W_kernel * H_kernel + C_out * W_out * H_out) * 2
        else:
            raise NotImplementedError

    return read_in, read_params, read_potential, write_out, write_potential  # a, b, c, d, e


# C. Addressing
def addressing_cost(input_size, output_size, kernel_size=None, neuron: str = None, T=None, fr_in=None, fr_out=None,
                    stride=1):
    # MAC = 0.
    # ACC = 0.

    if isinstance(input_size, int or float):
        """ fully_connect"""
        if neuron == None:
            # ANN calculating
            MAC = 0.  # Weight
            ACC = input_size + output_size  # bias

        elif neuron == "LIF":
            # SNN LIF calculation
            spike_num_in = input_size * fr_in
            spike_num_out = output_size * fr_out

            MAC = 0.
            ACC = spike_num_in * output_size

        elif neuron == "CLIF":
            # SNN LIF calculation
            spike_num_in = input_size * fr_in
            spike_num_out = output_size * fr_out

            MAC = 0.
            ACC = spike_num_in * output_size * 2
        else:
            raise NotImplementedError

    else:
        """convolution"""
        C_in, H_in, W_in = input_size
        C_out, H_out, W_out = output_size
        _, _, H_kernel, W_kernel = kernel_size
        # stride = kerner.stride

        if neuron == None:
            # ANN calculating
            MAC = 0.
            ACC = C_in * H_in * W_in \
                  + C_out * H_out * W_out \
                  + C_out * H_kernel * W_kernel
        elif neuron == "LIF":
            # SNN LIF calculation
            spike_num_in = (C_in * H_in * W_in) * fr_in
            spike_num_out = (C_out * H_out * W_out) * fr_out

            MAC = spike_num_in * 2
            ACC = spike_num_in * C_out * H_kernel * W_kernel
        elif neuron == "CLIF":
            # SNN LIF calculation
            spike_num_in = (C_in * H_in * W_in) * fr_in
            spike_num_out = (C_out * H_out * W_out) * fr_out

            MAC = spike_num_in * 2 * 2
            ACC = spike_num_in * C_out * H_kernel * W_kernel * 2

        else:
            raise NotImplementedError

    return MAC, ACC


# in paper  E_RdRAM = E_WrRAM
# For SRAM memory accesses, we compute a linear interpolation function based on 3 particular values :
# 8 kB (10pJ), 32 kB (20pJ) and 1 MB (100pJ).
# This function enables to compute the energy cost of a memory access knowing the memory size
# (i.e. knowing the network hyperparameters).
def E_func(Memory):
    E1 = 10  # pj
    E2 = 20  # pj
    E3 = 100  # pj

    m1 = 8 * 1024
    m2 = 32 * 1024
    m3 = 1024 * 1024
    if Memory <= m1:
        # E = Memory
        E = ((E1 - 0) / (m1 - 0)) * (Memory - 0)
    elif (Memory > m1) and (Memory <= m2):
        E = ((E2 - E1) / (m2 - m1)) * (Memory - m1)
    elif (Memory > m2) and (Memory <= m3):
        E = ((E3 - E2) / (m3 - m2)) * (Memory - m2)
    else:
        E = E3
        # raise NotImplementedError
    return E


# E_RdRAM = E_func()
# E_WrRAM = E_func()
#
# def calculation_E(read_in, read_params, read_potential, write_out, write_potential, MAC_op, ACC_op, MAC_adr, ACC_adr):
#     return read_in * E_func(read_in)

def calculate_all_operations(net_layers, neuron, T, spike_fr):
    mac_op = 0.
    acc_op = 0.

    read_in, read_params, read_potential, write_out, write_potential = 0., 0., 0., 0., 0.

    mac_addr = 0.
    acc_addr = 0.

    for i in range(len(net_layers)):

        input_size, output_size, kernel_size, stride = net_layers[i]

        if i == 0:
            MAC_op, ACC_op = synaptic_op(input_size=input_size, output_size=output_size, kernel_size=kernel_size,
                                         neuron=neuron, T=T, fr_in=1., fr_out=spike_fr[0], stride=stride)
            a, b, c, d, e = memory_cost(input_size=input_size, output_size=output_size, kernel_size=kernel_size,
                                        neuron=neuron, T=T, fr_in=1., fr_out=spike_fr[0], stride=stride)
            ma, aa = addressing_cost(input_size=input_size, output_size=output_size,
                                     kernel_size=kernel_size,
                                     neuron=neuron, T=T, fr_in=1., fr_out=spike_fr[0], stride=stride)
        else:
            fr_in = spike_fr[i - 1]

            if i == len(net_layers) - 1:
                fr_out = 0.
            else:
                fr_out = spike_fr[i]
            MAC_op, ACC_op = synaptic_op(input_size=input_size, output_size=output_size, kernel_size=kernel_size,
                                         neuron=neuron, T=T, fr_in=fr_in, fr_out=fr_out, stride=stride)
            a, b, c, d, e = memory_cost(input_size=input_size, output_size=output_size, kernel_size=kernel_size,
                                        neuron=neuron, T=T, fr_in=fr_in, fr_out=fr_out, stride=stride)
            ma, aa = addressing_cost(input_size=input_size, output_size=output_size,
                                     kernel_size=kernel_size,
                                     neuron=neuron, T=T, fr_in=fr_in, fr_out=fr_out, stride=stride)

        mac_op += MAC_op
        acc_op += ACC_op

        read_in += a
        read_params += b
        read_potential += c
        write_out += d
        write_potential += e

        mac_addr += ma
        acc_addr += aa
    return mac_op, acc_op, read_in, read_params, read_potential, write_out, write_potential, mac_addr, acc_addr


# input_size, output_size, kernel_size=None, stride
resnet_18_cifar10_network = [
    [(3, 32, 32), (64, 32, 32), (3, 64, 3, 3), 1],
    [(64, 32, 32), (64, 32, 32), (64, 64, 3, 3), 1],

    [(64, 32, 32), (64, 32, 32), (64, 64, 3, 3), 1],
    [(64, 32, 32), (64, 32, 32), (64, 64, 3, 3), 1],

    [(64, 32, 32), (64, 32, 32), (64, 64, 3, 3), 1],
    [(64, 32, 32), (128, 16, 16), (64, 128, 3, 3), 2],

    [(128, 16, 16), (128, 16, 16), (128, 128, 3, 3), 1],
    [(128, 16, 16), (128, 16, 16), (128, 128, 3, 3), 1],

    [(128, 16, 16), (128, 16, 16), (128, 128, 3, 3), 1],
    [(128, 16, 16), (256, 8, 8), (128, 256, 3, 3), 2],

    [(256, 8, 8), (256, 8, 8), (256, 256, 3, 3), 1],
    [(256, 8, 8), (256, 8, 8), (256, 256, 3, 3), 1],

    [(256, 8, 8), (256, 8, 8), (256, 256, 3, 3), 1],
    [(256, 8, 8), (512, 4, 4), (256, 512, 3, 3), 2],

    [(512, 4, 4), (512, 4, 4), (512, 512, 3, 3), 1],
    [(512, 4, 4), (512, 4, 4), (512, 512, 3, 3), 1],

    [(512, 4, 4), (512, 4, 4), (512, 512, 3, 3), 1],
    [512, 10, None, None],  # ignoring pooling
]

# input_size, output_size, kernel_size=None, stride
resnet_18_cifar100_network = [
    [(3, 32, 32), (64, 32, 32), (3, 64, 3, 3), 1],
    [(64, 32, 32), (64, 32, 32), (64, 64, 3, 3), 1],

    [(64, 32, 32), (64, 32, 32), (64, 64, 3, 3), 1],
    [(64, 32, 32), (64, 32, 32), (64, 64, 3, 3), 1],

    [(64, 32, 32), (64, 32, 32), (64, 64, 3, 3), 1],
    [(64, 32, 32), (128, 16, 16), (64, 128, 3, 3), 2],

    [(128, 16, 16), (128, 16, 16), (128, 128, 3, 3), 1],
    [(128, 16, 16), (128, 16, 16), (128, 128, 3, 3), 1],

    [(128, 16, 16), (128, 16, 16), (128, 128, 3, 3), 1],
    [(128, 16, 16), (256, 8, 8), (128, 256, 3, 3), 2],

    [(256, 8, 8), (256, 8, 8), (256, 256, 3, 3), 1],
    [(256, 8, 8), (256, 8, 8), (256, 256, 3, 3), 1],

    [(256, 8, 8), (256, 8, 8), (256, 256, 3, 3), 1],
    [(256, 8, 8), (512, 4, 4), (256, 512, 3, 3), 2],

    [(512, 4, 4), (512, 4, 4), (512, 512, 3, 3), 1],
    [(512, 4, 4), (512, 4, 4), (512, 512, 3, 3), 1],

    [(512, 4, 4), (512, 4, 4), (512, 512, 3, 3), 1],
    [512, 100, None, None],  # ignoring pooling
]

# input_size, output_size, kernel_size=None, stride
vgg13_tinyimagenet_network = [
    [(3, 64, 64), (64, 64, 64), (3, 64, 3, 3), 1],
    [(64, 64, 64), (64, 64, 64), (64, 64, 3, 3), 1],
    # pooling
    [(64, 32, 32), (128, 32, 32), (64, 128, 3, 3), 1],
    [(128, 32, 32), (128, 32, 32), (128, 128, 3, 3), 1],
    # pooling
    [(128, 16, 16), (256, 16, 16), (128, 256, 3, 3), 1],
    [(256, 16, 16), (256, 16, 16), (256, 256, 3, 3), 1],
    # pooling
    [(256, 8, 8), (512, 8, 8), (256, 512, 3, 3), 1],
    [(512, 8, 8), (512, 8, 8), (512, 512, 3, 3), 1],
    # pooling
    [(512, 4, 4), (512, 4, 4), (512, 512, 3, 3), 1],
    [(512, 4, 4), (512, 4, 4), (512, 512, 3, 3), 1],
    #  adaptive pooling
    [25088, 200, None, None]
]

# input_size, output_size, kernel_size=None, stride
vgg11_dvsgesture_network = [
    [(2, 128, 128), (64, 128, 128), (2, 64, 3, 3), 1],
    # pooling
    [(64, 64, 64), (128, 64, 64), (64, 128, 3, 3), 1],
    # pooling
    [(128, 32, 32), (256, 32, 32), (128, 256, 3, 3), 1],
    [(256, 32, 32), (256, 32, 32), (256, 256, 3, 3), 1],
    # pooling
    [(256, 16, 16), (512, 16, 16), (256, 512, 3, 3), 1],
    [(512, 16, 16), (512, 16, 16), (512, 512, 3, 3), 1],
    # pooling
    [(512, 8, 8), (512, 8, 8), (512, 512, 3, 3), 1],
    [(512, 8, 8), (512, 8, 8), (512, 512, 3, 3), 1],
    #  adaptive pooling
    [25088, 11, None, None]
]

# input_size, output_size, kernel_size=None, stride
vgg11_dvscifar_network = [
    [(2, 48, 48), (64, 48, 48), (2, 64, 3, 3), 1],
    # pooling
    [(64, 24, 24), (128, 24, 24), (64, 128, 3, 3), 1],
    # pooling
    [(128, 12, 12), (256, 12, 12), (128, 256, 3, 3), 1],
    [(256, 12, 12), (256, 12, 12), (256, 256, 3, 3), 1],
    # pooling
    [(256, 6, 6), (512, 6, 6), (256, 512, 3, 3), 1],
    [(512, 6, 6), (512, 6, 6), (512, 512, 3, 3), 1],
    # pooling
    [(512, 3, 3), (512, 3, 3), (512, 512, 3, 3), 1],
    [(512, 3, 3), (512, 3, 3), (512, 512, 3, 3), 1],
    # adaptive pooling
    [25088, 10, None, None]
]

data = {
    'cifar10': {
        'CLIF': np.array(
            [24.552, 17.135, 17.304, 13.42, 18.855, 15.632, 15.024, 8.242, 13.895, 12.87, 10.686, 7.517, 8.341, 7.368,
             2.895, 4.127, 3.685]) / 100,
        'LIF': np.array(
            [27.828, 19.727, 23.881, 17.381, 25.658, 19.541, 18.871, 9.919, 17.236, 14.486, 12.91, 8.546, 9.847, 8.887,
             4.845, 6.161, 4.651]) / 100
    },
    'cifar100': {
        'CLIF': np.array(
            [20.606, 12.596, 12.614, 9.659, 15.608, 12.062, 11.114, 7.435, 13.702, 10.801, 8.062, 6.885, 8.734, 6.068,
             2.841, 3.747, 11.773]) / 100,
        'LIF': np.array(
            [17.861, 10.554, 12.224, 12.212, 15.801, 14.979, 13.228, 7.799, 14.914, 12.335, 9.982, 6.734, 10.328, 7.609,
             4.158, 5.472, 15.007]) / 100
    },
    'tinyimagenet': {
        'CLIF': np.array([25.339, 16.132, 15.492, 12.749, 12.49, 9.638, 10.632, 7.403, 12.791, 8.153]) / 100,
        'LIF': np.array([29.265, 22.261, 22.562, 19.926, 18.289, 13.737, 14.664, 10.071, 17.244, 9.031]) / 100
    },
    'dvsgesture': {
        'CLIF': np.array([4.55, 2.29, 1.242, 1.726, 1.058, 0.909, 0.937, 0.408]) / 100,
        'LIF': np.array([4.353, 2.384, 1.564, 1.903, 1.528, 1.174, 1.17, 0.211]) / 100
    },
    'dvscifar': {
        'CLIF': np.array([7.115, 7.113, 8.237, 5.674, 4.843, 2.277, 2.075, 1.543]) / 100,
        'LIF': np.array([8.66, 9.285, 10.229, 7.347, 5.989, 3.562, 2.028, 1.132]) / 100
    }
}

network = {
    "cifar10": resnet_18_cifar10_network,
    "cifar100": resnet_18_cifar100_network,
    "tinyimagenet": vgg13_tinyimagenet_network,
    "dvscifar": vgg11_dvscifar_network,
    "dvsgesture": vgg11_dvsgesture_network,
}

Timestep = {
    "cifar10": 6,
    "cifar100": 6,
    "tinyimagenet": 6,
    "dvscifar": 10,
    "dvsgesture": 20,
}

if __name__ == '__main__':
    # ref: Lemaire, E., et al.An analytical estimation of spiking neural networks energy efficiency. In International Conference on Neural Information Processing, 2022.
    E_ADD = 0.1  # pJ
    E_MUL = 3.1  # pJ

    # task = "cifar10"
    # task = "cifar100"
    # task = "tinyimagenet"
    task = "dvscifar"
    # task = "dvsgesture"
    # neuron = "LIF"  # None means ANN
    # neuron = "CLIF"  # None means ANN
    neuron = None  # None means ANN

    T = Timestep[task]
    net_layers = network[task]
    spike_fr = data[task][neuron] if neuron is not None else [None] * 18
    # CLIF_fr = data[task]["CLIF"]

    mac_op, acc_op, read_in, read_params, read_potential, write_out, write_potential, mac_addr, acc_addr = calculate_all_operations(
        net_layers, neuron, T, spike_fr)

    E_op = acc_op * E_ADD + mac_op * (E_MUL + E_ADD)
    E_addr = acc_addr * E_ADD + mac_addr * (E_MUL + E_ADD)

    E_potential = read_potential * E_func(read_potential) + write_potential * E_func(write_potential)
    E_inout = read_in * E_func(read_in) + write_out * E_func(write_out)
    E_params = read_params * E_func(read_params)

    if task == "dvscifar" or "dvsgesture":
        if neuron == None:
            E_potential *= T
            E_params *= T
            E_inout *= T

            E_op *= T
            E_addr *= T

    print(f'now task is {task}, neuron is {"relu" if neuron == None else neuron}')
    print("mem: E_potential", E_potential)
    print("mem: E_params", E_params)
    print("mem: E_inout", E_inout)

    print("E_op", E_op)
    print("E_addr", E_addr)
