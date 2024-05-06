# CLIF: Complementary Leaky Integrate-and-Fire Neuron for Spiking Neural Networks (**[ICML2024](https://arxiv.org/pdf/2402.04663)**)
![avatar](/main_fig.pdf)

## Dependencies
- Python 3
- PyTorch, torchvision
- spikingjelly 0.0.0.0.12
- Python packages: `pip install tqdm progress torchtoolbox thop`


## Training
We use single GTX4090 GPU for running all the experiments. Multi-GPU training is not supported in the current codes.


### Setup
CIFAR-10, CIFAR-100, Tiny-Imagenet, DVS-CIFAR10, and DVS-Gesture:

    # CIFAR-10
	python train_BPTT.py -data_dir ./data_dir -dataset cifar10 -model spiking_resnet18 -T_max 200 -epochs 200 -weight_decay 5e-5 -neuron CLIF
    
    # CIFAR-100
    python train_BPTT.py -data_dir ./data_dir -dataset cifar100 -model spiking_resnet18 -T_max 200 -epochs 200 -neuron CLIF
    
    # Tiny-Imagenet
    python train_BPTT.py -data_dir ./data_dir -dataset tiny_imagenet -model spiking_vgg13_bn -neuron CLIF
       
    # DVS-CIFAR10
	python train_BPTT.py -data_dir ./data_dir -dataset DVSCIFAR10 -T 10 -drop_rate 0.3 -model spiking_vgg11_bn -lr=0.05  -mse_n_reg -neuron CLIF
	
	# DVS-Gesture
    python train_BPTT.py -data_dir ./data_dir -dataset dvsgesture -model spiking_vgg11_bn -T 20 -b 16 -drop_rate 0.4  -neuron CLIF

If you change the neuron, you can directly switch to ``LIF`` or ``PLIF`` by modifying the hyperparameters after ``-neuron``.

For example to setup LIF neuron for CIFAR-10 task:

    # LIF neuron for CIFAR-10
	python train_BPTT.py -data_dir ./data_dir -dataset cifar10 -model spiking_resnet18 -amp -T_max 200 -epochs 200 -weight_decay 5e-5 -neuron LIF
    


## Inference
The inference setup could refer file: ``run_inference_script``
