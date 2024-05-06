import argparse
import collections
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from spikingjelly.clock_driven import functional, surrogate as surrogate_sj
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchtoolbox.transform import Cutout
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

from models import spiking_resnet, vgg_model, spiking_vgg_bn
from modules import neuron
from modules import surrogate as surrogate_self
from utils import Bar, AverageMeter, accuracy, static_cifar_util, augmentation
from utils.augmentation import ToPILImage, Resize, ToTensor
from utils.cifar10_dvs import CIFAR10DVS, DVSCifar10
from utils.data_loaders import TinyImageNet
from thop import profile

# from torchtoolbox.transform import Cutout

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('-seed', default=2022, type=int)
    parser.add_argument('-name', default='', type=str, help='specify a name for the checkpoint and log files')
    parser.add_argument('-T', default=6, type=int, help='simulating time-steps')
    parser.add_argument('-tau', default=2.0, type=float, help='a hyperparameter for the LIF model')
    parser.add_argument('-b', default=128, type=int, help='batch size')

    parser.add_argument('-j', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default='./data', help='directory of the used dataset')
    parser.add_argument('-dataset', default='cifar10', type=str,
                        help='should be cifar10, cifar100, DVSCIFAR10, dvsgesture, or imagenet')
    parser.add_argument('-out_dir', type=str, default='./logs_infer', help='root dir for saving logs and checkpoint')
    parser.add_argument('-surrogate', default='rectangle', type=str,
                        help='used surrogate function. should be sigmoid, rectangle, or triangle')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-pre_train', type=str, help='load a pretrained model. used for imagenet')
    parser.add_argument('-amp', default=True, type=bool, help='automatic mixed precision training')

    parser.add_argument('-model', type=str, default='spiking_vgg11_bn', help='use which SNN model')
    parser.add_argument('-drop_rate', type=float, default=0.0, help='dropout rate. used for DVSCIFAR10')
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-loss_lambda', type=float, default=0.05,  help='the scaling factor for the MSE term in the loss')
    parser.add_argument('-mse_n_reg', action='store_true', help='loss function setting')
    parser.add_argument('-loss_means', type=float, default=1.0, help='used in the loss function when mse_n_reg=False')
    parser.add_argument('-save_init', action='store_true', help='save the initialization of parameters')
    parser.add_argument('-neuron_model', type=str, default='LIF', help='save the initialization of parameters')
    parser.add_argument('-multiple_step', type=bool, default=False, help='whether multiple steps')
    parser.add_argument('-cutupmix_auto', action='store_true', help='cutupmix autoaugmentation for cifar and tinyimagenet')
    parser.add_argument('-label_smoothing', type=float, default=0.0, help='label_smoothing for cross entropy')

    args = parser.parse_args()
    print(args)

    _seed_ = args.seed
    random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    np.random.seed(_seed_)

    ##########################################################
    # data loading
    ##########################################################
    in_dim = None
    c_in = None
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':

        c_in = 3
        if args.dataset == 'cifar10':
            dataloader = torchvision.datasets.CIFAR10
            num_classes = 10
            normalization_mean = (0.4914, 0.4822, 0.4465)
            normalization_std = (0.2023, 0.1994, 0.2010)
        elif args.dataset == 'cifar100':
            dataloader = torchvision.datasets.CIFAR100
            num_classes = 100
            normalization_mean = (0.5071, 0.4867, 0.4408)
            normalization_std = (0.2675, 0.2565, 0.2761)
        else:
            raise NotImplementedError

        if args.cutupmix_auto:
            mixup_transforms = []
            mixup_transforms.append(static_cifar_util.RandomMixup(num_classes, p=1.0, alpha=0.2))
            mixup_transforms.append(static_cifar_util.RandomCutmix(num_classes, p=1.0, alpha=1.))
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
            collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

            transform_train = static_cifar_util.ClassificationPresetTrain(mean=normalization_mean,
                                                                          std=normalization_std,
                                                                          interpolation=InterpolationMode('bilinear'),
                                                                          auto_augment_policy='ta_wide',
                                                                          random_erase_prob=0.1)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(normalization_mean, normalization_std),
            ])

            train_set = dataloader(
                root=args.data_dir,
                train=True,
                transform=transform_train,
                download=False, )

            test_set = dataloader(
                root=args.data_dir,
                train=False,
                transform=transform_test,
                download=False)

            train_data_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=args.b,
                collate_fn=collate_fn,
                shuffle=True,
                drop_last=True,
                num_workers=args.j,
                pin_memory=True
            )

            test_data_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=args.b,
                shuffle=False,
                drop_last=False,
                num_workers=args.j,
                pin_memory=True
            )
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                Cutout(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(normalization_mean, normalization_std),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(normalization_mean, normalization_std),
            ])

            trainset = dataloader(root=args.data_dir, train=True, download=True, transform=transform_train)
            train_data_loader = DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.j)

            testset = dataloader(root=args.data_dir, train=False, download=False, transform=transform_test)
            test_data_loader = DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j)

    elif args.dataset == 'DVSCIFAR10':
        c_in = 2
        num_classes = 10

        transform_train = transforms.Compose([
            ToPILImage(),
            Resize(48),
            # augmentation.Cutout(),
            # augmentation.RandomSizedCrop(48),
            augmentation.RandomHorizontalFlip(),
            augmentation.RandomRotation(),
            ToTensor(),

        ])

        transform_test = transforms.Compose([
            ToPILImage(),
            Resize(48),
            ToTensor(),
        ])

        trainset = CIFAR10DVS(args.data_dir, train=True, use_frame=True, frames_num=args.T, split_by='number',
                              normalization=None, transform=transform_train)
        testset = CIFAR10DVS(args.data_dir, train=False, use_frame=True, frames_num=args.T, split_by='number',
                             normalization=None, transform=transform_test)

        train_data_loader = DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.j)
        test_data_loader = DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j)

    elif args.dataset == 'DVSCIFAR10-pt':
        c_in = 2
        num_classes = 10
        in_dim = 48
        train_path = args.data_dir + '/train'
        val_path = args.data_dir + '/test'
        trainset = DVSCifar10(root=train_path, transform=True)
        testset = DVSCifar10(root=val_path, transform=False)
        train_data_loader = DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.j)
        test_data_loader = DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j, drop_last=False,
                                      pin_memory=True)

    elif args.dataset == 'dvsgesture':
        c_in = 2
        num_classes = 11
        in_dim = 128

        trainset = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T,
                                 split_by='number')
        train_data_loader = DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.j, drop_last=True,
                                       pin_memory=True)

        testset = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T,
                                split_by='number')
        test_data_loader = DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j, drop_last=False,
                                      pin_memory=True)

    elif args.dataset == 'tiny_imagenet':
        data_dir = args.data_dir
        c_in = 3
        num_classes = 200
        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])

        transoform_list = [
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(0.5),
        ]

        if args.cutupmix_auto:
            transoform_list.append(autoaugment.AutoAugment())

        transoform_list += [transforms.ToTensor(), normalize]

        train_transforms = transforms.Compose(transoform_list)
        val_transforms = transforms.Compose([transforms.ToTensor(), normalize, ])

        train_data = TinyImageNet(data_dir, train=True, transform=train_transforms)
        test_data = TinyImageNet(data_dir, train=False, transform=val_transforms)

        train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.b, shuffle=True,
            num_workers=args.j, pin_memory=True)

        test_data_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.b, shuffle=False,
            num_workers=args.j, pin_memory=True)
    else:
        raise NotImplementedError

    ##########################################################
    # model preparing
    ##########################################################
    if args.surrogate == 'sigmoid':
        surrogate_function = surrogate_sj.Sigmoid()
    elif args.surrogate == 'rectangle':
        surrogate_function = surrogate_self.Rectangle()
    elif args.surrogate == 'triangle':
        surrogate_function = surrogate_sj.PiecewiseQuadratic()
    else:
        raise NotImplementedError

    if args.neuron_model == 'LIF':
        neuron_model = neuron.BPTTNeuron
    elif args.neuron_model == 'CLIF':
        neuron_model = neuron.ComplementaryLIFNeuron
    elif args.neuron_model == 'PLIF':
        neuron_model = neuron.PLIFNeuron
    elif args.neuron_model == 'relu':
        neuron_model = neuron.ReLU
        args.T = 1
    else:
        raise NotImplementedError

    if args.model in ['spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50', 'spiking_resnet101',
                      'spiking_resnet152']:
        net = spiking_resnet.__dict__[args.model](neuron=neuron_model, num_classes=num_classes,
                                                  neuron_dropout=args.drop_rate,
                                                  tau=args.tau, surrogate_function=surrogate_function, c_in=c_in,
                                                  fc_hw=1)
        print('using Resnet model.')
    elif args.model in ['spiking_vgg11_bn', 'spiking_vgg13_bn', 'spiking_vgg16_bn', 'spiking_vgg19_bn']:
        net = spiking_vgg_bn.__dict__[args.model](neuron=neuron_model, num_classes=num_classes,
                                                  neuron_dropout=args.drop_rate,
                                                  tau=args.tau, surrogate_function=surrogate_function, c_in=c_in,
                                                  fc_hw=in_dim if in_dim else None)
        print('using Spiking VGG model.')
    elif args.model in ['vggsnn', 'snn5_noAP']:  # snn5_noAP use for statistical experiment
        net = vgg_model.__dict__[args.model](neuron=neuron_model, num_classes=num_classes,
                                             neuron_dropout=args.drop_rate,
                                             tau=args.tau, surrogate_function=surrogate_function, c_in=c_in,
                                             fc_hw=in_dim if in_dim else None)
        print('using Spiking VGG model.')
    else:
        raise NotImplementedError

    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.cuda()
    #
    # ##########################################################
    # # optimizer preparing
    # ##########################################################
    # if args.opt == 'SGD':
    #     optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
    #                                 weight_decay=args.weight_decay)
    # elif args.opt == 'AdamW':
    #     optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # else:
    #     raise NotImplementedError(args.opt)
    #
    # if args.lr_scheduler == 'StepLR':
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # elif args.lr_scheduler == 'CosALR':
    #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    # else:
    #     raise NotImplementedError(args.lr_scheduler)
    #
    # scaler = None
    # if args.amp:
    #     scaler = amp.GradScaler()

    ##########################################################
    # loading models from checkpoint
    ##########################################################

    max_test_acc = 0

    if args.resume:
        print('resuming...')
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        print('start epoch:', start_epoch, ', max test acc:', max_test_acc)

    if args.pre_train:
        checkpoint = torch.load(args.pre_train, map_location='cpu')
        state_dict2 = collections.OrderedDict([(k, v) for k, v in checkpoint['net'].items()])
        net.load_state_dict(state_dict2)
        print('use pre-trained model, max test acc:', checkpoint['max_test_acc'])

    ##########################################################
    # output setting
    ##########################################################
    out_dir = os.path.join(args.out_dir,
                           f'inference_{args.dataset}_{args.model}_{args.name}_T{args.T}_tau{args.tau}_bs{args.b}')

    if args.neuron_model != 'LIF':
        out_dir += f'_{args.neuron_model}_'

    # if args.lr_scheduler == 'CosALR':
    #     out_dir += f'CosALR_{args.T_max}'
    # elif args.lr_scheduler == 'StepLR':
    #     out_dir += f'StepLR_{args.step_size}_{args.gamma}'
    # else:
    #     raise NotImplementedError(args.lr_scheduler)

    if args.amp:
        out_dir += '_amp'

    if args.cutupmix_auto:
        out_dir += '_cutupmix_auto'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print(out_dir)

    # save the initialization of parameters
    if args.save_init:
        checkpoint = {
            'net': net.state_dict(),
            'epoch': 0,
            'max_test_acc': 0.0
        }
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_0.pth'))

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    ##########################################################
    #  testing
    ##########################################################
    criterion_mse = nn.MSELoss()

    start_time = time.time()
    net.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(test_data_loader))

    test_loss = 0
    test_acc = 0
    test_samples = 0
    batch_idx = 0
    with torch.no_grad():
        for data in test_data_loader:
            if args.dataset == 'SHD':
                frame, label, _ = data
            else:
                frame, label = data

            batch_idx += 1
            if (args.dataset != 'DVSCIFAR10'):
                frame = frame.float().cuda()

                if args.dataset == 'dvsgesture' or args.dataset == "SHD" or args.dataset == "DVSCIFAR10-pt":
                    frame = frame.transpose(0, 1)
            label = label.cuda()

            t_step = args.T
            if args.dataset == 'SHD':
                t_step = len(frame)

            label_real = torch.cat([label for _ in range(t_step)], 0)
            # print(t_step)

            out_all = []
            for t in range(t_step):

                if (args.dataset == 'DVSCIFAR10'):
                    input_frame = frame[t].float().cuda()
                elif args.dataset == 'dvsgesture' or args.dataset == "SHD" or args.dataset == "DVSCIFAR10-pt":
                    input_frame = frame[t]
                else:
                    input_frame = frame
                if t == 0:
                    out_fr = net(input_frame)
                    total_fr = out_fr.clone().detach()
                    out_all.append(out_fr)
                else:
                    out_fr = net(input_frame)
                    total_fr += out_fr.clone().detach()
                    out_all.append(out_fr)

            out_all = torch.cat(out_all, 0)
            # Calculate the loss
            if args.loss_lambda > 0.0:  # the loss is a cross entropy term plus a mse term
                if args.mse_n_reg:  # the mse term is not treated as a regularizer
                    label_one_hot = F.one_hot(label_real, num_classes).float()
                else:
                    label_one_hot = torch.zeros_like(out_all).fill_(args.loss_means).to(out_all.device)
                mse_loss = criterion_mse(out_all, label_one_hot)
                loss = ((1 - args.loss_lambda) * F.cross_entropy(out_all, label_real,
                                                                 label_smoothing=args.label_smoothing) + args.loss_lambda * mse_loss)
            else:  # the loss is just a cross entropy term
                loss = F.cross_entropy(out_all, label_real, label_smoothing=args.label_smoothing)
            total_loss = loss

            test_samples += label.numel()
            test_loss += total_loss.item() * label.numel()
            test_acc += (total_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
            losses.update(total_loss, input_frame.size(0))
            top1.update(prec1.item(), input_frame.size(0))
            top5.update(prec5.item(), input_frame.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx,
                size=len(test_data_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
    bar.finish()

    test_loss /= test_samples
    test_acc /= test_samples

    ############### calculation  ###############

    total_time = time.time() - start_time
    info = f'test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={total_time}'
    print(info)
    mem_cost = "after one epoch: %fGB" % (torch.cuda.max_memory_cached(0) / 1024 / 1024 / 1024)
    print(mem_cost)



    B, C, H, W = input_frame.shape
    optimal_batch_size = B
    dummy_input = torch.randn(optimal_batch_size, C, H, W, dtype=torch.float).cuda()

    repetitions = 100
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = net(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * optimal_batch_size) / total_time

    print("Final Throughput:", Throughput)
    with open(os.path.join(out_dir, 'args.txt'), 'a+', encoding='utf-8') as args_txt:
        args_txt.write("\n")
        args_txt.write(info + "\n")
        args_txt.write(mem_cost + "\n")

        args_txt.write("Throughput" + "\n")
        args_txt.write(str(Throughput) + "\n")



if __name__ == '__main__':
    main()
