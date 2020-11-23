'''Train CIFAR10 with PyTorch.'''
import sys
sys.path.append('./')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from easydict import EasyDict as edict

from models import ZFNet
from modules.utils import progress_bar, RGBPreprocess
from modules.quantize.observer import ExpScaleMinMaxObserver
from modules.quantize.hooks import SaveIntermediateHook
import torch.quantization
from pathlib import Path

import pickle

CIFAR_ROOT = '/media/HD1/Datasets/cifar'
MAX_EPOCH = 150
OUT_CKPT = './checkpoint/qat_ckpt.pth'
qat_cfg = edict({
    'input_cfg': {
        'int_bit': 3,
        'float_bit': 4
    },
    'weight_cfg': {
        'int_bit': 1,
        'float_bit': 8
    },
    'output_cfg': {
        'int_bit': 3,
        'float_bit': 4
    }
})

def lr_lambda(epoch):
    if epoch < start_epoch+50:
        return 1
    elif epoch < start_epoch+100:
        return 0.1
    else:
        return 0.01

parser = argparse.ArgumentParser(description='static_quantize')
parser.add_argument('--resume', '-r', type=str, default='checkpoint/fp_ckpt.pth', help='resume from checkpoint')
args = parser.parse_args()

device = 'cpu' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    RGBPreprocess(64, 2)
])

transform_test = transforms.Compose([
    RGBPreprocess(64, 2)
])

trainset = torchvision.datasets.CIFAR10(
    root=CIFAR_ROOT, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=CIFAR_ROOT, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ZFNet(quantize=True)

# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.resume)
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

net = net.to(device)

def test(epoch):
    global best_acc
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'
                         % (100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, OUT_CKPT)
        best_acc = acc

net.fuse_model()
net.qconfig = torch.quantization.QConfig(
    activation=ExpScaleMinMaxObserver.with_args(dtype=torch.quint8, reduce_range=True),
    weight=ExpScaleMinMaxObserver.with_args(dtype=torch.qint8, reduce_range=False, qscheme=torch.per_tensor_symmetric)
)
torch.quantization.prepare(net, inplace=True)
test(0)
torch.quantization.convert(net, inplace=True)
test(1)
pass
hook = SaveIntermediateHook()

for name, module in net.named_modules():
    if isinstance(module, torch.nn.quantized.modules.Quantize):
        module.register_forward_hook(hook.get_hook(name))
    elif isinstance(module, torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d):
        module.register_forward_hook(hook.get_hook(name))
    elif isinstance(module, torch.nn.intrinsic.quantized.modules.linear_relu.LinearReLU):
        module.register_forward_hook(hook.get_hook(name))
    elif isinstance(module, torch.nn.quantized.modules.linear.Linear):
        module.register_forward_hook(hook.get_hook(name))

testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)
loader_iter = iter(testloader)
inputs, targets = next(loader_iter)
inputs, targets = next(loader_iter)
print('targets:', targets)
inputs, targets = inputs.to(device), targets.to(device)
outputs = net(inputs)

out_data = hook.output_data()
pickle.dump(out_data, open('quant_data.pkl', 'wb'))

def save_processed_images(num):
    dir = Path('data/processed_images/')
    dir.mkdir(exist_ok=True, parents=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    loader_iter = iter(testloader)
    fp = open(dir/'labels.txt', 'w')
    for i in range(num):
        hook.reset()
        inputs, targets = next(loader_iter)
        net(inputs)
        out_data = hook.output_data()
        in_data = out_data['quant']['output']['data']
        in_data = in_data.reshape([-1])
        img_txt = dir/f'{i:03d}.txt'
        np.savetxt(str(img_txt), in_data, fmt='%d', delimiter=' ')
        fp.write(f'{img_txt.name}, {targets[0]}\n')
        
# save_processed_images(10)
