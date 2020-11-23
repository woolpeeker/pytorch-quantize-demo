import sys
sys.path.append('./')
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

CFG = edict({
    'ckpt': 'checkpoint/fp_ckpt.pth',
    'onnx_file': 'data/zfnet.onnx'
})
CIFAR_ROOT = '/media/HD1/Datasets/cifar'

net = ZFNet()

# Load checkpoint.
print('==> Resuming from checkpoint..')
checkpoint = torch.load(CFG.ckpt)
net.load_state_dict(checkpoint['net'])

# data
transform_test = transforms.Compose([
    RGBPreprocess(64, 2)
])
testset = torchvision.datasets.CIFAR10(
    root=CIFAR_ROOT, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=2)
loader_iter = iter(testloader)
inputs, targets = next(loader_iter)
print('==> export onnx..')
torch.onnx.export(
    model=net,
    args=(inputs,),
    f=CFG.onnx_file,
    input_names=['input'],
    output_names=['output'],
    opset_version=9
)