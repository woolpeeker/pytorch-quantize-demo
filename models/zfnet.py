import torch
import torch.nn as nn
from torch.quantization.stubs import QuantStub, DeQuantStub

__all__ = ['ZFNet']

class ConvReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class LinearReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


class ZFNet(nn.Module):
    def __init__(self, quantize=False):
        super(ZFNet, self).__init__()
        self.conv = nn.Sequential(
            # 第一层
            ConvReLU(3, 96, 7, 2, bias=False),
            ConvReLU(96, 160, 5, 2, bias=False),
            # 第三层
            ConvReLU(160, 256, 3, 1, 1, bias=False),
            # 第四层
            ConvReLU(256, 256, 3, 1, 0, bias=False),
            # 第五层
            ConvReLU(256, 256, 3, 1, 0, bias=False),
        )
        # 全连接层
        self.fc = nn.Sequential(
            LinearReLU(256, 256, bias=False),
            nn.Dropout(0.5),
            nn.Linear(256, 10, bias=False),
        )
        self.quanize_flag = quantize
        if quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
    
    def forward(self, img):
        BS = img.shape[0]

        if self.quanize_flag:
            img = self.quant(img)
        
        feature = self.conv(img)
        # print(feature.shape)
        feature = feature.reshape([BS, -1])
        output = self.fc(feature)
        
        if self.quanize_flag:
            output = self.dequant(output)
        
        return output

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvReLU:
                torch.quantization.fuse_modules(m, ['conv', 'relu'], inplace=True)
            if type(m) == LinearReLU:
                torch.quantization.fuse_modules(m, ['linear', 'relu'], inplace=True)
                