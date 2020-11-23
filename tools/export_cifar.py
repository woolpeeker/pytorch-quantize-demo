import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from PIL import Image

CIFAR_ROOT = '/media/HD1/Datasets/cifar'
OUTPUT_ROOT = './data/cifar'
TRAIN = False

if __name__ == '__main__':
    if TRAIN:
        image_dir = Path(OUTPUT_ROOT) / 'images_train'
        txt_file = Path(OUTPUT_ROOT) / 'train_labels.txt'
    else:
        image_dir = Path(OUTPUT_ROOT) / 'images_val'
        txt_file = Path(OUTPUT_ROOT) / 'val_labels.txt'

    trainset = torchvision.datasets.CIFAR10(
        root=CIFAR_ROOT, train=True, download=False
    )

    image_dir.mkdir(parents=True, exist_ok=True)
    fp = open(txt_file, 'w')

    for idx, (image, label) in enumerate(trainset):
        out_file = str(image_dir / f'{idx:06d}.png')
        image.save(out_file)
        fp.write('%d\n' % label)
