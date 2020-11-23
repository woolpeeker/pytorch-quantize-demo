from pathlib import Path
import numpy as np
from PIL import Image


imgs_root = './cifar_img_samples'

def convert_img_to_txt(img_file, txt_file):
    img = Image.open(img_file).convert('RGB')
    img = np.array(img).transpose([2, 0, 1])
    data = (np.round(img / 2) -64)
    np.savetxt(str(txt_file), data.reshape([-1]).aastype(int), '%d')

if __name__ == '__main__':
    files = Path(imgs_root).glob('*.png')
    for img_file in files:
        txt_file = img_file.with_suffix('.txt')
        convert_img_to_txt(str(img_file), str(txt_file))

