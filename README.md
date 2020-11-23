
## install requirements
python==3.7
pytorch==1.5
easydict
pillow
建议使用conda工具进行安装，安装好conda之后，执行
`conda activate base`
`conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch`
`conda install pillow`
`pip install easydict`



## 训练
执行`python tools/train.py`
默认位300个epoch，训练结束之后，网络权重保存在./checkpoint/fp_ckpt.pkl

## 量化
执行 `python tools/static_quantize.py`
会将量化的权重保存在./checkpoint/qat_ckpt.pth
为了方便使用，还输出了中间变量和量化后的权重到quant_data.pkl