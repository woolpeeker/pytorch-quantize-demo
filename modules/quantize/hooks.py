
from collections import OrderedDict

class _Hook:
    def __init__(self, name:str, data:dict, save_weight: bool =True):
        self.name = name
        self.data = data
        self.save_weight = save_weight
    
    def __call__(self, module, inp, out):
        self.data.setdefault('output', []).append(out)
        if self.save_weight and hasattr(module, 'weight'):
            self.data['weight'] = module.weight()

class SaveIntermediateHook:
    def __init__(self):
        self.data = OrderedDict()
    
    def get_hook(self, name):
        self.data[name] = {}
        return _Hook(name, self.data[name])

    def output_data(self):
        OUTPUT = OrderedDict()
        for name, layer_data in self.data.items():
            OUTPUT[name] = {}
            if 'output' in layer_data:
                out_tensor = layer_data['output'][0]
                out_scale = out_tensor.q_scale()
                out_zero_point = out_tensor.q_zero_point()
                out_data = out_tensor.dequantize() / out_scale + out_zero_point
                OUTPUT[name]['output'] = {
                    'scale': out_scale,
                    'zero_point': out_zero_point,
                    'data': out_data.numpy()
                }
            if 'weight' in layer_data:
                weight = layer_data['weight']
                weight_scale = weight.q_scale()
                weight_zero_point = weight.q_zero_point()
                weight_data = weight.dequantize() / weight_scale + weight_zero_point
                OUTPUT[name]['weight'] = {
                    'scale': weight_scale,
                    'zero_point': weight_zero_point,
                    'data': weight_data.numpy()
                }
        return OUTPUT
    
    def reset(self):
        for name, layer_data in self.data.items():
            layer_data.clear()
