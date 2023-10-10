import torch
import os

if os.name == 'nt':
    torch.classes.load_library('out/build/x64-Release/pytorch-plugin/pytorch-plugin.dll')
else:
    torch.classes.load_library('build/Debug/pytorch-plugin/libpytorch-plugin.so')

torch.cuda.init()

weights = torch.tensor(list(range(10))).to(torch.float).cuda()

model = torch.classes.GGMLPlugin.Model(weights)

tensor = torch.rand(10).cuda()
out = model.forward(tensor)
print(out)
