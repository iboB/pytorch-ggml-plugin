import torch
import os

if os.name == 'nt':
    torch.classes.load_library('out/build/x64-Release/pytorch-plugin/pytorch-plugin.dll')
else:
    torch.classes.load_library('build/Debug/pytorch-plugin/libpytorch-plugin.so')

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    torch.cuda.init()
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

weights = torch.tensor(list(range(10))).to(torch.float).to(device=device)

model = torch.classes.GGMLPlugin.Model(weights)

tensor = torch.rand(10).to(device=device)
out = model.forward(tensor)

print(out)
