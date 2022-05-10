import torch
import numpy as np
import pymongo
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import time
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy  # PyCharm compatible code
import os
from Verified.scripts.fileReaders import fileReader
from torchsummary import summary


# TODO make the number of layers a variable input

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if not self.inplace:
            return x * torch.sigmoid(x)
        x.mul_(torch.sigmoid(x))  # vermenigvuldigt elke index in tensor met de scalar
        return x


class MySquared(nn.Module):
    def __init__(self, inplace=True):  # wat is het nut van inplace?
        super(MySquared, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.square(x)  # Returns a new tensor with the square of the elements of input


class Net2(nn.Module):

    # The __init__ function stack the layers of the
    # network Sequentially
    def __init__(self, input_n, h_n):
        super(Net2, self).__init__()  # super() alone returns a temporary object of the superclass that then allows
        # you to call that superclass’s methods.
        # geen batchnorm oid tussendoor (?)
        self.main = nn.Sequential(  # 10 lagen, 8 deep layers
            nn.Linear(input_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, 1),  # geef terug 1 waarde voor U. Kan eventueel ook 3? (U,V,p)
        )

    # This function defines the forward rule of
    # output respect to input.
    # def forward(self,x):
    def forward(self, x):
        return self.main(x)

class Net2_2(nn.Module): # TODO: is het nodig om al de layers apart te benoemen?
    def __init__(self, input_n, h_n):
        super(Net2_2, self).__init__()

        self.layer1 = torch.nn.Linear(input_n, h_n)
        self.layer2 = torch.nn.Linear(h_n, h_n)
        self.layer3 = torch.nn.Linear(h_n, h_n)
        self.layer4 = torch.nn.Linear(h_n, h_n)
        self.layer5 = torch.nn.Linear(h_n, h_n)
        self.layer6 = torch.nn.Linear(h_n, h_n)
        self.layer7 = torch.nn.Linear(h_n, h_n)
        self.layer8 = torch.nn.Linear(h_n, h_n)
        self.layer9 = torch.nn.Linear(h_n, h_n)
        self.layer10 = torch.nn.Linear(h_n, 1)
        self.swish = Swish()


    def forward(self, x_in):
        x = self.layer1(x_in)
        x = self.swish(x)
        x = self.layer2(x)
        x = self.swish(x)
        x = self.layer3(x)
        x = self.swish(x)
        x = self.layer4(x)
        x = self.swish(x)
        x = self.layer5(x)
        x = self.swish(x)
        x = self.layer6(x)
        x = self.swish(x)
        x = self.layer7(x)
        x = self.swish(x)
        x = self.layer8(x)
        x = self.swish(x)
        x = self.layer9(x)
        x = self.swish(x)
        x = self.layer10(x)
        return x

class Net3(nn.Module):
    def __init__(self, input_n, h_n):
        super(Net3, self).__init__()

        self.fc1 = torch.nn.Linear(input_n, h_n)
        self.fc2 = torch.nn.Linear(h_n, h_n)
        self.fc3 = torch.nn.Linear(h_n, h_n)
        self.fc4 = torch.nn.Linear(h_n, h_n)
        self.fc5 = torch.nn.Linear(h_n, h_n)
        self.fc6 = torch.nn.Linear(h_n, h_n)
        self.fc7 = torch.nn.Linear(h_n, h_n)
        self.fc8 = torch.nn.Linear(h_n, h_n)
        self.fc9 = torch.nn.Linear(h_n, h_n)
        self.fc10 = torch.nn.Linear(h_n, 1)
        self.encoder1 = torch.nn.Linear(input_n, h_n)
        self.encoder2 = torch.nn.Linear(input_n, h_n)
        self.swish = Swish()

    def forward(self, x_in):
        H = self.swish(self.fc1(x_in))
        U = self.swish(self.encoder1(x_in))
        V = self.swish(self.encoder2(x_in))  #  laag 1

        z = self.swish(self.fc2(H))  # Z_k  # laag 2
        H = torch.mul((1-z), U) + torch.mul(z, V)

        z = self.swish(self.fc3(H))   # laag 3
        H = torch.mul((1-z), U) + torch.mul(z, V)

        z = self.swish(self.fc4(H))  # laag 4
        H = torch.mul((1 - z), U) + torch.mul(z, V)

        z = self.swish(self.fc5(H))  # laag 5
        H = torch.mul((1 - z), U) + torch.mul(z, V)

        z = self.swish(self.fc6(H))  # laag 6
        H = torch.mul((1 - z), U) + torch.mul(z, V)

        z = self.swish(self.fc7(H))  # laag 7
        H = torch.mul((1 - z), U) + torch.mul(z, V)

        z = self.swish(self.fc8(H))  # laag 8
        H = torch.mul((1 - z), U) + torch.mul(z, V)

        z = self.swish(self.fc9(H))  # laag 9
        H = torch.mul((1 - z), U) + torch.mul(z, V)

        H = self.swish(self.fc10(H))  # laag 10
        return H

# class Net3(nn.Module):
#     def __init__(self, input_n, h_n, encoders):
#         super(Net3, self).__init__()
#         self.encoders = encoders
#
#     def forward(self, x):
#         for encoder in self.encoders:
#             x = F.relu(model(x))
#         x = self.models[-1](x)  # don't use relu for last model
#         return x
#
# class MySmallModel(nn.Module):
#     def __init__(self):
#         super(MySmallModel, self).__init__()
#         self.lin = nn.Linear(10, 10)
#
#     def forward(self, x):
#         x = self.lin(x)
#         return x

# models = nn.ModuleList()
# for _ in range(10):
#     models.append(MySmallModel())
#
# model = MyHugeModel(models)
# x = torch.randn(1, 10)
# output = model(x)


class Net2P(nn.Module):  # TODO: verwijderen als stenose2D opnieuw is getraind

    # The __init__ function stack the layers of the
    # network Sequentially
    def __init__(self, input_n, h_n):
        super(Net2P, self).__init__()  # super() alone returns a temporary object of the superclass that then allows
        # you to call that superclass’s methods.
        # geen batchnorm oid tussendoor (?)
        self.main = nn.Sequential(  # 10 lagen, 8 deep layers
            nn.Linear(input_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, 1),  # geef terug 1 waarde voor U. Kan eventueel ook 3? (U,V,p)
        )

    # This function defines the forward rule of
    # output respect to input.
    # def forward(self,x):
    def forward(self, x):
        return self.main(x)

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)

def NNreader(case, device, data_u = "data_u", data_v="data_v", data_w="data_w", data_p="data_p"):

    file = case.mesh_file

    x, y, z, _ = fileReader(file, case.input_n, mesh=True)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)

    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    z = torch.tensor(z).to(device)

    net2_u = Net2(case.input_n, case.h_n).to(device)
    net2_v = Net2(case.input_n, case.h_n).to(device)
    net2_w = Net2(case.input_n, case.h_n).to(device)
    net2_p = Net2P(case.input_n, case.h_n).to(device)  # TODO opletten! Sommige NNs hadden een extra laag voor P

    if device == "cuda":
        x = x.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        z = z.type(torch.cuda.FloatTensor)
    else:
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        z = z.type(torch.FloatTensor)
    # print(summary(net2_p, (2, 128)))

    net2_u.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_u + ".pt"))
    net2_v.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_v + ".pt"))
    if case.input_n == 3:
        net2_w.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_w + ".pt"))
    net2_p.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_p + ".pt"))

    net_in = torch.cat((x, y, z), 1) if case.input_n == 3 else torch.cat((x, y), 1)
    u = net2_u(net_in)
    u = u.view(len(u), -1)
    v = net2_v(net_in)
    v = v.view(len(v), -1)
    P = net2_p(net_in)
    P = P.view(len(P), -1)
    if case.input_n == 3:
        w = net2_w(net_in)
        w = w.view(len(w), -1)
        return u, v, P, w, net2_u, net2_v, net2_p, net2_w
    return u, v, P, net2_u, net2_v, net2_p

def NNreader2(case, device, data_1="data_w", data_2="data_p", flag_P=False):

    file = case.mesh_file

    x, y, z, _ = fileReader(file, case.input_n, mesh=True)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)

    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    z = torch.tensor(z).to(device)

    net1 = Net2(case.input_n, case.h_n).to(device)
    net2 = Net2(case.input_n, case.h_n).to(device)
    if flag_P:
        net2 = Net2P(case.input_n, case.h_n).to(device)

    if device == "cuda":
        x = x.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        z = z.type(torch.cuda.FloatTensor)
    else:
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        z = z.type(torch.FloatTensor)


    net1.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_1 + ".pt"))
    net2.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_2 + ".pt"))

    net_in = torch.cat((x, y, z), 1) if case.input_n == 3 else torch.cat((x, y), 1)
    out1 = net1(net_in)
    out1 = out1.view(len(out1), -1)
    out2 = net2(net_in)
    out2 = out2.view(len(out2), -1)

    return out1, out2, net1, net2