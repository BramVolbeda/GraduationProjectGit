import torch
import torch.nn as nn

# Swish activation function
class Swish(nn.Module):
    """
    Swish activation function. Currently not available as standard PyTorch module. 
    More info at https://medium.com/@neuralnets/swish-activation-function-by-google-53e1ea86f820. 

    """
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if not self.inplace:
            return x * torch.sigmoid(x)
        x.mul_(torch.sigmoid(x)) 
        return x


class Net(nn.Module):
    """
    The architecture of the Neural Network used in the PINN framework. Fully connected NN with encoders U and V
    connected to each hidden layer, based on  based on https://arxiv.org/pdf/2001.04536.pdf. 

    Parameters: 
    input_n (int) : Number of input dimensions (2D or 3D). 
    h_sizes (int) : Number of hidden layers.
    h_n (int) : Number of neurons in each hidden layer.  

    """
    def __init__(self, input_n, h_sizes, h_n):
        super(Net, self).__init__()
        
        self.swish = Swish()
        self.input_layer = nn.Linear(input_n, h_n)
        self.hidden = nn.ModuleList()
        for _ in range(h_sizes):
            self.hidden.append(nn.Linear(h_n, h_n))

        self.output_layer = nn.Linear(h_n, 1)
        self.encoder1 = nn.Linear(input_n, h_n)
        self.encoder2 = nn.Linear(input_n, h_n)

    def forward(self, x_in):
        H = self.swish(self.input_layer(x_in))
        U = self.swish(self.encoder1(x_in))
        V = self.swish(self.encoder2(x_in)) 

        for layer in self.hidden:
            z = self.swish(layer(H))
            H = torch.mul((1 - z), U) + torch.mul(z, V)

        H = self.swish(self.output_layer(H))
        return H

def init_normal(m):
    """
    'He initialization', as described in https://ieeexplore.ieee.org/document/7410480
    """
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
