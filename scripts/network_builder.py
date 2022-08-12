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


###############################
# Old Net2 class to load old .pt files
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