import torch

def weight_annealing_algorithm(networks, loss_eqn, loss_bnc, loss_data, lambda_bc, lambda_data, weight_factor_list, alpha):
    """
    Execution of the learning weight annealing algorithm as proposed by https://arxiv.org/pdf/2001.04536.pdf. 

    Based on the gradients of the NNs w.r.t. the different losses a new estimate is calculated for the respective
    loss weights. The ratio between the maximum gradient value w.r.t. the equation loss and the mean of the gradients
    w.r.t. the boundary / data loss determines the new proposal for the weight factors. The actual value is updated via 
    a moving average algorithm, with alpha ([0 1]) determing the relative relevance of the current values (alpha=0) versus 
    the proposed value (alpha=1). Gradients  of a NN are computed in PyTorch during a loss.backward() call, so the seperate 
    loss values are given as an input.     

    Parameters:
    networks (list) : List of the NNs defined in the PINN framework (u, v, w and p). To reduce computation time, the gradients
                        of one of these networks are calculated and stored. Current default is net_u (highest variance). 
    loss_eqn (float) : Value of the equation loss (based on Navier-Stokes and conservation of mass). 
    loss_bnc (float) : Value of the boundary loss (based on no-slip condition). 
    loss_data (float) : Value of the data loss (based on sensor data). 
    lambda_bc (float) : Current value for the boundary loss weighting factor. 
    lambda_data (float) : Current value of the data loss weighting factor. 
    weight_factor_list (list) : Stores the new weight factor values (boundary index 0, data index 1). 
    alpha (float) : Parameter influencing the relative importance of the proposed weight factor versus the current factor. 

    Returns: 
    weight_factor_list (list) : Updated list with new weight factors added. 
    lambda_bc (float) : New value for the boundary loss weighting factor. 
    lambda_data (float) : New value for the data loss weighting factor. 
    """
    
    print('executing learning weight annealing algorithm..')
    net_u, _, _, _ = networks

    max_grad_geo_list = []
    mean_grad_bc_list = []
    mean_grad_data_list = []
    loss_eqn.backward(retain_graph=True)

    layer = 0
    for name, param in net_u.named_parameters(prefix=''):
        if "weight" in name:
            layer += 1
            max_grad_geo_list.append(torch.max(abs(param.grad)))

    loss_bnc.backward(retain_graph=True)
    layer = 0
    for name, param in net_u.named_parameters(prefix=''):
        if "weight" in name:
            layer += 1
            mean_grad_bc_list.append(torch.mean(abs(param.grad)))

    loss_data.backward(retain_graph=True)

    layer = 0
    for name, param in net_u.named_parameters(prefix=''):
        if "weight" in name:
            layer += 1
            mean_grad_data_list.append(torch.mean(abs(param.grad)))

    max_grad_geo = torch.max(torch.tensor(max_grad_geo_list))
    mean_grad_bc = torch.mean(torch.tensor(mean_grad_bc_list))
    mean_grad_data = torch.mean(torch.tensor(mean_grad_data_list))

    lambda_bc_adaptive = max_grad_geo / mean_grad_bc
    lambda_data_adaptive = max_grad_geo / mean_grad_data

    # Moving average
    lambda_bc = lambda_bc * (1 - alpha) + lambda_bc_adaptive * alpha
    lambda_data = lambda_data * (1 - alpha) + lambda_data_adaptive * alpha

    #Store new values of the weight factors
    weight_factor_list[0].append(lambda_bc)  
    weight_factor_list[1].append(lambda_data)  

    return weight_factor_list, lambda_bc, lambda_data