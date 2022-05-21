import torch
import torch.nn as nn

def loss_geo(case, predictions, batch_locations):
    """
    Takes in the velocity and pressure predictions for the collocation points, calculates gradients 
    w.r.t. axes and computes loss based on the Navier-Stokes eq. and conservation of mass. 

    Parameters: 
    case (class) : The case to which the PINN is applied, contains case-specific information. 
    batch_locations (list) : List of tensors with batch coordinates for x, y (and z). 
    predictions (list) : PINN prediction values for velocity (u, v (,w) and p) at batch_locations

    Returns: 
    loss_geo (float) : Value for the physics-based loss (Navier-Stokes & conservation of mass)
    """
    x, y, *z = batch_locations
    u, v, w, p = predictions

    # Calculate gradients of output parameters w.r.t. coordinates
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]

    if z:
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
        u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
        v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
        v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

        w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
        w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

        p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

    grads = torch.tensor([[torch.min(u_x), torch.max(u_x),torch.min(u_xx), torch.max(u_xx)], [torch.min(u_y), torch.max(u_y),
                          torch.min(u_yy), torch.max(u_yy)], [torch.min(v_x), torch.max(v_x), torch.min(v_xx), torch.max(v_xx)],
                          [torch.min(v_y), torch.max(v_y), torch.min(v_yy), torch.max(v_yy)],
                          [torch.min(p_x), torch.max(p_x), torch.min(p_y), torch.max(p_y)]])
    
    # Loss functions 
    if z:
        loss_1 = u * u_x + v * u_y + w * u_z - 1/case.Reynolds * (u_xx + u_yy + u_zz) + p_x  # X-dir
        loss_2 = u * v_x + v * v_y + w * v_z - 1/case.Reynolds * (v_xx + v_yy + v_zz) + p_y  # Y-dir
        loss_3 = u * w_x + v * w_y + w * w_z - 1/case.Reynolds * (w_xx + w_yy + w_zz) + p_z  # Z-dir
        loss_4 = (u_x + v_y + w_z)  # continuity
    else:
        loss_1 = u * u_x + v * u_y - 1/case.Reynolds * (u_xx + u_yy) + p_x  # X-dir
        loss_2 = u * v_x + v * v_y - 1/case.Reynolds * (v_xx + v_yy) + p_y  # Y-dir
        loss_3 = (u_x + v_y)  # continuity


    print(loss_1.size(), loss_2.size(), loss_3.size())
    # Mean Squared Error Loss
    loss_f = nn.MSELoss()

    loss_geo = loss_f(loss_1, torch.zeros_like(loss_1)) + loss_f(loss_2, torch.zeros_like(loss_2)) + \
           loss_f(loss_3, torch.zeros_like(loss_3)) + loss_f(loss_4, torch.zeros_like(loss_4)) \
        if z else loss_f(loss_1, torch.zeros_like(loss_1)) + \
                             loss_f(loss_2, torch.zeros_like(loss_2)) + \
                             loss_f(loss_3, torch.zeros_like(loss_3))

    return loss_geo, grads


def loss_bnc(case, predictions):
    """
    Take in the predictions for the velocity of the boundary coordinates and calculate loss based on
    no-slip condition (velocity vector = 0). 

    Parameters: 
    case (class) : The case to which the PINN is applied, contains case-specific information. 
    predictions (list) : PINN prediction values for velocity (u, v (,w)) at boundary coordinates

    Return: 
    loss_bnc (float) : Value for the no-slip loss.
    """
    u, v, w, _ = predictions

    # Mean Squard Error Loss
    loss_f = nn.MSELoss()
    print(u.size(), v.size(), w.size())

    loss_bnc = (
        loss_f(u, torch.zeros_like(u))
        + loss_f(v, torch.zeros_like(v))
        + loss_f(w, torch.zeros_like(w))
    ) if case.input_dimension == 3 else loss_f(u, torch.zeros_like(u)) + loss_f(v, torch.zeros_like(v))

    return loss_bnc


def loss_data(case, predictions, solution):
    """
    Take in the predictions for the velocity of the sensor coordinates and calculate loss based on
    difference with sensor values.  

    Parameters: 
    case (class) : The case to which the PINN is applied, contains case-specific information. 
    predictions (list) : PINN prediction values for velocity (u, v (,w)) at boundary coordinates
    solution (list) : Velocity (u, v (,w)) values of the sensors. 

    Return: 
    loss_data (float) : Value for the data loss.
    """
    u, v, w, _ = predictions
    solution = [axis[:, None] for axis in solution]  
    ud, vd, *wd = solution

    print(u.size(), v.size(), w.size(), ud.size(), vd.size())
    # Mean Squared Error Loss
    loss_f = nn.MSELoss()

    loss_data = loss_f(u, ud) + loss_f(v, vd) + loss_f(w, wd) if case.input_dimension == 3 else loss_f(u, ud) + loss_f(v, vd)

    return loss_data