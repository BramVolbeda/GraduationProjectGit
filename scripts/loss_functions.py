import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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
    
    loss_bnc_f = (
        loss_f(u, torch.zeros_like(u))
        + loss_f(v, torch.zeros_like(v))
        + loss_f(w, torch.zeros_like(w))
    ) if case.input_dimension == 3 else loss_f(u, torch.zeros_like(u)) + loss_f(v, torch.zeros_like(v))

    return loss_bnc_f


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

    # Mean Squared Error Loss
    loss_f = nn.MSELoss()
    if case.input_dimension == 3: 
        wd = wd[0]
        loss_data = loss_f(u, ud) + loss_f(v, vd) + loss_f(w, wd) 
    else: 
        loss_data = loss_f(u, ud) + loss_f(v, vd)
    
    return loss_data


def loss_geo_v2(case, predictions, batch_locations, means, stds, flag_vtk=False, flag_values=False):
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
        z_ = z[0]
        u_z = torch.autograd.grad(u, z_, grad_outputs=torch.ones_like(z_), create_graph=True, only_inputs=True)[0]
        u_zz = torch.autograd.grad(u_z, z_, grad_outputs=torch.ones_like(z_), create_graph=True, only_inputs=True)[0]
        v_z = torch.autograd.grad(v, z_, grad_outputs=torch.ones_like(z_), create_graph=True, only_inputs=True)[0]
        v_zz = torch.autograd.grad(v_z, z_, grad_outputs=torch.ones_like(z_), create_graph=True, only_inputs=True)[0]

        w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        w_z = torch.autograd.grad(w, z_, grad_outputs=torch.ones_like(z_), create_graph=True, only_inputs=True)[0]
        w_zz = torch.autograd.grad(w_z, z_, grad_outputs=torch.ones_like(z_), create_graph=True, only_inputs=True)[0]

        p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(z_), create_graph=True, only_inputs=True)[0]

    # grads = torch.tensor([[torch.min(u_x), torch.max(u_x),torch.min(u_xx), torch.max(u_xx)], [torch.min(u_y), torch.max(u_y),
    #                       torch.min(u_yy), torch.max(u_yy)], [torch.min(v_x), torch.max(v_x), torch.min(v_xx), torch.max(v_xx)],
    #                       [torch.min(v_y), torch.max(v_y), torch.min(v_yy), torch.max(v_yy)],
    #                       [torch.min(p_x), torch.max(p_x), torch.min(p_y), torch.max(p_y)]])
    grads = [[u_x, u_xx], [u_y, u_yy], [v_x, v_xx], [v_y, v_yy], [p_x, p_y]]
    # Loss functions 
    if z:
        loss_x = 1/stds[0]*(u * u_x) + 1/stds[1]*(v * u_y) + 1/stds[2]*(w * u_z) - 1/case.Reynolds*(1/(stds[0]**2)*u_xx + 
        1/(stds[1]**2)*u_yy + 1/(stds[2]**2)*u_zz - 1/stds[0] * p_x)  # X-dir
        loss_y = 1/stds[0]*(u * v_x) + 1/stds[1]*(v * v_y) + 1/stds[2]*(w * v_z) - 1/case.Reynolds*(1/(stds[0]**2)*v_xx + 
        1/(stds[1]**2)*v_yy + 1/(stds[2]**2)*v_zz - 1/stds[1] * p_y)  # Y-dir
        loss_z = 1/stds[0]*(u * w_x) + 1/stds[1]*(v * w_y) + 1/stds[2]*(w * w_z) - 1/case.Reynolds*(1/(stds[0]**2)*w_xx + 
        1/(stds[1]**2)*w_yy + 1/(stds[2]**2)*w_zz - 1/stds[2] * p_z)  # Z-dir
        loss_c = case.U_scale/(stds[0]*case.L_scale)*u_x + case.U_scale/(stds[1]*case.L_scale)*v_y + case.U_scale/(stds[2]*case.L_scale)*w_z  # continuity
    else:
        loss_x = 1/stds[0]*(u * u_x) + 1/stds[1]*(v * u_y) - 1/case.Reynolds*(1/(stds[0]**2)*u_xx + 1/(stds[1]**2)*u_yy - 1/stds[0]*p_x)  # X-dir
        loss_y = 1/stds[0]*(u * v_x) + 1/stds[1]*(v * v_y) - 1/case.Reynolds*(1/(stds[0]**2)*v_xx + 1/(stds[1]**2)*v_yy - 1/stds[1]*p_y)  # Y-dir
        loss_c = case.U_scale/(stds[0]*case.L_scale)*u_x + case.U_scale/(stds[1]*case.L_scale)*v_y  # continuity

    # Mean Squared Error Loss
    loss_f = nn.MSELoss()

    # For postprocessing the loss values for each collocation point are required, no MSE loss
    
    total_loss = abs(loss_x) + abs(loss_y) + abs(loss_z) + abs(loss_c) if z else abs(loss_x) + abs(loss_y) + abs(loss_c)
    loss_x_f = loss_f(loss_x, torch.zeros_like(loss_x))
    loss_y_f = loss_f(loss_y, torch.zeros_like(loss_y))
    loss_c_f = loss_f(loss_c, torch.zeros_like(loss_c))


    loss_geo = loss_x_f + loss_y_f + loss_f(loss_z, torch.zeros_like(loss_z)) + loss_c_f if z else loss_x_f + loss_y_f + loss_c_f

    if flag_values & flag_vtk: 
        loss_terms = torch.tensor(
            [loss_f(u*u_x, torch.zeros_like(u)), loss_f(v*u_y, torch.zeros_like(u)),
            loss_f(u*v_x, torch.zeros_like(u)), loss_f(v*v_y, torch.zeros_like(u)),
            loss_f(u_xx, torch.zeros_like(u)), loss_f(u_yy, torch.zeros_like(u)),
            loss_f(v_xx, torch.zeros_like(u)), loss_f(v_yy, torch.zeros_like(u)),
            loss_f(p_x, torch.zeros_like(u)), loss_f(p_y, torch.zeros_like(u)), 
            loss_f(u_x, torch.zeros_like(u)), loss_f(u_y, torch.zeros_like(u))])
        return loss_geo, grads, loss_terms, total_loss, loss_c, loss_x, loss_y, loss_c_f, loss_x_f, loss_y_f 

    elif flag_values: 

        loss_terms = torch.tensor(
            [loss_f(u*u_x, torch.zeros_like(u)), loss_f(v*u_y, torch.zeros_like(u)),
            loss_f(u*v_x, torch.zeros_like(u)), loss_f(v*v_y, torch.zeros_like(u)),
            loss_f(u_xx, torch.zeros_like(u)), loss_f(u_yy, torch.zeros_like(u)),
            loss_f(v_xx, torch.zeros_like(u)), loss_f(v_yy, torch.zeros_like(u)),
            loss_f(p_x, torch.zeros_like(u)), loss_f(p_y, torch.zeros_like(u)), 
            loss_f(u_x, torch.zeros_like(u)), loss_f(u_y, torch.zeros_like(u))])
        
        return loss_geo, grads, loss_terms, total_loss, loss_x, loss_y, loss_c
    
    elif flag_vtk: 
        return total_loss, loss_c, loss_x, loss_y, loss_c_f, loss_x_f, loss_y_f 
        
    return loss_geo, grads 

def evo(total_loss, loss_x, loss_y, loss_c, batch_locations, lim=0.6):
    """
    """
    batchsize = len(batch_locations[0])

    threshold = torch.mean(total_loss)
    idx = (total_loss > threshold).nonzero(as_tuple=True)[0]
    idx = idx.to(dtype=torch.long, device='cuda')
    
    nr_retained = len(idx)
    frac_retained = nr_retained / batchsize
    if frac_retained > lim:
        idx = idx[:int(lim*batchsize)]

    batch_locations_x = batch_locations[0][idx]
    batch_locations_y = batch_locations[1][idx]
    batch_locations_retain = [batch_locations_x, batch_locations_y]
    
    total_loss_retain = total_loss[idx]
    loss_x_retain = loss_x[idx]
    loss_y_retain = loss_y[idx]
    loss_c_retain = loss_c[idx]

    return batch_locations_retain, total_loss_retain, loss_x_retain, loss_y_retain, loss_c_retain



def loss_geo_v3(case, predictions, batch_locations, means, stds, flag_post=False, flag_values=False):
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

    d = 0.3
    U = 0.5
    nu = 0.01 
    rho = 1.  # TODO what to choose for rho? 
    eta = nu*rho

    p0 = eta*U/d 
    #p0 = rho*(U**2)
    
    Re = d*U/nu

    # Calculate gradients of output parameters w.r.t. coordinates
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True, allow_unused=True)[0]
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


    # Nondimensionalizing the parameters
    u = u/U
    v = v/U
    # w = w/U
    
    x = x/d
    y = y/d
    # z = z/d
    p = p/p0

    u_x = d/U * u_x
    u_xx = (d**2)/U * u_xx
    u_y = d/U * u_y
    u_yy = (d**2)/U * u_yy
    v_x = d/U * v_x
    v_xx = (d**2)/U * v_xx
    v_y = d/U * v_y
    v_yy = (d**2)/U * v_yy

    grads_scaled = [[u_x], [u_xx], [u_y], [u_yy], [v_x], [v_xx], [v_y], [v_yy], [p_x], [p_y]]
    # grads = torch.tensor([[torch.min(u_x), torch.max(u_x),torch.min(u_xx), torch.max(u_xx)], [torch.min(u_y), torch.max(u_y),
    #                       torch.min(u_yy), torch.max(u_yy)], [torch.min(v_x), torch.max(v_x), torch.min(v_xx), torch.max(v_xx)],
    #                       [torch.min(v_y), torch.max(v_y), torch.min(v_yy), torch.max(v_yy)],
    #                       [torch.min(p_x), torch.max(p_x), torch.min(p_y), torch.max(p_y)]])
    
    # Loss functions 
    if z:
        loss_x = 1/stds[0]*(u * u_x) + 1/stds[1]*(v * u_y) + 1/stds[2]*(w * u_z) - 1/Re*(1/(stds[0]**2)*u_xx + 
        1/(stds[1]**2)*u_yy + 1/(stds[2]**2)*u_zz - 1/stds[0] * p_x)  # X-dir
        loss_y = 1/stds[0]*(u * v_x) + 1/stds[1]*(v * v_y) + 1/stds[2]*(w * v_z) - 1/Re*(1/(stds[0]**2)*v_xx + 
        1/(stds[1]**2)*v_yy + 1/(stds[2]**2)*v_zz - 1/stds[1] * p_y)  # Y-dir
        loss_z = 1/stds[0]*(u * w_x) + 1/stds[1]*(v * w_y) + 1/stds[2]*(w * w_z) - 1/Re*(1/(stds[0]**2)*w_xx + 
        1/(stds[1]**2)*w_yy + 1/(stds[2]**2)*w_zz - 1/stds[2] * p_z)  # Z-dir
        loss_c = case.U_scale/(stds[0]*case.L_scale)*u_x + case.U_scale/(stds[1]*case.L_scale)*v_y + case.U_scale/(stds[2]*case.L_scale)*w_z  # continuity
    else:
        loss_x = 1/stds[0]*(u * u_x) + 1/stds[1]*(v * u_y) - 1/Re*(1/(stds[0]**2)*u_xx + 1/(stds[1]**2)*u_yy - 1/stds[0]*p_x)  # X-dir
        loss_y = 1/stds[0]*(u * v_x) + 1/stds[1]*(v * v_y) - 1/Re*(1/(stds[0]**2)*v_xx + 1/(stds[1]**2)*v_yy - 1/stds[1]*p_y)  # Y-dir
        loss_c = case.U_scale/(stds[0]*d)*u_x + case.U_scale/(stds[1]*d)*v_y  # continuity

    # Scaling parameters back to original prediction 
    u = u*U
    v = v*U
    # w = w*U
    x = x*d
    y = y*d
    # z = z*d
    p = p*p0

    u_x = U/d * u_x
    u_xx = U/(d**2) * u_xx
    u_y = U/d * u_y
    u_yy = U/(d**2) * u_yy
    v_x = U/d * v_x
    v_xx = U/(d**2) * v_xx
    v_y = U/d * v_y
    v_yy = U/(d**2) * v_yy

    grads = [[u_x], [u_xx], [u_y], [u_yy], [v_x], [v_xx], [v_y], [v_yy], [p_x], [p_y]]
    
    # Mean Squared Error Loss
    loss_f = nn.MSELoss()

    # For postprocessing the loss values for each collocation point are required, no MSE loss
    
    total_loss = abs(loss_x) + abs(loss_y) + abs(loss_z) + abs(loss_c) if z else abs(loss_x) + abs(loss_y) + abs(loss_c)

    loss_geo = loss_f(loss_x, torch.zeros_like(loss_x)) + loss_f(loss_y, torch.zeros_like(loss_y)) + \
           loss_f(loss_z, torch.zeros_like(loss_z)) + loss_f(loss_c, torch.zeros_like(loss_c)) \
        if z else loss_f(loss_x, torch.zeros_like(loss_x)) + \
                             loss_f(loss_y, torch.zeros_like(loss_y)) + \
                             loss_f(loss_c, torch.zeros_like(loss_c))

    if flag_post:
        return total_loss, loss_c, loss_x, loss_y, loss_z if z else total_loss, loss_c, loss_x, loss_y 

    if flag_values:
        loss_terms = torch.tensor(
                [loss_f(u*u_x, torch.zeros_like(u)), loss_f(v*u_y, torch.zeros_like(u)),
                loss_f(u*v_x, torch.zeros_like(u)), loss_f(v*v_y, torch.zeros_like(u)),
                loss_f(u_xx, torch.zeros_like(u)), loss_f(u_yy, torch.zeros_like(u)),
                loss_f(v_xx, torch.zeros_like(u)), loss_f(v_yy, torch.zeros_like(u)),
                loss_f(p_x, torch.zeros_like(u)), loss_f(p_y, torch.zeros_like(u)), 
                loss_f(u_x, torch.zeros_like(u)), loss_f(u_y, torch.zeros_like(u))])
        
        return loss_geo, grads, loss_terms
    
    return loss_geo, grads 

def loss_geo_v4(case, predictions, batch_locations, means, stds, flag_post=False, flag_values=False):
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

    # grads = torch.tensor([[torch.min(u_x), torch.max(u_x),torch.min(u_xx), torch.max(u_xx)], [torch.min(u_y), torch.max(u_y),
    #                       torch.min(u_yy), torch.max(u_yy)], [torch.min(v_x), torch.max(v_x), torch.min(v_xx), torch.max(v_xx)],
    #                       [torch.min(v_y), torch.max(v_y), torch.min(v_yy), torch.max(v_yy)],
    #                       [torch.min(p_x), torch.max(p_x), torch.min(p_y), torch.max(p_y)]])
    grads = [[u_x, u_xx], [u_y, u_yy], [v_x, v_xx], [v_y, v_yy], [p_x, p_y]]
    # Loss functions 
    if z:
        loss_x = 1/stds[0]*(u * u_x) + 1/stds[1]*(v * u_y) + 1/stds[2]*(w * u_z) - 1/case.Reynolds*(1/(stds[0]**2)*u_xx + 
        1/(stds[1]**2)*u_yy + 1/(stds[2]**2)*u_zz - 1/stds[0] * p_x)  # X-dir
        loss_y = 1/stds[0]*(u * v_x) + 1/stds[1]*(v * v_y) + 1/stds[2]*(w * v_z) - 1/case.Reynolds*(1/(stds[0]**2)*v_xx + 
        1/(stds[1]**2)*v_yy + 1/(stds[2]**2)*v_zz - 1/stds[1] * p_y)  # Y-dir
        loss_z = 1/stds[0]*(u * w_x) + 1/stds[1]*(v * w_y) + 1/stds[2]*(w * w_z) - 1/case.Reynolds*(1/(stds[0]**2)*w_xx + 
        1/(stds[1]**2)*w_yy + 1/(stds[2]**2)*w_zz - 1/stds[2] * p_z)  # Z-dir
        loss_c = case.U_scale/(stds[0]*case.L_scale)*u_x + case.U_scale/(stds[1]*case.L_scale)*v_y + case.U_scale/(stds[2]*case.L_scale)*w_z  # continuity
    else:
        loss_x = 1/stds[0]*(u * u_x) + 1/stds[1]*(v * u_y) - 1/case.Reynolds*(1/(stds[0]**2)*u_xx + 1/(stds[1]**2)*u_yy - 1/stds[0]*p_x)  # X-dir
        loss_y = 1/stds[0]*(u * v_x) + 1/stds[1]*(v * v_y) - 1/case.Reynolds*(1/(stds[0]**2)*v_xx + 1/(stds[1]**2)*v_yy - 1/stds[1]*p_y)  # Y-dir
        loss_c = case.U_scale/(stds[0]*case.L_scale)*u_x + case.U_scale/(stds[1]*case.L_scale)*v_y  # continuity

    # Mean Squared Error Loss
    loss_MSE = nn.MSELoss()
    loss_Hubers = nn.HuberLoss()
    loss_MAE = nn.L1Loss()


    # For postprocessing the loss values for each collocation point are required, no MSE loss
    
    total_loss = abs(loss_x) + abs(loss_y) + abs(loss_z) + abs(loss_c) if z else abs(loss_x) + abs(loss_y) + abs(loss_c)

    loss_geo_MSE = loss_MSE(loss_x, torch.zeros_like(loss_x)) + loss_MSE(loss_y, torch.zeros_like(loss_y)) + \
           loss_MSE(loss_z, torch.zeros_like(loss_z)) + loss_MSE(loss_c, torch.zeros_like(loss_c)) \
        if z else loss_MSE(loss_x, torch.zeros_like(loss_x)) + \
                             loss_MSE(loss_y, torch.zeros_like(loss_y)) + \
                             loss_MSE(loss_c, torch.zeros_like(loss_c))

    loss_geo_Hubers = loss_Hubers(loss_x, torch.zeros_like(loss_x)) + loss_Hubers(loss_y, torch.zeros_like(loss_y)) + \
           loss_Hubers(loss_z, torch.zeros_like(loss_z)) + loss_Hubers(loss_c, torch.zeros_like(loss_c)) \
        if z else loss_Hubers(loss_x, torch.zeros_like(loss_x)) + \
                             loss_Hubers(loss_y, torch.zeros_like(loss_y)) + \
                             loss_Hubers(loss_c, torch.zeros_like(loss_c))
    
    loss_geo_MAE = loss_MAE(loss_x, torch.zeros_like(loss_x)) + loss_MAE(loss_y, torch.zeros_like(loss_y)) + \
           loss_MAE(loss_z, torch.zeros_like(loss_z)) + loss_MAE(loss_c, torch.zeros_like(loss_c)) \
        if z else loss_MAE(loss_x, torch.zeros_like(loss_x)) + \
                             loss_MAE(loss_y, torch.zeros_like(loss_y)) + \
                             loss_MAE(loss_c, torch.zeros_like(loss_c))

    if flag_post:
        return total_loss, loss_c, loss_x, loss_y, loss_z if z else total_loss, loss_c, loss_x, loss_y 

    if flag_values:
        loss_terms_MSE = torch.tensor(
                [loss_MSE(u*u_x, torch.zeros_like(u)), loss_MSE(v*u_y, torch.zeros_like(u)),
                loss_MSE(u*v_x, torch.zeros_like(u)), loss_MSE(v*v_y, torch.zeros_like(u)),
                loss_MSE(u_xx, torch.zeros_like(u)), loss_MSE(u_yy, torch.zeros_like(u)),
                loss_MSE(v_xx, torch.zeros_like(u)), loss_MSE(v_yy, torch.zeros_like(u)),
                loss_MSE(p_x, torch.zeros_like(u)), loss_MSE(p_y, torch.zeros_like(u)), 
                loss_MSE(u_x, torch.zeros_like(u)), loss_MSE(u_y, torch.zeros_like(u))])
        
        loss_terms_Hubers = torch.tensor(
                [loss_Hubers(u*u_x, torch.zeros_like(u)), loss_Hubers(v*u_y, torch.zeros_like(u)),
                loss_Hubers(u*v_x, torch.zeros_like(u)), loss_Hubers(v*v_y, torch.zeros_like(u)),
                loss_Hubers(u_xx, torch.zeros_like(u)), loss_Hubers(u_yy, torch.zeros_like(u)),
                loss_Hubers(v_xx, torch.zeros_like(u)), loss_Hubers(v_yy, torch.zeros_like(u)),
                loss_Hubers(p_x, torch.zeros_like(u)), loss_Hubers(p_y, torch.zeros_like(u)), 
                loss_Hubers(u_x, torch.zeros_like(u)), loss_Hubers(u_y, torch.zeros_like(u))])

        loss_terms_MAE = torch.tensor(
                [loss_MAE(u*u_x, torch.zeros_like(u)), loss_MAE(v*u_y, torch.zeros_like(u)),
                loss_MAE(u*v_x, torch.zeros_like(u)), loss_MAE(v*v_y, torch.zeros_like(u)),
                loss_MAE(u_xx, torch.zeros_like(u)), loss_MAE(u_yy, torch.zeros_like(u)),
                loss_MAE(v_xx, torch.zeros_like(u)), loss_MAE(v_yy, torch.zeros_like(u)),
                loss_MAE(p_x, torch.zeros_like(u)), loss_MAE(p_y, torch.zeros_like(u)), 
                loss_MAE(u_x, torch.zeros_like(u)), loss_MAE(u_y, torch.zeros_like(u))])

        return loss_geo_MSE, loss_geo_Hubers, loss_geo_MAE, grads, loss_terms_MSE, loss_terms_Hubers, loss_terms_MAE
    
    return loss_geo_MSE, loss_geo_Hubers, loss_geo_MAE, grads 



def loss_wss(case, predictions_bnc, networks, x, y, *z, flag_show=True):
    """
    """
    # x, y, *z = boundary_locations
    u, v, w, p = predictions_bnc
    y.requires_grad_()
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), only_inputs=True)[0]
    
    max_y = torch.max(y).item()
    y_wall = y[y==max_y]  # Contains only the max_y value   
    
    idx_y = (y == max_y).nonzero(as_tuple=True)[0]
    x_wall = x[idx_y]
    
    threshold = 0.075
    idx_x = (x_wall > threshold).nonzero(as_tuple=True)[0]
    
    yb = y_wall[idx_x]
    xb = x_wall[idx_x]
    print(torch.min(xb))
    u_wall = u[idx_y]
    ub = u_wall[idx_x]

    u_y_wall = u_y[idx_y]
    u_y_b = u_y_wall[idx_x]

    # Calculate WSS
    mu = 4e-3  # https://www.engineeringtoolbox.com/absolute-viscosity-liquids-d_1259.html

    wss = torch.mul(u_y_b, -mu)
    wss_max = torch.max(wss).item()
    wss_norm = torch.div(wss, wss_max)



    ################ 
    # Old calculation
    device = 'cuda'
    net2_u, net2_v, _, _ = networks
    X_start = torch.min(x.data).item()
    X_end = torch.max(x.data).item()
    Y_start = torch.min(y.data).item()
    Y_end = torch.max(y.data).item()

    nr_ypoints = x.data[x.data == 0].size()[0]
    # res = ((Y_end-Y_start)/(nr_ypoints+1)) #0.0023
    res = Y_end/100

    x_begin = 0.075
    x_begin = float("{0:.3f}".format(x_begin))

    points_from_wall = 1
    loc_wall = Y_end - points_from_wall*res
    perc_wall = (1 - loc_wall/Y_end)*100 # percentage from the wall compared to whole Y_domain
    perc_wall = float("{0:.2f}".format(perc_wall))
    loc_wall = float("{0:.4f}".format(loc_wall))
    # y_data = y.data[y.data == loc_wall].to(device) #werkt ook! houdt alleen waardes met 0.3 over, indexen zijn wel verloren
    # y_data = y_data.view(len(y_data), -1)

    y_size = y.data[y.data == Y_end].size()[0]

    y_data = torch.tensor(loc_wall) # wat is de lengte van y_data
    y_data = y_data.repeat(y_size).to(device)
    y_data = y_data.view(len(y_data), -1)
    # y_size = y_data.size()[0]

    # y_size = y_size[0]
    # x_data = torch.linspace(X_start,X_end,y_size).to(device) # Moet na stenose kijken!
    x_data = torch.linspace(x_begin, X_end, y_size).to(device)
    x_data = x_data.view(len(x_data), -1)

    x_data.requires_grad = True # start bijhouden wijzigingen aan weights
    y_data.requires_grad = True

    net_in = torch.cat((x_data,y_data), 1)
    # net_in = torch.cat((x, y), 1) # concatenates the tensors, 1 stating the dimension

    # --> x en y gaan er dan niet apart in? --> Jawel, (256, 2)
    u = net2_u(net_in)
    u = u.view(len(u), -1) # rearrange the tensor, geef het len(u) rijen, grootte column (-1) mag het zelf uitrekenen
    v = net2_v(net_in)
    v = v.view(len(v), -1)  # (256, 1 ) afhankelijk dus van batch_size grootte
    # P = net2_p(net_in)
    # P = P.view(len(P), -1)


    u_y = torch.autograd.grad(u, y_data, grad_outputs=torch.ones_like(y_data), only_inputs=True)[0]

    # print('calculation u and v complete')
    #code doet moeilijk in debugger somehow, code lijkt wel te werken, alleen letten op step into

    #Calculate WSS
    mu = 4e-3 # https://www.engineeringtoolbox.com/absolute-viscosity-liquids-d_1259.html
    #rho = 1   # staat in 2D stenose, maar eigenlijk 1000 kg/m3
    # 100g / L (dm3)
    # 10 g / dL (cm3)
    # 1 g / mL (mm3)
    # --> mu moet waarschijnlijk ook aangepast worden, berekening is echter WSS/WSSmax

    WSS = torch.mul(u_y, -mu)
    WSS_max = torch.max(WSS).item()
    WSS_norm = torch.div(WSS, WSS_max)

    plt.figure()
    plt.plot(x_data.cpu().data.numpy(), WSS_norm.cpu().data.numpy())
    plt.plot(xb.cpu().data.numpy(), wss_norm.cpu().data.numpy())
    plt.legend(['old', 'new'])
    plt.show()


    #######################
    # Toepassing van de stress tensor 
    nPt = 200
    
    radius = 0.5
    x_circle = torch.zeros(nPt)
    y_circle = torch.zeros(nPt)
    x_inner_c = torch.zeros(nPt)
    y_inner_c = torch.zeros(nPt)
    radians = torch.linspace(-0.60, 3.70, nPt)
    perc = 0.95


    M_np = np.array([[1.8], [0.7], [0]])
    radius_np = 0.5
    theta = np.linspace(-0.60, 3.70, nPt)
    circle = np.array([radius_np*np.cos(theta) + M_np[0], radius_np*np.sin(theta)+M_np[1]])

    e_n = np.zeros((nPt,2))
    len_e = np.zeros(nPt)


    for i in range(nPt):
        e_n_i = np.array([circle[0,i] - M_np[0], circle[1,i] - M_np[1]])
        mag = np.sqrt(e_n_i[0]**2 + e_n_i[1]**2)
        e_n[i, 0] = (circle[0, i] - M_np[0]) / mag * -1
        e_n[i,1] = (circle[1,i] - M_np[1]) / mag * -1
        len_e[i] = np.sqrt(e_n[i,0] **2 + e_n[i,1]**2)


    x_circ = circle[0,:]
    x_circ = x_circ.reshape(-1, 1)
    y_circ = circle[1,:]
    y_circ = y_circ.reshape(-1, 1)
    e_nx = e_n[:,0]
    e_nx = e_nx.reshape(-1, 1)
    e_ny = e_n[:,1]
    e_ny = e_ny.reshape(-1, 1)

    plt.figure()
    fig, ax = plt.subplots(figsize=(9,9))
    skip=(slice(None,None,5),slice(None,None,5)) #plot every 5 pts
    #ax.quiver(x.detach().numpy(), y.detach().numpy(), output_u , output_v,scale=5)
    ax.quiver(x_circ[skip], y_circ[skip], e_nx[skip], e_ny[skip],scale=50)#a smaller scale parameter makes the arrow longer.
    # ax[1].quiver(x_inner_e[skip], y_inner_e[skip], e_nX2[skip], e_nY2[skip],scale=50)
    # ax.quiver(x.detach().numpy()[skip], y.detach().numpy()[skip], U[skip], V[skip],scale=50)
    plt.title('NN results, Vel vector')
    ax.axis('equal')
    # plt.show()


    x_circle = torch.tensor(x_circ)
    y_circle = torch.tensor(y_circ)
    x_circle = x_circle.type(torch.cuda.FloatTensor)
    y_circle = y_circle.type(torch.cuda.FloatTensor)

    y_circle = y_circle.view(nPt, -1)  # vs len(y_data) in 2D stenose
    x_circle = x_circle.view(nPt, -1)
    # y_inner_c = y_inner_c.view(nPt, -1)
    # x_inner_c = x_inner_c.view(nPt, -1)

    # x_inner_e = x_inner_c / X_scale
    # y_inner_e = y_inner_c / Y_scale

    x_inner_e = x_circle / X_scale
    y_inner_e = y_circle / Y_scale

    x_inner_e = x_inner_e.to(device)
    y_inner_e = y_inner_e.to(device)
    x_inner_e.requires_grad = True
    y_inner_e.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    net_in = torch.cat((x_inner_e, y_inner_e), 1) #inner ellips coordinates are put into NN
    # net_in = torch.cat((x, y), 1)

    u = net2_u(net_in)
    u = u.view(len(u), -1) # rearrange the tensor, geef het len(u) rijen, grootte column (-1) mag het zelf uitrekenen
    v = net2_v(net_in)
    v = v.view(len(v), -1)  # (256, 1 ) afhankelijk dus van batch_size grootte
    P = net2_p(net_in)
    P = P.view(len(P), -1)

    u_y = torch.autograd.grad(u, y_inner_e, grad_outputs=torch.ones_like(y_inner_e), create_graph=True, only_inputs=True)[0]
    u_x = torch.autograd.grad(u, x_inner_e, grad_outputs=torch.ones_like(x_inner_e), create_graph=True, only_inputs=True)[0]

    v_y = torch.autograd.grad(v, y_inner_e, grad_outputs=torch.ones_like(y_inner_e), create_graph=True, only_inputs=True)[0]
    v_x = torch.autograd.grad(v, x_inner_e, grad_outputs=torch.ones_like(x_inner_e), create_graph=True, only_inputs=True)[0]

    # u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    # u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    #
    # v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    # v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]

    mu = 4e-3  # https://www.engineeringtoolbox.com/absolute-viscosity-liquids-d_1259.html

    stress = torch.zeros(nPt, 2, 2)  # 200 points, 3x3 matrix for each point
    stress_n = torch.zeros(nPt, 2)
    stress_nn = torch.zeros(nPt, 1).to(device)
    stress_nn_vec = torch.zeros(nPt, 2)
    WSS_vec = torch.zeros(nPt, 2)
    WSS_mag = torch.zeros(nPt, 1)

    e_nx = torch.tensor(e_nx).to(device)
    e_ny = torch.tensor(e_ny).to(device)
    # e_nx = e_nx.cpu()
    # e_ny = e_ny.cpu()

    for i in range(nPt):

        # building stress tensor
        p = P[i]  # local pressure
        ux_i = u_x[i]
        uy_i = u_y[i]
        vx_i = v_x[i]
        vy_i = v_y[i]
        stress[i, 0, 0] = 2*ux_i - p
        stress[i, 0, 1] = uy_i + vx_i

        stress[i, 1, 0] = uy_i + vx_i
        stress[i, 1, 1] = 2*vy_i - p

        #niet vergeten stress aan het eind met mu te vermenigvuldigen

        stress_n[i, 0] = (2 * ux_i - p) * e_nx[i] + ((uy_i + vx_i) * e_ny[i])   # rij komt overeen met index punt
        stress_n[i, 1] = (2 * uy_i - p) * e_ny[i] + ((uy_i + vx_i) * e_nx[i])

        stress_nn[i] = e_nx[i] * stress_n[i, 0] + e_ny[i] * stress_n[i, 1]   #  make it a vector
        stress_nn_vec[i, 0] = stress_nn[i] * e_nx[i]
        stress_nn_vec[i, 1] = stress_nn[i] * e_ny[i]

        WSS_vec[i, :] = torch.mul(mu, stress_n[i, :]) - torch.mul(mu, stress_nn_vec[i, :])  # WSS vector
        WSS_mag[i] = np.sqrt(torch.inner(WSS_vec[i, :], WSS_vec[i, :]))

    WSS_max = torch.max(WSS_mag).item()
    WSS_norm = torch.div(WSS_mag, WSS_max)

    X_max_c = torch.max(x_circle).item()
    X_min_c = torch.min(x_circle).item()
    # radius_c = (X_max_c - X_min_c) / 2  # radius of the circle

    frac_c = (radians[-1] - radians[0])/(2*np.pi)  # fractie van cirkel meegenomen voor WSS
    S = torch.linspace(0, radius*frac_c*2*np.pi, nPt)  # 'parametrization', circumferention


    plt.figure()
    plt.plot(S, WSS_norm)
    plt.xlabel('S')
    plt.ylabel('WSS/WSSmax')
    plt.title('WSS/WSSmax versus circumferention S')
    plt.show()



    plt.figure()
    plt.scatter(xb.cpu().detach().numpy(), wss_norm.cpu().detach().numpy())
    plt.scatter(x_data.cpu().detach().numpy(), WSS_norm.cpu().detach().numpy())
    plt.scatter(xb.cpu().detach().numpy(), yb.cpu().detach().numpy())
    # plt.scatter(x_data.cpu().detach().numpy(), y_data.cpu().detach().numpy())
    plt.scatter(x_wall.cpu().detach().numpy(), y_wall.cpu().detach().numpy())
    plt.ylabel('WSS/WSS_max')
    plt.xlabel('x_pos (domain: [0 1])')
    plt.legend(['own', 'old'])
    # plt.savefig(path + "/Plots/2DstenoseWSS")
    plt.show()
    
    if flag_show == True:
        plt.show()

    return