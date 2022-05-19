import torch
import torch.nn as nn

def criterion(geometry_locations, networks, case, predictions, criterion_plot = False):

    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True

    # loss_eqn = torch.tensor([0], requires_grad=True)

    net_in = torch.cat((x, y, z), 1) if input_n == 3 else torch.cat((x, y), 1)
    u = net2_u(net_in)
    u = u.view(len(u), -1)
    v = net2_v(net_in)
    v = v.view(len(v), -1)
    P = net2_p(net_in)
    P = P.view(len(P), -1)

    if input_n == 3:
        w = net2_w(net_in)
        w = w.view(len(w), -1)
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

        P_z = torch.autograd.grad(P, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]

    P_x = torch.autograd.grad(P, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    P_y = torch.autograd.grad(P, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]

    grads = torch.tensor([[torch.min(u_x), torch.max(u_x),torch.min(u_xx), torch.max(u_xx)], [torch.min(u_y), torch.max(u_y),
                          torch.min(u_yy), torch.max(u_yy)], [torch.min(v_x), torch.max(v_x), torch.min(v_xx), torch.max(v_xx)],
                          [torch.min(v_y), torch.max(v_y), torch.min(v_yy), torch.max(v_yy)],
                          [torch.min(P_x), torch.max(P_x), torch.min(P_y), torch.max(P_y)]])
    # XX_scale = U_scale * (X_scale ** 2)  # Dimensionless NS equation
    # YY_scale = U_scale * (Y_scale ** 2)
    # UU_scale = U_scale ** 2

    XX_scale = (X_scale ** 2)  # Dimensionless NS equation
    YY_scale = (Y_scale ** 2)
    ZZ_scale = (Z_scale ** 2)
    UU_scale = 1.
    U_scale = 1.

    if input_n == 3:
        ZZ_scale = U_scale * (Z_scale ** 2)
        loss_1 = u * u_x / X_scale + v * u_y / Y_scale + w * u_z / Z_scale - Diff * (u_xx / XX_scale + u_yy / YY_scale
                                 + u_zz / ZZ_scale) + 1 / rho * (P_x / (X_scale * UU_scale))  # X-dir
        loss_2 = u * v_x / X_scale + v * v_y / Y_scale + w * v_z / Z_scale - Diff * (v_xx / XX_scale + v_yy / YY_scale
                                 + v_zz / ZZ_scale) + 1 / rho * (P_y / (Y_scale * UU_scale))  # Y-dir
        loss_3 = u * w_x / X_scale + v * w_y / Y_scale + w * w_z / Z_scale - Diff * (w_xx / XX_scale + w_yy / YY_scale
                                 + w_zz / ZZ_scale) + 1 / rho * (P_z / (Y_scale * UU_scale))  # Z-dir
        loss_4 = (u_x / X_scale + v_y / Y_scale + w_z / Z_scale)  # continuity
    else:
        loss_1 = u * u_x / X_scale + v * u_y / Y_scale - \
                 Diff * (u_xx / XX_scale + u_yy / YY_scale) + 1 / rho * (P_x / (X_scale * UU_scale))  # X-dir
        loss_2 = u * v_x / X_scale + v * v_y / Y_scale - \
                 Diff * (v_xx / XX_scale + v_yy / YY_scale) + 1 / rho * (P_y / (Y_scale * UU_scale))  # Y-dir
        loss_3 = (u_x / X_scale + v_y / Y_scale)  # continuity

    # MSE LOSS
    loss_f = nn.MSELoss()

    if criterion_plot:
        final_loss = abs(loss_1) + abs(loss_2) + abs(loss_3) + abs(loss_4) if input_n == 3 else abs(loss_1)
        + abs(loss_2) + abs(loss_3)# for plotting
        return final_loss, abs(loss_2), abs(
            loss_3)  # Criterion plot, checken of die overeenkomt met de al gedefinieerde fu

    loss = loss_f(loss_1, torch.zeros_like(loss_1)) + loss_f(loss_2, torch.zeros_like(loss_2)) + \
           loss_f(loss_3, torch.zeros_like(loss_3)) + loss_f(loss_4, torch.zeros_like(loss_4)) \
        if input_n == 3 else loss_f(loss_1, torch.zeros_like(loss_1)) + \
                             loss_f(loss_2, torch.zeros_like(loss_2)) + \
                             loss_f(loss_3, torch.zeros_like(loss_3))

    return loss, grads