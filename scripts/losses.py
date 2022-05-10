import torch
import torch.nn as nn

#TODO manier voor reduceren if statements en variabel afhankelijk maken oid
# TODO kunnen diff etc vervangen worden door case.?
def criterion(x, y, z, net2_u, net2_v, net2_w, net2_p, X_scale, Y_scale, Z_scale, U_scale, Diff, rho, input_n,
              criterion_plot = False):

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


def Loss_BC(xb, yb, zb, net2_u, net2_v, net2_w, input_n):

    if input_n == 3:
        net_in1 = torch.cat((xb, yb, zb), 1)
    else:
        net_in1 = torch.cat((xb, yb), 1)

    out1_u = net2_u(net_in1)
    out1_u = out1_u.view(len(out1_u), -1)
    out1_v = net2_v(net_in1)
    out1_v = out1_v.view(len(out1_v), -1)

    loss_f = nn.MSELoss()

    if input_n == 3:
        out1_w = net2_w(net_in1)
        out1_w = out1_w.view(len(out1_w), 1)
        return (
            loss_f(out1_u, torch.zeros_like(out1_u))
            + loss_f(out1_v, torch.zeros_like(out1_v))
            + loss_f(out1_w, torch.zeros_like(out1_w))
        )

    else:
        return loss_f(out1_u, torch.zeros_like(out1_u)) + loss_f(
            out1_v, torch.zeros_like(out1_v)
        )

def Loss_data(xd, yd, zd, ud, vd, wd, net2_u, net2_v, net2_w, input_n):

    if input_n == 3:
        net_in1 = torch.cat((xd, yd, zd), 1)
    else:
        net_in1 = torch.cat((xd, yd), 1)

    out1_u = net2_u(net_in1)
    out1_u = out1_u.view(len(out1_u), -1)
    out1_v = net2_v(net_in1)
    out1_v = out1_v.view(len(out1_v), -1)

    loss_f = nn.MSELoss()

    if input_n == 3:
        out1_w = net2_w(net_in1)
        out1_w = out1_w.view(len(out1_w), -1)
        loss_d = loss_f(out1_u, ud) + loss_f(out1_v, vd) + loss_f(out1_w, wd)

    loss_d = loss_f(out1_u, ud) + loss_f(out1_v, vd)

    return loss_d

def criterion_plot(x, y, net2_u, net2_v, net2_p, X_scale, Y_scale, U_scale, Diff, rho):

    x.requires_grad = True
    y.requires_grad = True

    # net_in = torch.cat((x),1)
    net_in = torch.cat((x, y), 1)
    u = net2_u(net_in)
    u = u.view(len(u), -1)
    v = net2_v(net_in)
    v = v.view(len(v), -1)
    P = net2_p(net_in)
    P = P.view(len(P), -1)

    # u = u * t + V_IC #Enforce I.C???
    # v = v * t + V_IC #Enforce I.C???

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

    # u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]
    # v_t = torch.autograd.grad(v,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]

    XX_scale = U_scale * (X_scale ** 2)
    YY_scale = U_scale * (Y_scale ** 2)
    UU_scale = U_scale ** 2

    loss_2 = u * u_x / X_scale + v * u_y / Y_scale - Diff * (u_xx / XX_scale + u_yy / YY_scale) + 1 / rho * (
                P_x / (X_scale * UU_scale))  # X-dir
    loss_1 = u * v_x / X_scale + v * v_y / Y_scale - Diff * (v_xx / XX_scale + v_yy / YY_scale) + 1 / rho * (
                P_y / (Y_scale * UU_scale))  # Y-dir
    loss_3 = (u_x / X_scale + v_y / Y_scale)  # continuity

    # MSE LOSS
    loss_f = nn.MSELoss()

    # Note our target is zero. It is residual so we use zeros_like
    # loss = loss_f(loss_1,torch.zeros_like(loss_1))+ loss_f(loss_2,torch.zeros_like(loss_2))+loss_f(loss_3,torch.zeros_like(loss_3))

    final_loss = abs(loss_1) + abs(loss_2) + abs(loss_3)  # for plotting
    return final_loss, abs(loss_2), abs(loss_3) # Criterion plot, checken of die overeenkomt met de al gedefinieerde functie

def criterion2(x, y, z, net2_u, net2_v, net2_w, net2_p, X_scale, Y_scale, Z_scale, U_scale, Diff, rho, input_n,
              criterion_plot = False):
    "criterion function, but all the losses are split in terms for better monitoring"

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

    convU_ux = u * u_x
    convV_uy = v * u_y
    convU_vx = u * v_x
    convV_vy = v * v_y

    diffXu = Diff * u_xx
    diffYu = Diff * u_yy
    diffXv = Diff * v_xx
    diffYv = Diff * v_yy
    forceX = 1/rho * P_x
    forceY = 1/rho * P_y

    if input_n == 3:
        convW_uz = w * u_z
        convW_vz = w * v_z
        convU_wx = u * w_x
        convV_wy = v * w_y
        convW_wz = w * w_z
        diffZu = Diff * u_zz
        diffZv = Diff * v_zz
        diffXw = Diff * w_xx
        diffYw = Diff * w_yy
        diffZw = Diff * w_zz
        forceZ = 1/rho * P_z

        ZZ_scale = U_scale * (Z_scale ** 2)

        loss_1 = convU_ux / X_scale + convV_uy / Y_scale + convW_uz / Z_scale - (diffXu / XX_scale + diffYu / YY_scale + diffZu / ZZ_scale) \
                + forceX / (X_scale * UU_scale)

        loss_2 = convU_vx / X_scale + convV_vy / Y_scale + convW_vz / Z_scale - (diffXv / XX_scale + diffYv / YY_scale + diffZv / ZZ_scale) \
                 + forceY / (Y_scale * UU_scale)  # Y-dir

        loss_3 = convU_wx / X_scale + convV_wy / Y_scale + convW_wz / Z_scale - (diffXw / XX_scale + diffYw / YY_scale + diffZw / ZZ_scale) \
                 + forceZ / (Y_scale * UU_scale)  # Z-dir

        loss_4 = (u_x / X_scale + v_y / Y_scale + w_z / Z_scale)  # continuity
    else:
        loss_1 = convU_ux / X_scale + convV_uy / Y_scale - (diffXu / XX_scale + diffYu / YY_scale ) \
                 + forceX / (X_scale * UU_scale)

        loss_2 = convU_vx / X_scale + convV_vy / Y_scale - (diffXv / XX_scale + diffYv / YY_scale ) \
                 + forceY / (Y_scale * UU_scale)  # Y-dir
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

    if input_n == 3:
        loss_terms = torch.tensor(
            [loss_f(convU_ux, torch.zeros_like(convU_vx)), loss_f(convV_uy, torch.zeros_like(convU_vx)),
             loss_f(convW_uz, torch.zeros_like(convU_vx)), loss_f(convU_vx, torch.zeros_like(convU_vx)),
             loss_f(convV_vy, torch.zeros_like(convU_vx)), loss_f(convW_vz, torch.zeros_like(convU_vx)),
             loss_f(convU_wx, torch.zeros_like(convU_vx)), loss_f(convV_wy, torch.zeros_like(convU_vx)),
             loss_f(convW_wz, torch.zeros_like(convU_vx)), loss_f(diffXu - torch.zeros_like(convU_vx)),
             loss_f(diffYu, torch.zeros_like(convU_vx)), loss_f(diffZu, torch.zeros_like(convU_vx)),
             loss_f(diffXv, torch.zeros_like(convU_vx)), loss_f(diffYv, torch.zeros_like(convU_vx)),
             loss_f(diffZv, torch.zeros_like(convU_vx)), loss_f(diffXw, torch.zeros_like(convU_vx)),
             loss_f(diffYw, torch.zeros_like(convU_vx)), loss_f(diffZw, torch.zeros_like(convU_vx)),
             loss_f(forceX, torch.zeros_like(convU_vx)), loss_f(forceY, torch.zeros_like(convU_vx)),
             loss_f(forceZ, torch.zeros_like(convU_vx))])
    else:
        loss_terms = torch.tensor(
            [loss_f(convU_ux, torch.zeros_like(convU_vx)), loss_f(convV_uy, torch.zeros_like(convU_vx)),
             loss_f(convU_vx, torch.zeros_like(convU_vx)), loss_f(convV_vy, torch.zeros_like(convU_vx)),
             loss_f(diffXu, torch.zeros_like(convU_vx)), loss_f(diffYu, torch.zeros_like(convU_vx)),
             loss_f(diffXv, torch.zeros_like(convU_vx)), loss_f(diffYv, torch.zeros_like(convU_vx)),
             loss_f(forceX, torch.zeros_like(convU_vx)), loss_f(forceY, torch.zeros_like(convU_vx))])

    return loss, grads, loss_terms

def criterion3(x, y, z, net2_u, net2_v, net2_w, net2_p, X_scale, Y_scale, Z_scale, U_scale, Diff, rho, input_n,
              criterion_plot = False):
    "criterion function, but all the losses are split in terms for better monitoring"

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

    convU_ux = u * u_x
    convV_uy = v * u_y
    convU_vx = u * v_x
    convV_vy = v * v_y

    diffXu = Diff * u_xx
    diffYu = Diff * u_yy
    diffXv = Diff * v_xx
    diffYv = Diff * v_yy
    forceX = 1/rho * P_x
    forceY = 1/rho * P_y

    if input_n == 3:
        convW_uz = w * u_z
        convW_vz = w * v_z
        convU_wx = u * w_x
        convV_wy = v * w_y
        convW_wz = w * w_z
        diffZu = Diff * u_zz
        diffZv = Diff * v_zz
        diffXw = Diff * w_xx
        diffYw = Diff * w_yy
        diffZw = Diff * w_zz
        forceZ = 1/rho * P_z

        ZZ_scale = U_scale * (Z_scale ** 2)

        loss_1 = convU_ux + convV_uy + convW_uz - (diffXu + diffYu + diffZu) + forceX

        loss_2 = convU_vx + convV_vy + convW_vz - (diffXv + diffYv + diffZv) + forceY  # Y-dir

        loss_3 = convU_wx + convV_wy + convW_wz - (diffXw + diffYw + diffZw) + forceZ  # Z-dir

        loss_4 = u_x + v_y + w_z  # continuity
    else:
        loss_1 = convU_ux + convV_uy - (diffXu + diffYu  ) + forceX

        loss_2 = convU_vx + convV_vy - (diffXv + diffYv )  + forceY  # Y-dir
        loss_3 = u_x + v_y   # continuity

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

    if input_n == 3:
        loss_terms = torch.tensor(
            [loss_f(convU_ux, torch.zeros_like(convU_vx)), loss_f(convV_uy, torch.zeros_like(convU_vx)),
             loss_f(convW_uz, torch.zeros_like(convU_vx)), loss_f(convU_vx, torch.zeros_like(convU_vx)),
             loss_f(convV_vy, torch.zeros_like(convU_vx)), loss_f(convW_vz, torch.zeros_like(convU_vx)),
             loss_f(convU_wx, torch.zeros_like(convU_vx)), loss_f(convV_wy, torch.zeros_like(convU_vx)),
             loss_f(convW_wz, torch.zeros_like(convU_vx)), loss_f(diffXu - torch.zeros_like(convU_vx)),
             loss_f(diffYu, torch.zeros_like(convU_vx)), loss_f(diffZu, torch.zeros_like(convU_vx)),
             loss_f(diffXv, torch.zeros_like(convU_vx)), loss_f(diffYv, torch.zeros_like(convU_vx)),
             loss_f(diffZv, torch.zeros_like(convU_vx)), loss_f(diffXw, torch.zeros_like(convU_vx)),
             loss_f(diffYw, torch.zeros_like(convU_vx)), loss_f(diffZw, torch.zeros_like(convU_vx)),
             loss_f(forceX, torch.zeros_like(convU_vx)), loss_f(forceY, torch.zeros_like(convU_vx)),
             loss_f(forceZ, torch.zeros_like(convU_vx))])
    else:
        loss_terms = torch.tensor(
            [loss_f(convU_ux, torch.zeros_like(convU_vx)), loss_f(convV_uy, torch.zeros_like(convU_vx)),
             loss_f(convU_vx, torch.zeros_like(convU_vx)), loss_f(convV_vy, torch.zeros_like(convU_vx)),
             loss_f(diffXu, torch.zeros_like(convU_vx)), loss_f(diffYu, torch.zeros_like(convU_vx)),
             loss_f(diffXv, torch.zeros_like(convU_vx)), loss_f(diffYv, torch.zeros_like(convU_vx)),
             loss_f(forceX, torch.zeros_like(convU_vx)), loss_f(forceY, torch.zeros_like(convU_vx))])

    return loss, grads, loss_terms