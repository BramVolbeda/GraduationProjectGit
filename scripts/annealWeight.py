import torch

def GradLosses(case, device, ex_ID, epoch, epoch_pretrain, x, y, z, xb, yb, zb, xd, yd, zd, ud, vd, wd, net2_u,
               net2_v, net2_w, net2_p, alpha, anneal_weight, Flag_grad, Lambda_BC, Lambda_data, Flag_notrain=True):

    # Flag_grad = True
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)
    z = torch.Tensor(z).to(device)

    xb = torch.Tensor(xb).to(device)
    yb = torch.Tensor(yb).to(device)
    zb = torch.Tensor(zb).to(device)

    xd = torch.Tensor(xd).to(device)
    yd = torch.Tensor(yd).to(device)
    zd = torch.Tensor(zd).to(device)

    ud = torch.Tensor(ud).to(device)
    vd = torch.Tensor(vd).to(device)
    wd = torch.Tensor(wd).to(device)

    net2_u.zero_grad()
    net2_v.zero_grad()
    net2_p.zero_grad()
    net2_w.zero_grad()

    x = x[0::10]
    y = y[0::10]
    z = z[0::10]

    loss_eqn = criterion(x, y, z, net2_u, net2_v, net2_w, net2_p, case.X_scale, case.Y_scale,
                         case.Z_scale, case.U_scale, case.Diff, case.rho, case.input_n)
    loss_bc = Loss_BC(xb, yb, zb, net2_u, net2_v, net2_w, case.input_n)
    loss_data = Loss_data(xd, yd, zd, ud, vd, wd, net2_u, net2_v, net2_w, case.input_n)

    # anneal_weight = [[1.] for _ in range(nr_losses - 1)]
    nr_layers = 10  # includes input and output layers
    max_grad_r_list = []
    mean_grad_bc_list = []
    mean_grad_data_list = []
    loss_eqn.backward(retain_graph=True)

    layer = 0
    for name, param in net2_u.named_parameters(prefix=''):
        if "weight" in name and layer < nr_layers - 1:
            layer += 1
            max_grad_r_list.append(torch.max(abs(param.grad)))


    loss_bc.backward(retain_graph=True)
    layer = 0
    for name, param in net2_u.named_parameters(prefix=''):
        if "weight" in name and layer < nr_layers - 1:
            layer += 1
            mean_grad_bc_list.append(torch.mean(abs(param.grad)))

    loss_data.backward(retain_graph=True)

    layer = 0
    for name, param in net2_u.named_parameters(prefix=''):
        if "weight" in name and layer < nr_layers - 1:
            layer += 1
            mean_grad_data_list.append(torch.mean(abs(param.grad)))


    print('execute annealed learning weight algorithm')
    max_grad_r = torch.max(torch.tensor(max_grad_r_list))
    mean_grad_bc = torch.mean(torch.tensor(mean_grad_bc_list))
    mean_grad_data = torch.mean(torch.tensor(mean_grad_data_list))

    Lambda_BC_adaptive = max_grad_r / mean_grad_bc
    Lambda_data_adaptive = max_grad_r / mean_grad_data

    # Moving average

    Lambda_BC = Lambda_BC * (1 - alpha) + Lambda_BC_adaptive * alpha
    Lambda_data = Lambda_data * (1 - alpha) + Lambda_data_adaptive * alpha

    anneal_weight[0].append(Lambda_BC)  # boundary loss annealed learning rate
    anneal_weight[1].append(Lambda_data)  # data loss annealed learning rate

    return anneal_weight, Lambda_BC, Lambda_data