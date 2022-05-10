import torch
from matplotlib import pyplot as plt
import math
import numpy as np
import pandas as pd
import vtk
import vtkmodules.util.numpy_support as VN
from scripts.fileReaders import fileReader

def WSS_aneurysm(nr_WSSpoints, X_scale, Y_scale, Z_scale, net2_u, net2_v, net2_w, net2_p, device, show=True):
    # heb nu het 3D aneurysma gekopieerd
    # TODO: moet werken voor zowel 2D als 3D aneurysma

    M = torch.tensor([1.8, 0.7, 0])
    radius = 0.4

    x_circle = torch.zeros(nr_WSSpoints)
    y_circle = torch.zeros(nr_WSSpoints)
    z_circle = torch.zeros(nr_WSSpoints)
    x_inner_c = torch.zeros(nr_WSSpoints)
    y_inner_c = torch.zeros(nr_WSSpoints)
    z_inner_c = torch.zeros(nr_WSSpoints)
    # radians = torch.linspace(0, 4.7, nr_WSSpoints) # [0 270] graden
    radians = torch.linspace(-0.45, 3.7, nr_WSSpoints)

    perc = 0.99

    for i in range(nr_WSSpoints):
        radian = radians[i]
        x_circle[i] = radius * math.cos(radian) + M[0]
        y_circle[i] = radius * math.sin(radian) + M[1]
        x_inner_c[i] = perc * radius * math.cos(radian) + M[0]
        y_inner_c[i] = perc * radius * math.sin(radian) + M[1]

    size = x_circle.size()
    y_circle = y_circle.view(nr_WSSpoints, -1)  # vs len(y_data) in 2D stenose
    x_circle = x_circle.view(nr_WSSpoints, -1)
    z_circle = z_circle.view(nr_WSSpoints, -1)
    y_inner_c = y_inner_c.view(nr_WSSpoints, -1)
    x_inner_c = x_inner_c.view(nr_WSSpoints, -1)
    z_inner_c = z_inner_c.view(nr_WSSpoints, -1)

    x_ellips = x_circle / X_scale
    y_ellips = y_circle / Y_scale
    z_ellips = z_circle / Z_scale

    x_inner_e = x_inner_c / X_scale
    y_inner_e = y_inner_c / Y_scale
    z_inner_e = z_inner_c / Z_scale

    xmin_ellips = torch.min(x_ellips).item()  # nog een keer dubbel checken
    ymax_ellips = torch.max(y_ellips).item()
    xmin_circle = torch.min(x_circle).item()
    ymax_circle = torch.max(y_circle).item()

    x_inner_e = x_inner_e.to(device)
    y_inner_e = y_inner_e.to(device)
    z_inner_e = z_inner_e.to(device)

    x_inner_e.requires_grad = True  # start bijhouden wijzigingen aan weights
    y_inner_e.requires_grad = True
    z_inner_e.requires_grad = True

    net_in = torch.cat((x_inner_e, y_inner_e, z_inner_e), 1)  # inner ellips coordinates are put into NN

    u = net2_u(net_in)
    u = u.view(len(u), -1)  # rearrange the tensor, geef het len(u) rijen, grootte column (-1) mag het zelf uitrekenen
    v = net2_v(net_in)
    v = v.view(len(v), -1)  # (256, 1 ) afhankelijk dus van batch_size grootte
    w = net2_w(net_in)
    w = w.view(len(w), -1)
    P = net2_p(net_in)
    P = P.view(len(P), -1)

    # vel = torch.tensor(u, v) # eventueel nog fixen

    u_z = \
    torch.autograd.grad(u, z_inner_e, grad_outputs=torch.ones_like(z_inner_e), create_graph=True, only_inputs=True)[0]
    u_y = \
    torch.autograd.grad(u, y_inner_e, grad_outputs=torch.ones_like(y_inner_e), create_graph=True, only_inputs=True)[0]
    u_x = \
    torch.autograd.grad(u, x_inner_e, grad_outputs=torch.ones_like(x_inner_e), create_graph=True, only_inputs=True)[0]

    v_z = \
    torch.autograd.grad(v, z_inner_e, grad_outputs=torch.ones_like(z_inner_e), create_graph=True, only_inputs=True)[0]
    v_y = \
    torch.autograd.grad(v, y_inner_e, grad_outputs=torch.ones_like(y_inner_e), create_graph=True, only_inputs=True)[0]
    v_x = \
    torch.autograd.grad(v, x_inner_e, grad_outputs=torch.ones_like(x_inner_e), create_graph=True, only_inputs=True)[0]

    w_z = \
    torch.autograd.grad(w, z_inner_e, grad_outputs=torch.ones_like(z_inner_e), create_graph=True, only_inputs=True)[0]
    w_y = \
    torch.autograd.grad(w, y_inner_e, grad_outputs=torch.ones_like(y_inner_e), create_graph=True, only_inputs=True)[0]
    w_x = \
    torch.autograd.grad(w, y_inner_e, grad_outputs=torch.ones_like(x_inner_e), create_graph=True, only_inputs=True)[0]

    mu = 4e-3  # https://www.engineeringtoolbox.com/absolute-viscosity-liquids-d_1259.html

    e_n = torch.zeros(nr_WSSpoints, 3)  # normal unitvector. net_in is already of the form r*cos(radian) / X_scale
    stress = torch.zeros(nr_WSSpoints, 3, 3)  # 200 points, 3x3 matrix for each point
    stress_n = torch.zeros(nr_WSSpoints, 3)
    stress_nn = torch.zeros(nr_WSSpoints, 1)
    stress_nn_vec = torch.zeros(nr_WSSpoints, 3)
    WSS_vec = torch.zeros(nr_WSSpoints, 3)
    WSS_mag = torch.zeros(nr_WSSpoints, 1)

    for i in range(nr_WSSpoints):
        xn, yn, zn = net_in[i, :]
        vec = torch.tensor([xn, yn, zn])
        norm = math.sqrt(torch.inner(vec, vec))
        e_n[i, :] = vec / norm

        # building stress tensor
        p = P[i]  # local pressure
        ux_i = u_x[i]
        uy_i = u_y[i]
        uz_i = u_z[i]
        vx_i = v_x[i]
        vy_i = v_y[i]
        vz_i = v_z[i]
        wx_i = w_x[i]
        wy_i = w_y[i]
        wz_i = w_z[i]

        stress[i, 0, 0] = 2 * ux_i - p
        stress[i, 0, 1] = uy_i + vx_i
        stress[i, 0, 2] = uz_i + wx_i

        stress[i, 1, 0] = uy_i + vx_i
        stress[i, 1, 1] = 2 * vy_i - p
        stress[i, 1, 2] = vz_i + wy_i

        stress[i, 2, 0] = wx_i + uz_i
        stress[i, 2, 1] = wy_i + vz_i
        stress[i, 2, 2] = 2 * wz_i - p

        # niet vergeten stress aan het eind met mu te vermenigvuldigen

        stress_n[i, 0] = (2 * ux_i - p) * e_n[i, 0] + ((uy_i + vx_i) * e_n[i, 1]) + (
                    (uz_i + wx_i) * e_n[i, 2])  # rij komt overeen met index punt
        stress_n[i, 1] = (2 * uy_i - p) * e_n[i, 1] + ((uy_i + vx_i) * e_n[i, 0]) + ((vz_i + wy_i) * e_n[i, 2])
        stress_n[i, 2] = (2 * wz_i - p) * e_n[i, 2] + ((wx_i + uz_i) * e_n[i, 0]) + ((wy_i + vz_i) * e_n[i, 1])

        stress_nn[i] = e_n[i, 0] * stress_n[i, 0] + e_n[i, 1] * stress_n[i, 1] + e_n[i, 2] * stress_n[
            i, 2]  # make it a vector
        stress_nn_vec[i, 0] = stress_nn[i] * e_n[i, 0]
        stress_nn_vec[i, 1] = stress_nn[i] * e_n[i, 1]
        stress_nn_vec[i, 2] = stress_nn[i] * e_n[i, 2]

        WSS_vec[i, :] = torch.mul(mu, stress_n[i, :]) - torch.mul(mu, stress_nn_vec[i, :])  # WSS vector
        WSS_mag[i] = math.sqrt(torch.inner(WSS_vec[i, :], WSS_vec[i, :]))

    WSS_max = torch.max(WSS_mag).item()
    WSS_norm = torch.div(WSS_mag, WSS_max)

    X_max_c = torch.max(x_circle).item()
    X_min_c = torch.min(x_circle).item()
    # radius_c = (X_max_c - X_min_c) / 2  # radius of the circle

    frac_c = (radians[-1] - radians[0]) / (2 * np.pi)  # fractie van cirkel meegenomen voor WSS
    S = torch.linspace(0, radius * frac_c * 2 * np.pi, nr_WSSpoints)  # 'parametrization', circumferention

    ## CFD solution
    data = pd.read_csv('v_wall_y_1499_mesh1.csv', delimiter=',')
    data_ref = pd.read_csv('v_wall_y_1499_mesh2_refine.csv', delimiter=',')

    x_coord = np.array(data.iloc[:, 0])
    x_coord_ref = np.array(data_ref.iloc[:, 0])
    vel_u = np.array(data.iloc[:, 1])
    vel_u_ref = np.array(data_ref.iloc[:, 1])

    mu = 4e-3  # https://www.engineeringtoolbox.com/absolute-viscosity-liquids-d_1259.html
    diff_y = 0.0001

    du_y = -vel_u / diff_y
    du_y_ref = -vel_u_ref / diff_y
    WSS = -mu * du_y
    WSS_ref = - mu * du_y_ref
    WSS_max = np.max(WSS)
    WSS_max_ref = np.max(WSS_ref)
    WSS_norm = np.divide(WSS, WSS_max)
    WSS_norm_ref = np.divide(WSS_ref, WSS_max_ref)

    plt.figure()
    plt.plot(x_coord, WSS_norm)
    plt.plot(x_coord, WSS_norm_ref)
    plt.plot(S, WSS_norm)
    plt.legend(['mesh1', 'mesh2_refined', 'PINN'])
    plt.show()

    if show == True:
        plt.figure()
        plt.plot(S, WSS_norm)
        plt.xlabel('S')
        plt.ylabel('WSS/WSSmax')
        plt.title('WSS/WSSmax versus circumferention S')
        plt.show()

    return

def WSS_stenose(file, device, net2_u, net2_v, net2_p, path, U_scale, show=True):
    # gaat nu alleen van 2D stenose uit

    x, y, z, data_vtk = fileReader(file, mesh=True)  # FileReader doet geen schaling
    x = torch.tensor(x)
    y = torch.tensor(y)
    Y_end = torch.max(y).item()
    X_end = X_end = torch.max(x).item()
    res = Y_end / 100

    output_filename = path + "/Outputs/out_sten"
    x_begin = 0.575
    x_begin = float("{0:.3f}".format(x_begin))

    points_from_wall = 1
    loc_wall = Y_end - points_from_wall * res
    perc_wall = (1 - loc_wall / Y_end) * 100  # percentage from the wall compared to whole Y_domain
    perc_wall = float("{0:.2f}".format(perc_wall))

    loc_wall = float("{0:.4f}".format(loc_wall))
    y_size = y.data[y.data == Y_end].size()[0]

    y_data = torch.tensor(loc_wall)  # wat is de lengte van y_data
    y_data = y_data.repeat(y_size).to(device)
    y_data = y_data.view(len(y_data), -1)

    x_data = torch.linspace(x_begin, X_end, y_size).to(device)
    x_data = x_data.view(len(x_data), -1)

    x_data.requires_grad = True  # start bijhouden wijzigingen aan weights
    y_data.requires_grad = True

    net_in = torch.cat((x_data, y_data), 1)

    u = net2_u(net_in)
    u = u.view(len(u), -1)  # rearrange the tensor, geef het len(u) rijen, grootte column (-1) mag het zelf uitrekenen
    output_u = u.cpu().data.numpy()
    v = net2_v(net_in)
    v = v.view(len(v), -1)  # (256, 1 ) afhankelijk dus van batch_size grootte
    output_v = v.cpu().data.numpy()

    u_y = torch.autograd.grad(u, y_data, grad_outputs=torch.ones_like(y_data), only_inputs=True)[0]

    # Calculate WSS
    mu = 4e-3  # https://www.engineeringtoolbox.com/absolute-viscosity-liquids-d_1259.html

    WSS = torch.mul(u_y, -mu)
    WSS_max = torch.max(WSS).item()
    WSS_norm = torch.div(WSS, WSS_max)

    x_data = x_data.cpu()
    WSS_norm = WSS_norm.cpu()

    #TODO opslaan figuur na plotten, hoeft niet per se the showen

    plt.figure()
    plt.plot(x_data.detach().numpy(), WSS_norm.detach().numpy())

    plt.ylabel('WSS/WSS_max')
    plt.xlabel('x_pos (domain: [0 1])')
    plt.savefig(path + "/Plots/2DstenoseWSS")

    if show == True:
        plt.show()

    n_points = len(output_v)
    Velocity = np.zeros((n_points, 3))  # Velocity vector
    Velocity[:, 0] = output_u[:, 0] * U_scale
    Velocity[:, 1] = output_v[:, 0] * U_scale
    theta_vtk = convertArray(Velocity, 'Vel_PINN', data_vtk)
    output_p = net2_p(net_in)  # evaluate model
    output_p = output_p.cpu()
    output_p = output_p.data.numpy()

    theta_vtk = convertArray(output_p, 'P_PINN', data_vtk)
    myoutput = vtk.vtkDataSetWriter()
    myoutput.SetInputData(data_vtk)
    myoutput.SetFileName(output_filename)
    myoutput.Write()

    return WSS, WSS_norm

def convertArray(output, name, data_vtk):
    # Velocity[:, 2] = output_w[:, 0] * U_scale

    # Save VTK
    result = VN.numpy_to_vtk(output)
    result.SetName(name)
    data_vtk.GetPointData().AddArray(result)

    return result