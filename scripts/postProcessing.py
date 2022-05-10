import torch
import numpy as np
# import foamFileOperation
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.optim as optim
import pdb
# from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from math import exp, sqrt, pi
from Verified.scripts.NNbuilder import Swish, Net2, MySquared, Net2P
from Verified.scripts.losses import criterion, Loss_BC, Loss_data
from Verified.scripts.fileReaders import fileReader, planeReader
import scipy.stats as stats

from Verified.scripts.NNbuilder import NNreader
import time
import vtk
import vtkmodules.util.numpy_support as VN  # PyCharm compatible code


# from PINNtoVTK import convert_to_VTK


# plot the loss on CPU (first load the net)
def WSS_calculation(case, device, data_u, data_v, data_w, data_p, check_normal=False):
    nPt = 200

    net2_u = Net2(case.input_n, case.h_n).to(device)
    net2_v = Net2(case.input_n, case.h_n).to(device)
    net2_w = Net2(case.input_n, case.h_n).to(device)
    net2_p = Net2P(case.input_n, case.h_n).to(device)

    print('Reading (pretrain) functions first...')
    net2_u.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_u + ".pt"))
    net2_v.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_v + ".pt"))
    net2_p.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_p + ".pt"))
    if case.input_n == 3:
        net2_w.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_w + ".pt"))

    M_np = np.array([[case.xM], [case.yM], [0]])
    radius_np = case.radius
    theta = np.linspace(case.theta[0], case.theta[1], nPt)
    circle = np.array([radius_np * np.cos(theta) + M_np[0], radius_np * np.sin(theta) + M_np[1]])
    # TODO: sphere coordinaten? voor volledige beschrijving 3D optie

    e_n = np.zeros((nPt, 3))
    len_e = np.zeros(nPt)

    for i in range(nPt):
        e_n_i = np.array([circle[0, i] - M_np[0], circle[1, i] - M_np[1]])
        mag = np.sqrt(e_n_i[0] ** 2 + e_n_i[1] ** 2)
        e_n[i, 0] = (circle[0, i] - M_np[0]) / mag * -1
        e_n[i, 1] = (circle[1, i] - M_np[1]) / mag * -1
        len_e[i] = np.sqrt(e_n[i, 0] ** 2 + e_n[i, 1] ** 2)

    x_circ = circle[0, :]
    x_circ = x_circ.reshape(-1, 1)
    y_circ = circle[1, :]
    y_circ = y_circ.reshape(-1, 1)
    e_nx = e_n[:, 0]
    e_nx = e_nx.reshape(-1, 1)
    e_ny = e_n[:, 1]
    e_ny = e_ny.reshape(-1, 1)
    e_nz = e_n[:, 2]
    e_nz = e_nz.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(9, 9))
    skip = (slice(None, None, 5), slice(None, None, 5))  # plot every 5 pts
    ax.quiver(x_circ[skip], y_circ[skip], e_nx[skip], e_ny[skip],
              scale=50)  # a smaller scale parameter makes the arrow longer.
    plt.title('Normal vector check')
    ax.axis('equal')
    if check_normal:
        plt.show()

    x_circle = torch.tensor(x_circ)
    y_circle = torch.tensor(y_circ)
    z_circle = torch.zeros(nPt)
    x_circle = x_circle.type(torch.cuda.FloatTensor) if device == "cuda" else x_circle.type(torch.FloatTensor)
    y_circle = y_circle.type(torch.cuda.FloatTensor) if device == "cuda" else y_circle.type(torch.FloatTensor)
    z_circle = z_circle.type(torch.cuda.FloatTensor) if device == "cuda" else z_circle.type(torch.FloatTensor)

    y_circle = y_circle.view(nPt, -1)  # vs len(y_data) in 2D stenose
    x_circle = x_circle.view(nPt, -1)
    z_circle = z_circle.view(nPt, -1)
    # y_inner_c = y_inner_c.view(nPt, -1)
    # x_inner_c = x_inner_c.view(nPt, -1)

    # x_inner_e = x_inner_c / X_scale
    # y_inner_e = y_inner_c / Y_scale

    x_inner_e = x_circle / case.X_scale
    y_inner_e = y_circle / case.Y_scale
    z_inner_e = z_circle / case.Z_scale

    x_inner_e = x_inner_e.to(device)
    y_inner_e = y_inner_e.to(device)
    z_inner_e = z_inner_e.to(device)
    x_inner_e.requires_grad = True
    y_inner_e.requires_grad = True
    z_inner_e.requires_grad = True

    net_in = torch.cat((x_inner_e, y_inner_e, z_inner_e), 1) if case.input_n == 3 else torch.cat((x_inner_e, y_inner_e),
                                                                                                 1)  # inner ellips coordinates are put into NN
    # net_in = torch.cat((x, y), 1)

    u = net2_u(net_in)
    u = u.view(len(u), -1)  # rearrange the tensor, geef het len(u) rijen, grootte column (-1) mag het zelf uitrekenen
    v = net2_v(net_in)
    v = v.view(len(v), -1)  # (256, 1 ) afhankelijk dus van batch_size grootte
    P = net2_p(net_in)
    P = P.view(len(P), -1)

    u_y = \
        torch.autograd.grad(u, y_inner_e, grad_outputs=torch.ones_like(y_inner_e), create_graph=True, only_inputs=True)[
            0]
    u_x = \
        torch.autograd.grad(u, x_inner_e, grad_outputs=torch.ones_like(x_inner_e), create_graph=True, only_inputs=True)[
            0]

    v_y = \
        torch.autograd.grad(v, y_inner_e, grad_outputs=torch.ones_like(y_inner_e), create_graph=True, only_inputs=True)[
            0]
    v_x = \
        torch.autograd.grad(v, x_inner_e, grad_outputs=torch.ones_like(x_inner_e), create_graph=True, only_inputs=True)[
            0]

    if case.input_n == 3:
        w = net2_w(net_in)
        w = w.view(len(w), -1)

        u_z = \
            torch.autograd.grad(u, z_inner_e, grad_outputs=torch.ones_like(z_inner_e), create_graph=True,
                                only_inputs=True)[
                0]
        v_z = \
            torch.autograd.grad(v, z_inner_e, grad_outputs=torch.ones_like(z_inner_e), create_graph=True,
                                only_inputs=True)[
                0]

        w_z = \
            torch.autograd.grad(w, z_inner_e, grad_outputs=torch.ones_like(z_inner_e), create_graph=True,
                                only_inputs=True)[0]
        w_y = \
            torch.autograd.grad(w, y_inner_e, grad_outputs=torch.ones_like(y_inner_e), create_graph=True,
                                only_inputs=True)[0]
        w_x = \
            torch.autograd.grad(w, y_inner_e, grad_outputs=torch.ones_like(x_inner_e), create_graph=True,
                                only_inputs=True)[0]

    mu = 4e-3  # https://www.engineeringtoolbox.com/absolute-viscosity-liquids-d_1259.html

    stress = torch.zeros(nPt, 3, 3)  # 200 points, 3x3 matrix for each point
    stress_n = torch.zeros(nPt, 3)
    stress_nn = torch.zeros(nPt, 1).to(device)
    stress_nn_vec = torch.zeros(nPt, 3)
    WSS_vec = torch.zeros(nPt, 3)
    WSS_mag = torch.zeros(nPt, 1)
    #
    # stress = torch.zeros(nPt, 2, 2)  # 200 points, 3x3 matrix for each point
    # stress_n = torch.zeros(nPt, 2)
    # stress_nn = torch.zeros(nPt, 1).to(device)
    # stress_nn_vec = torch.zeros(nPt, 2)
    # WSS_vec = torch.zeros(nPt, 2)
    # WSS_mag = torch.zeros(nPt, 1)

    e_nx = torch.tensor(e_nx).to(device)
    e_ny = torch.tensor(e_ny).to(device)
    e_nz = torch.tensor(e_nz).to(device)

    for i in range(nPt):

        # building stress tensor
        p = P[i]  # local pressure
        ux_i = u_x[i]
        uy_i = u_y[i]
        vx_i = v_x[i]
        vy_i = v_y[i]

        if case.input_n == 3:
            uz_i = u_z[i]
            vz_i = v_z[i]
            wx_i = w_x[i]
            wy_i = w_y[i]
            wz_i = w_z[i]
        else:
            uz_i = 0
            vz_i = 0
            wx_i = 0
            wy_i = 0
            wz_i = 0

        stress[i, 0, 0] = (2 * mu * ux_i - p)
        stress[i, 0, 1] = (uy_i + vx_i) * mu
        stress[i, 0, 2] = (uz_i + wx_i) * mu

        stress[i, 1, 0] = (uy_i + vx_i) * mu
        stress[i, 1, 1] = (2 * mu * vy_i - p)
        stress[i, 1, 2] = (vz_i + wy_i) * mu

        stress[i, 2, 0] = (wx_i + uz_i) * mu
        stress[i, 2, 1] = (wy_i + vz_i) * mu
        stress[i, 2, 2] = (2 * mu * wz_i - p)

        stress_n[i, 0] = stress[i, 0, 0] * e_nx[i] + stress[i, 0, 1] * e_ny[i] + stress[i, 0, 2] * e_nz[i]
        stress_n[i, 1] = stress[i, 1, 0] * e_nx[i] + stress[i, 1, 1] * e_ny[i] + stress[i, 1, 2] * e_nz[i]
        stress_n[i, 2] = stress[i, 2, 0] * e_nx[i] + stress[i, 2, 1] * e_ny[i] + stress[i, 2, 2] * e_nz[i]

        stress_nn[i] = e_nx[i] * stress_n[i, 0] + e_ny[i] * stress_n[i, 1] + e_nz[i] * stress_n[
            i, 2]  # make it a vector
        stress_nn_vec[i, 0] = stress_nn[i] * e_nx[i]
        stress_nn_vec[i, 1] = stress_nn[i] * e_ny[i]
        stress_nn_vec[i, 2] = stress_nn[i] * e_nz[i]

        # WSS_vec[i, :] = torch.mul(mu, stress_n[i, :]) - torch.mul(mu, stress_nn_vec[i, :])  # WSS vector
        # WSS_mag[i] = math.sqrt(torch.inner(WSS_vec[i, :], WSS_vec[i, :]))

        WSS_vec[i, :] = stress_n[i, :] - stress_nn_vec[i, :]  # WSS vector
        WSS_mag[i] = math.sqrt(torch.inner(WSS_vec[i, :], WSS_vec[i, :]))
        # WSS_mag2 = torch.zeros(nPt, 1)
        # WSS_mag2[i] = math.sqrt(torch.inner(WSS_vec[i, 0:2], WSS_vec[i, 0:2]))
        # print('vec', WSS_vec[0, 0:2])
    WSS_max = torch.max(WSS_mag).item()
    WSS_norm = torch.div(WSS_mag, WSS_max)

    # for i in range(nPt):
    #     # building stress tensor
    #     p = P[i]  # local pressure
    #     ux_i = u_x[i]
    #     uy_i = u_y[i]
    #     vx_i = v_x[i]
    #     vy_i = v_y[i]
    #
    #     if case.input_n == 3:
    #         uz_i = u_z[i]
    #         vz_i = v_z[i]
    #         wx_i = w_x[i]
    #         wy_i = w_y[i]
    #         wz_i = w_z[i]
    #     else:
    #         uz_i = 0
    #         vz_i = 0
    #         wx_i = 0
    #         wy_i = 0
    #         wz_i = 0
    #
    #
    #     stress[i, 0, 0] = 2 * mu * ux_i - p
    #     stress[i, 0, 1] = (uy_i + vx_i) * mu
    #     # stress[i, 0, 2] = (uz_i + wx_i) * mu
    #
    #     stress[i, 1, 0] = (uy_i + vx_i) * mu
    #     stress[i, 1, 1] = 2 * mu * vy_i - p
    #     # stress[i, 1, 2] = (vz_i + wy_i) * mu
    #
    #     # stress[i, 2, 0] = (wx_i + uz_i) * mu
    #     # stress[i, 2, 1] = (wy_i + vz_i) * mu
    #     # stress[i, 2, 2] = 2 * mu * wz_i - p
    #
    #     # niet vergeten stress aan het eind met mu te vermenigvuldigen
    #
    #     stress_n[i, 0] = (2 * ux_i - p) * e_nx[i] + ((uy_i + vx_i) * e_ny[i]) # + (uz_i + wx_i) * e_nz[i] # rij komt overeen met index punt
    #     stress_n[i, 1] = (2 * uy_i - p) * e_ny[i] + ((uy_i + vx_i) * e_nx[i]) # + (vz_i + wy_i) * e_nz[i]
    #     # stress_n[i, 2] = (wx_i + uz_i) * e_nx[i] + (wy_i + vz_i) * e_ny[i] + (2 * wz_i - p) * e_nz[i]
    #     # stress_n[i, 0] = (2 * ux_i - p) * e_nx[i] + ((uy_i + vx_i) * e_ny[i])
    #     # stress_nn[i] = e_nx[i] * stress_n[i, 0] + e_ny[i] * stress_n[i, 1]  # make it a vector
    #
    #     stress_nn_vec[i, 0] = stress_nn[i] * e_nx[i]
    #     stress_nn_vec[i, 1] = stress_nn[i] * e_ny[i]
    #     # stress_nn_vec[i, 2] = stress_nn[i] * e_nz[i]
    #
    #     WSS_vec[i, :] = torch.mul(mu, stress_n[i, :]) - torch.mul(mu, stress_nn_vec[i, :])  # WSS vector
    #     # WSS_vec[i, :] = stress_n[i, :] - stress_nn_vec[i, :]
    #     WSS_mag[i] = math.sqrt(torch.inner(WSS_vec[i, :], WSS_vec[i, :]))
    #
    #     # WSS_vec[i, :] = stress_n[i, :] - stress_nn_vec[i, :]  # WSS vector
    #     # WSS_mag[i] = math.sqrt(torch.inner(WSS_vec[i, :], WSS_vec[i, :]))
    #
    # WSS_max = torch.max(WSS_mag).item()
    # WSS_norm2 = torch.div(WSS_mag, WSS_max)

    frac_c = (theta[-1] - theta[0]) / (2 * np.pi)  # fractie van cirkel meegenomen voor WSS
    S = torch.linspace(0, radius_np * frac_c * 2 * np.pi, nPt)  # 'parametrization', circumferention

    plt.figure(5)
    plt.plot(S, WSS_norm)
    # plt.plot(S, WSS_norm2)
    plt.xlabel('S')
    plt.ylabel('WSS/WSSmax')
    plt.title('WSS/WSSmax versus circumferention S')
    # plt.legend(['mu pre', 'mu post'])
    plt.show()

    plt.savefig(case.path + "Outputs/" + case.pretrain + "WSS" + ".png")

    return WSS_vec


def post_process(case, file, data_u, data_v, data_w, data_p, Flag_physical, plot_vecfield=True,
                 plot_streamline=True, writeVTK=True):
    device = "cpu"
    # TODO tijdens trainen device = cuda, anders cpu --> testen
    if (not Flag_physical):
        X_scale = 1.
        Y_scale = 1.

    x, y, z, _ = fileReader(file, case.input_n, mesh=True)  # nog niet geschaald naar normalizatie domein
    # TODO: aantal datapunten stuk kleiner als meshfile wordt gelezen en niet wordt geschaald. Werkt dit dan nog steeds?

    # x = x / case.X_scale  # TODO waar schalen?
    # y = y / case.Y_scale
    # z = z / case.Z_scale

    x = torch.Tensor(x).to(device).type(torch.FloatTensor)
    y = torch.Tensor(y).to(device).type(torch.FloatTensor)
    z = torch.tensor(z).to(device).type(torch.FloatTensor)

    # if case.input_n == 3:
    #     u, v, p, w, net_u, net_v, net_p, net_w = NNreader(case, device, data_u="data_u_e8000", data_v="data_v_e8000",
    #                                                   data_w="data_w_e8000", data_p="data_p_e8000")
    # else:
    #     u, v, p, net2_u, net2_v, net2_p = NNreader(case, device, data_u="data_u_e5500", data_v="data_v_e5500",
    #                                                data_p="data_p_e5500")
    net2_u = Net2(case.input_n, case.h_n).to(device)
    net2_v = Net2(case.input_n, case.h_n).to(device)
    net2_p = Net2P(case.input_n, case.h_n).to(device)
    net2_w = Net2(case.input_n, case.h_n).to(device)

    print('load the network')
    net2_u.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_u + ".pt"))
    net2_v.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_v + ".pt"))
    net2_p.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_p + ".pt"))
    if case.input_n == 3:
        net2_w.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_w + ".pt"))
        net2_w.eval()

    net2_u.eval()
    net2_v.eval()
    net2_p.eval()

    if writeVTK:
        create_vtk(case, data_u, data_v, data_w, data_p)
        # convert_to_VTK()

    if plot_vecfield:
        if case.input_n == 3:
            x, y, z = planeReader(file, case.input_n, mesh=True, z_plane=0)
            x = torch.tensor(x).to(device).type(torch.FloatTensor)
            y = torch.tensor(y).to(device).type(torch.FloatTensor)
            z = torch.tensor(z).to(device).type(torch.FloatTensor)
        vectorfield(case, x, y, z, net2_u, net2_v)

    if plot_streamline:
        streamline(case, device, net2_u, net2_v)


# TODO: vectorfield gaat wsl buggen als 3D aneurysma
def vectorfield(case, x, y, z, net2_u, net2_v):
    net_in = torch.cat((x.requires_grad_(), y.requires_grad_(), z.requires_grad_()), 1)

    output_u = net2_u(net_in)
    output_u = output_u.cpu()
    output_u = output_u.data.numpy()
    output_v = net2_v(net_in)
    output_v = output_v.cpu()  # evaluate model
    output_v = output_v.data.numpy()
    # Normalize the arrows:
    U = output_u / np.sqrt(output_u ** 2 + output_v ** 2)
    V = output_v / np.sqrt(output_u ** 2 + output_v ** 2)
    plt.figure()
    fig, ax = plt.subplots(figsize=(9, 9))
    if case.input_n == 3:
        skip = (slice(None, None, 5), slice(None, None, 5))  # plot every 5 pts
    else:
        skip = (slice(None, None, 1), slice(None, None, 1))
    # ax.quiver(x.detach().numpy(), y.detach().numpy(), output_u , output_v,scale=5)
    ax.quiver(x.detach().numpy()[skip], y.detach().numpy()[skip], U[skip], V[skip],
              scale=50)  # a smaller scale parameter makes the arrow longer.
    plt.title('NN results, Vel vector')
    plt.show()

    plt.savefig(case.path + "Outputs/" + case.pretrain + "vecfield" + ".png")


def streamline(case, device, net2_u, net2_v):
    # Streamline  # TODO: domein waarover deze plot nuttig is vinden

    nPt = 130
    xStart = 0.
    xEnd = 1.
    yStart = 0.
    yEnd = 1.0

    plt.figure()
    fig2, ax2 = plt.subplots()

    xs = np.linspace(xStart, xEnd, nPt)
    ys = np.linspace(yStart, yEnd, nPt)
    zs = np.zeros(len(xs) ** 2)
    xs, ys = np.meshgrid(xs, ys)
    xs = torch.tensor(xs).to(device)
    ys = torch.tensor(ys).to(device)
    zs = torch.tensor(zs).to(device)

    xs = xs.type(torch.FloatTensor)
    ys = ys.type(torch.FloatTensor)
    zs = zs.type(torch.FloatTensor)

    xs = xs.view(nPt ** 2, -1)
    ys = ys.view(nPt ** 2, -1)
    zs = zs.view(nPt ** 2, -1)

    net_in = torch.cat((xs.requires_grad_(), ys.requires_grad_(), zs.requires_grad_()), 1)
    output_us = net2_u(net_in)
    output_us = output_us.view(nPt, nPt)
    output_us = output_us.data.numpy()
    output_vs = net2_v(net_in)
    output_vs = output_vs.view(nPt, nPt)
    output_vs = output_vs.data.numpy()

    xs = xs.view(nPt, nPt)
    ys = ys.view(nPt, nPt)
    xs = xs.data.numpy()
    ys = ys.data.numpy()

    ax2.streamplot(xs, ys, output_us, output_vs, density=0.5)
    plt.title('NN results, Vel SL')
    plt.show()

    plt.savefig(case.path + "Outputs/" + case.pretrain + "streaml" + ".png")


def create_vtk(case, data_u, data_v, data_w, data_p):
    print('Writing output file..')
    device = 'cpu'
    reader = vtk.vtkXMLUnstructuredGridReader()
    # reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(case.mesh_file)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()

    x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    z_vtk_mesh = np.zeros((n_points, 1))
    VTKpoints = vtk.vtkPoints()
    for i in range(n_points):
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = pt_iso[0]
        y_vtk_mesh[i] = pt_iso[1]
        z_vtk_mesh[i] = pt_iso[2]
        VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)

    x = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh[:]), 1))  # / case.X_scale
    y = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1))  # / case.Y_scale
    z = np.reshape(z_vtk_mesh, (np.size(z_vtk_mesh[:]), 1))  # / case.Z_scale

    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)
    z = torch.Tensor(z).to(device)

    net2_u = Net2(case.input_n, case.h_n).to(device)
    net2_v = Net2(case.input_n, case.h_n).to(device)
    net2_w = Net2(case.input_n, case.h_n).to(device)
    net2_p = Net2(case.input_n, case.h_n).to(device)

    net2_u.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_u + ".pt"))  # TODO algemeen
    net2_v.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_v + ".pt"))
    if case.input_n == 3:
        net2_w.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_w + ".pt"))
    net2_p.load_state_dict(torch.load(case.path + "NNfiles/" + case.pretrain + data_p + ".pt"))

    net_in = torch.cat((x.requires_grad_(), y.requires_grad_(), z.requires_grad_()), 1) if case.input_n == 3 else \
        torch.cat((x.requires_grad_(), y.requires_grad_()), 1)
    # net_in = torch.cat((x.requires_grad_(), y.requires_grad_()), 1)
    output_u = net2_u(net_in)  # evaluate model
    output_u = output_u.data.numpy()
    output_v = net2_v(net_in)  # evaluate model
    output_v = output_v.data.numpy()

    Velocity = np.zeros((n_points, 3))  # Velocity vector
    Velocity[:, 0] = output_u[:, 0] * case.U_scale
    Velocity[:, 1] = output_v[:, 0] * case.U_scale
    if case.input_n == 3:
        output_w = net2_w(net_in)
        output_w = output_w.data.numpy()
        Velocity[:, 2] = output_w[:, 0] * case.U_scale

    # Save VTK
    theta_vtk = VN.numpy_to_vtk(Velocity)
    theta_vtk.SetName('Vel_PINN')  # TAWSS vector
    data_vtk.GetPointData().AddArray(theta_vtk)

    output_p = net2_p(net_in)  # evaluate model
    output_p = output_p.data.numpy()

    theta_vtk = VN.numpy_to_vtk(output_p)
    theta_vtk.SetName('P_PINN')
    data_vtk.GetPointData().AddArray(theta_vtk)

    myoutput = vtk.vtkDataSetWriter()
    myoutput.SetInputData(data_vtk)
    myoutput.SetFileName(case.output_filename)
    myoutput.Write()

    print('output file written')


def post(case, epochs, net2_u, net2_v, net2_w, net2_p, output_filename, ex_ID, plot_vecfield=True,
         plot_streamline=True, writeVTK = True):
    # device = torch.device("cuda")
    device = torch.device("cpu")

    nPt = 130
    xStart = 0.
    xEnd = 1.
    yStart = 0.
    yEnd = 1.0
    delta_wall = 0.2  # gelijk aan delta_circ?

    Flag_plot = True  # True: for also plotting in python
    Flag_physical = False  # False #True #IF True use the physical mesh, not the normalized dimension mesh

    net2_u = net2_u.cpu()
    net2_v = net2_v.cpu()
    net2_w = net2_w.cpu()
    net2_p = net2_p.cpu()

    if (Flag_physical):
        mesh_file = case.vel_file
        X_scale = case.X_scale
        Y_scale = case.Y_scale
        Z_scale = case.Z_scale
        U_scale = case.U_scale
    else:
        mesh_file = case.mesh_file
        X_scale = 1.
        Y_scale = 1.
        Z_scale = 1.
        U_scale = 1.

    print('Loading', mesh_file)  # TODO: is nog een keer inlezen wel nodig?
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()
    print('n_points of the mesh:', n_points)
    x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    z_vtk_mesh = np.zeros((n_points, 1))
    VTKpoints = vtk.vtkPoints()
    for i in range(n_points):
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = pt_iso[0]
        y_vtk_mesh[i] = pt_iso[1]
        z_vtk_mesh[i] = pt_iso[2]
        VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)

    print('Net input x: [', np.min(x_vtk_mesh), np.max(x_vtk_mesh), ']')
    print('Net input y: [', np.min(y_vtk_mesh), np.max(y_vtk_mesh), ']')
    print('Net input z: [', np.min(z_vtk_mesh), np.max(z_vtk_mesh), ']')

    x = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh[:]), 1)) / X_scale
    y = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1)) / Y_scale
    z = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1)) / Z_scale
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)
    z = torch.Tensor(z).to(device)

    if case.input_n == 3:
        x = x[0::10]
        y = y[0::10]
        z = z[0::10]

    net_in = torch.cat((x.requires_grad_(), y.requires_grad_(), z.requires_grad_()), 1) if case.input_n == 3 \
        else torch.cat((x.requires_grad_(), y.requires_grad_()), 1)
    output_u = net2_u(net_in)  # evaluate model
    output_u = output_u.cpu()
    output_u = output_u.data.numpy()
    output_v = net2_v(net_in)  # evaluate model
    output_v = output_v.cpu()
    output_v = output_v.data.numpy()

    # x = x[0::100]  # TODO volledige data voor echte runs
    # y = y[0::100]
    # z = z[0::100]
    #
    # output_u = output_u[0::100]
    # output_v = output_v[0::100]

    if writeVTK:
        # creating VTK
        Velocity = np.zeros((len(x), 3))  # Velocity vector
        Velocity[:, 0] = output_u[:, 0] * U_scale
        Velocity[:, 1] = output_v[:, 0] * U_scale

        if case.input_n == 3:
            output_w = net2_w(net_in)
            output_w = output_w.cpu()
            output_w = output_w.data.numpy()
            Velocity[:, 2] = output_w[:, 0] * U_scale

        Loss_plot, Loss_u, Loss_div = criterion(x, y, z, net2_u, net2_v, net2_w, net2_p, X_scale, Y_scale, Z_scale, U_scale
                                                , case.Diff, case.rho, case.input_n, criterion_plot=True)

        Loss_plot = Loss_plot.data.numpy()
        Loss_net = np.zeros((len(x), 1))
        Loss_net[:, 0] = Loss_plot[:, 0]

        # Save VTK

        theta_vtk = VN.numpy_to_vtk(Velocity)
        theta_vtk.SetName('Vel_PINN')  # TAWSS vector
        data_vtk.GetPointData().AddArray(theta_vtk)

        theta_vtk = VN.numpy_to_vtk(Loss_net)
        theta_vtk.SetName('loss_net')
        data_vtk.GetPointData().AddArray(theta_vtk)

        output_p = net2_p(net_in)  # evaluate model
        output_p = output_p.cpu()
        output_p = output_p.data.numpy()
        theta_vtk = VN.numpy_to_vtk(output_p)
        theta_vtk.SetName('P_PINN')
        data_vtk.GetPointData().AddArray(theta_vtk)

        Loss_plot_u = Loss_u.data.numpy()
        Loss_net_u = np.zeros((len(x), 1))
        Loss_net_u[:, 0] = Loss_plot_u[:, 0]
        theta_vtk = VN.numpy_to_vtk(Loss_net_u)
        theta_vtk.SetName('loss_net_u')
        data_vtk.GetPointData().AddArray(theta_vtk)

        Loss_plot_div = Loss_div.data.numpy()
        Loss_net_div = np.zeros((len(x), 1))
        Loss_net_div[:, 0] = Loss_plot_div[:, 0]
        theta_vtk = VN.numpy_to_vtk(Loss_net_div)
        theta_vtk.SetName('loss_net_div')
        data_vtk.GetPointData().AddArray(theta_vtk)

        myoutput = vtk.vtkDataSetWriter()
        myoutput.SetInputData(data_vtk)
        myoutput.SetFileName(output_filename)
        myoutput.Write()

        print('output file written!')
    # net_in = torch.cat((x.requires_grad_(), y.requires_grad_(), z.requires_grad_()),
    #                    1) if case.input_n == 3 else \
    #     torch.cat((x.requires_grad_(), y.requires_grad_()), 1)
    # net_in = torch.cat((x.requires_grad_(), y.requires_grad_()), 1)
    # output_u = net2_u(net_in)  # evaluate model
    #
    # output_v = net2_v(net_in)  # evaluate model
    # output_u = output_u.cpu().data.numpy()  # need to convert to cpu before converting to numpy
    # output_v = output_v.cpu().data.numpy()
    # x = x.cpu()
    # y = y.cpu()

    plt.figure()  #2D plot
    plt.subplot(2, 1, 1)
    plt.scatter(z.detach().numpy(), x.detach().numpy(), c=output_w, cmap='rainbow') if case.input_n == 3 \
        else plt.scatter(x.detach().numpy(), y.detach().numpy(), c=output_u, cmap='rainbow')
    plt.title('NN results, u (top) & v (bot) - epoch' + str(epochs))
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.scatter(z.detach().numpy(), x.detach().numpy(), c=output_u, cmap='rainbow') if case.input_n == 3 \
        else plt.scatter(x.detach().numpy(), y.detach().numpy(), c=output_v, cmap='rainbow')
    plt.colorbar()
    plt.savefig(case.path + "/plots/" + case.ID + "plotWU_" + ex_ID + "_" + str(epochs)) if case.input_n == 3 \
        else plt.savefig(case.path + "/plots/" + case.ID + "plotUV_" + ex_ID + "_" + str(epochs))
    # plt.show()
    # u_pred = np.tile(output_u, (1, len(output_u)))
    # v_pred = np.tile(output_v, (1, len(output_v)))
    # XX, YY = np.meshgrid(x.detach().numpy(),y.detach().numpy())
    # fig = plt.figure(figsize=(18, 5))
    # plt.subplot(1, 2, 1)
    # plt.pcolor(XX, YY, u_pred, cmap='jet', shading='auto')
    # plt.colorbar()
    # plt.xlabel('$t$')
    # plt.ylabel('$x$')
    # plt.title(r'Exact $u(x)$')
    # plt.tight_layout()
    #
    # plt.subplot(1, 2, 2)
    # plt.pcolor(XX, YY, v_pred, cmap='jet', shading='auto')
    # plt.colorbar()
    # plt.xlabel('$t$')
    # plt.ylabel('$x$')
    # plt.title(r'Predicted $u(x)$')
    # plt.tight_layout()
    # plt.show()

    # #TODO voor 3D andere assen
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.scatter(x.detach().numpy(), y.detach().numpy(), c=output_u, cmap='rainbow')
    # # plt.scatter(x.detach().numpy(), y.detach().numpy(), c = output_u ,vmin=0, vmax=0.58, cmap = 'rainbow')
    # plt.title('NN results, u')
    # plt.colorbar()
    # plt.show()

    # if plot_vecfield:
    #     if case.input_n == 3:
    #         x, y, z = planeReader(mesh_file, case.input_n, mesh=True, z_plane=0)
    #         x = torch.tensor(x).to(device).type(torch.FloatTensor)
    #         y = torch.tensor(y).to(device).type(torch.FloatTensor)
    #         z = torch.tensor(z).to(device).type(torch.FloatTensor)
    #     vectorfield(case, x, y, z, net2_u, net2_v, ex_ID)

    # if plot_streamline:
    #     streamline(case, device, net2_u, net2_v)

    # if (Flag_plot):  # Calculate WSS at the bottom wall
    #     xw = np.linspace(xStart + delta_wall, xEnd, nPt)
    #     yw = np.linspace(yStart, yStart, nPt)
    #     xw = np.reshape(xw, (np.size(xw[:]), 1))
    #     yw = np.reshape(yw, (np.size(yw[:]), 1))
    #     xw = torch.Tensor(xw).to(device)
    #     yw = torch.Tensor(yw).to(device)
    #
    #     wss = WSS(xw, yw, net2_u, case.Diff, case.rho)
    #     wss = wss.data.numpy()
    #
    #     plt.figure()
    #     plt.plot(xw.detach().numpy(), wss[0:nPt], 'go', label='Predict-WSS', alpha=0.5)  # PINN
    #     plt.legend(loc='best')
    #     plt.show()
    #
    # if (Flag_plot):  # Calculate near-wall velocity
    #     xw = np.linspace(xStart + delta_wall, xEnd, nPt)
    #     yw = np.linspace(yStart + 0.02, yStart + 0.02, nPt)
    #     xw = np.reshape(xw, (np.size(xw[:]), 1))
    #     yw = np.reshape(yw, (np.size(yw[:]), 1))
    #     xw = torch.Tensor(xw).to(device)
    #     yw = torch.Tensor(yw).to(device)
    #
    #     net_in = torch.cat((xw, yw), 1)
    #     output_u = net2_u(net_in)  # evaluate model
    #     output_u = output_u.data.numpy()
    #
    #     plt.figure()
    #     plt.plot(xw.detach().numpy(), output_u[0:nPt], 'go', label='Near-wall vel', alpha=0.5)  # PINN
    #     plt.legend(loc='best')
    #     plt.show()

    # print('Loading', mesh_file)
    # reader = vtk.vtkXMLUnstructuredGridReader()
    # reader.SetFileName(mesh_file)
    # reader.Update()
    # data_vtk = reader.GetOutput()

    print('Done!')

    return


def GradLosses(case, device, ex_ID, epoch, epoch_pretrain, x, y, z, xb, yb, zb, xd, yd, zd, ud, vd, wd, net2_u,
               net2_v, net2_w, net2_p, alpha, anneal_weight, Flag_grad, Lambda_BC, Lambda_data, Flag_notrain=True):
    # calculating gradients wrt losses, plotting for each layer

    if Flag_notrain:
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

    loss_eqn, grads = criterion(x, y, z, net2_u, net2_v, net2_w, net2_p, case.X_scale, case.Y_scale,
                         case.Z_scale, case.U_scale, case.Diff, case.rho, case.input_n)
    loss_bc = Loss_BC(xb, yb, zb, net2_u, net2_v, net2_w, case.input_n)
    loss_data = Loss_data(xd, yd, zd, ud, vd, wd, net2_u, net2_v, net2_w, case.input_n)

    # anneal_weight = [[1.] for _ in range(nr_losses - 1)]
    nr_layers = 10  # includes input and output layers
    max_grad_r_list = []
    mean_grad_bc_list = []
    mean_grad_data_list = []
    bins_res_list = [[] for _ in range(nr_layers-1)]
    y_res_list = [[] for _ in range(nr_layers-1)]
    bins_bc_list = [[] for _ in range(nr_layers - 1)]
    y_bc_list = [[] for _ in range(nr_layers - 1)]
    bins_data_list = [[] for _ in range(nr_layers - 1)]
    y_data_list = [[] for _ in range(nr_layers - 1)]
    loss_eqn.backward(retain_graph=True)

    fig_nr = 100 + epoch

    layer = 0
    if Flag_grad:
        bins = 50
        plt.figure(fig_nr)
        print('Analyzing gradients wrt residual loss')
    for name, param in net2_u.named_parameters(prefix=''):
        if "weight" in name and layer < nr_layers - 1:
            layer += 1
            if Flag_grad:

                print(f"Layer: {name} | Size: {param.size()}")
                data = param.grad.cpu().detach().numpy()

                (mu, sigma) = stats.norm.fit(data)
                n, bins_res, patches = plt.hist(data, bins)
                y_res = stats.norm.pdf(bins_res, mu, sigma)
                bins_res_list[layer-1] = bins_res
                y_res_list[layer-1] = y_res
            max_grad_r_list.append(torch.max(abs(param.grad)))


    # plt.close(fig_nr)
    loss_bc.backward(retain_graph=True)
    if Flag_grad:
        print('Analyzing gradients wrt boundary loss')
        plt.figure(fig_nr)
    layer = 0
    for name, param in net2_u.named_parameters(prefix=''):
        if "weight" in name and layer < nr_layers - 1:
            layer += 1
            if Flag_grad:

                print(f"Layer: {name} | Size: {param.size()}")
                data = param.grad.cpu().detach().numpy()

                (mu, sigma) = stats.norm.fit(data)
                n, bins_bc, patches = plt.hist(data, bins)
                y_bc = stats.norm.pdf(bins_bc, mu, sigma)
                bins_bc_list[layer - 1] = bins_bc
                y_bc_list[layer - 1] = y_bc

            mean_grad_bc_list.append(torch.mean(abs(param.grad)))
    # plt.close(fig_nr)
    loss_data.backward(retain_graph=True)
    if Flag_grad:
        print('Analyzing gradients wrt data loss')
        plt.figure(fig_nr)
    layer = 0
    for name, param in net2_u.named_parameters(prefix=''):
        if "weight" in name and layer < nr_layers - 1:
            layer += 1
            if Flag_grad:

                print(f"Layer: {name} | Size: {param.size()}")
                data = param.grad.cpu().detach().numpy()

                (mu, sigma) = stats.norm.fit(data)
                n, bins_data, patches = plt.hist(data, bins)
                y_data = stats.norm.pdf(bins_data, mu, sigma)

                bins_data_list[layer - 1] = bins_data
                y_data_list[layer - 1] = y_data
                # plt.xlim([-0.1, 0.1])

            mean_grad_data_list.append(torch.mean(abs(param.grad)))

    if Flag_grad:

        plt.figure(420, figsize=(12, 12))
        plt.title('epoch' + str(epoch + epoch_pretrain))
        for i in range(1, len(bins_res_list) + 1):
            plt.subplot(3, 3, i)
            # plt.xlim([-2, 2])
            plt.plot(bins_res_list[i - 1], y_res_list[i - 1], 'b--')
            plt.title('layer' + str(i))
            plt.plot(bins_bc_list[i - 1], y_bc_list[i - 1], 'r--')
            plt.plot(bins_data_list[i - 1], y_data_list[i - 1], 'g--')
            plt.suptitle("epoch" + str(epoch))
        plt.legend(['gradL_res', 'gradL_bc', 'gradL_data'])
        plt.savefig(case.path + "/plots/" + case.ID + "LossGrad_" + ex_ID + "_" + str(epoch + epoch_pretrain) + ".png")
        plt.close(420)
        # plt.show()

    if not Flag_notrain:
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
    return

def GradLosses_2(case, device, ex_ID, epoch, epoch_pretrain, x, y, z, xb, yb, zb, xd, yd, zd, ud, vd, wd, net2_u,
               net2_v, net2_w, net2_p, nr_losses, alpha, anneal_weight, anneal_weight2, Flag_grad, Flag_notrain=True):
    # calculating gradients wrt losses, plotting for each layer

    if Flag_notrain:
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
    bins_res_list = [[] for _ in range(nr_layers-1)]
    y_res_list = [[] for _ in range(nr_layers-1)]
    bins_bc_list = [[] for _ in range(nr_layers - 1)]
    y_bc_list = [[] for _ in range(nr_layers - 1)]
    bins_data_list = [[] for _ in range(nr_layers - 1)]
    y_data_list = [[] for _ in range(nr_layers - 1)]
    max_grad_r_list2 = []
    mean_grad_bc_list2 = []
    mean_grad_data_list2 = []


    loss_eqn.backward(retain_graph=True)

    max_grad_r_list2.append(torch.max(abs(net2_u.layer1.weight.grad)))
    max_grad_r_list2.append(torch.max(abs(net2_u.layer2.weight.grad)))
    max_grad_r_list2.append(torch.max(abs(net2_u.layer3.weight.grad)))
    max_grad_r_list2.append(torch.max(abs(net2_u.layer4.weight.grad)))
    max_grad_r_list2.append(torch.max(abs(net2_u.layer5.weight.grad)))
    max_grad_r_list2.append(torch.max(abs(net2_u.layer6.weight.grad)))
    max_grad_r_list2.append(torch.max(abs(net2_u.layer7.weight.grad)))
    max_grad_r_list2.append(torch.max(abs(net2_u.layer8.weight.grad)))
    max_grad_r_list2.append(torch.max(abs(net2_u.layer9.weight.grad)))
    max_grad_r2 = torch.max(torch.tensor(max_grad_r_list2))

    # plt.close(fig_nr)
    loss_bc.backward(retain_graph=True)


    mean_grad_bc_list2.append(torch.mean(abs(net2_u.layer1.weight.grad)))
    mean_grad_bc_list2.append(torch.mean(abs(net2_u.layer2.weight.grad)))
    mean_grad_bc_list2.append(torch.mean(abs(net2_u.layer3.weight.grad)))
    mean_grad_bc_list2.append(torch.mean(abs(net2_u.layer4.weight.grad)))
    mean_grad_bc_list2.append(torch.mean(abs(net2_u.layer5.weight.grad)))
    mean_grad_bc_list2.append(torch.mean(abs(net2_u.layer6.weight.grad)))
    mean_grad_bc_list2.append(torch.mean(abs(net2_u.layer7.weight.grad)))
    mean_grad_bc_list2.append(torch.mean(abs(net2_u.layer8.weight.grad)))
    mean_grad_bc_list2.append(torch.mean(abs(net2_u.layer9.weight.grad)))
    mean_grad_bc2 = torch.mean(torch.tensor(mean_grad_bc_list2))

    # plt.close(fig_nr)
    loss_data.backward(retain_graph=True)


    mean_grad_data_list2.append(torch.mean(abs(net2_u.layer1.weight.grad)))
    mean_grad_data_list2.append(torch.mean(abs(net2_u.layer2.weight.grad)))
    mean_grad_data_list2.append(torch.mean(abs(net2_u.layer3.weight.grad)))
    mean_grad_data_list2.append(torch.mean(abs(net2_u.layer4.weight.grad)))
    mean_grad_data_list2.append(torch.mean(abs(net2_u.layer5.weight.grad)))
    mean_grad_data_list2.append(torch.mean(abs(net2_u.layer6.weight.grad)))
    mean_grad_data_list2.append(torch.mean(abs(net2_u.layer7.weight.grad)))
    mean_grad_data_list2.append(torch.mean(abs(net2_u.layer8.weight.grad)))
    mean_grad_data_list2.append(torch.mean(abs(net2_u.layer9.weight.grad)))
    mean_grad_data2 = torch.mean(torch.tensor(mean_grad_data_list2))



    if not Flag_notrain:
        print('execute annealed learning weight algorithm')

        ##########################################################################
        max_grad_r2 = torch.max(torch.tensor(max_grad_r_list2))
        mean_grad_bc2 = torch.mean(torch.tensor(mean_grad_bc_list2))
        mean_grad_data2 = torch.mean(torch.tensor(mean_grad_data_list2))

        Lambda_BC_adaptive2 = max_grad_r2 / mean_grad_bc2
        Lambda_data_adaptive2 = max_grad_r2 / mean_grad_data2

        # Moving average
        case.Lambda_BC = case.Lambda_BC * (1 - alpha) + Lambda_BC_adaptive2 * alpha
        case.Lambda_data = case.Lambda_data * (1 - alpha) + Lambda_data_adaptive2 * alpha

        anneal_weight2[0].append(case.Lambda_BC)  # boundary loss annealed learning rate
        anneal_weight2[1].append(case.Lambda_data)  # data loss annealed learning rate

        return anneal_weight2, case.Lambda_BC, case.Lambda_data
    return