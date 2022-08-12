import numpy as np
import torch
import vtk
import vtkmodules.util.numpy_support as VN
import tkinter as tk
from tkinter import filedialog
from torch.utils.data import DataLoader, TensorDataset

from scripts.loss_functions import loss_geo_v2, loss_geo_v3
from scripts.network_builder import Net, init_normal, Net2, Net2P, Net3
from scripts.file_readers import data_reader, file_reader, file_reader_v2, data_reader_v2, create_dir
from scripts.figure_creation import Figures

def post_run(case, ex_ID, geometry_locations, networks, means, stds, epoch, run, flag_compare, plot_vecfield=True,
         plot_streamline=True, writeVTK = True):
 

    # if case.input_dimension == 3:
    #     x = x[0::10]
    #     y = y[0::10]
    #     z = z[0::10]
    Figures.velocity_prediction(Figures, case, ex_ID, geometry_locations, networks, epoch, run, batch_idx=1, show=True)
    input_network_geo = torch.cat(([axis for axis in geometry_locations]), 1)
    prediction_values = [network(input_network_geo).view(len(input_network_geo), -1) for network in networks[0:4]]
    
    u, v, w, p = [solution.cpu().data.numpy() for solution in prediction_values]
    
    x, y, *z = [axis.cpu().data.numpy() for axis in geometry_locations]    
    total_loss, loss_c, loss_x, loss_y, loss_c_f, loss_x_f, loss_y_f, *loss_z  = loss_geo_v2(case, prediction_values, geometry_locations, \
                                                                means, stds, flag_vtk=True)
    
    if flag_compare:

        # Reading 2nd file 
        # comp_file = case.directory + "sten_mesh_reduced4_T.vtu"
        prediction_values2 = [network(input_network_geo).view(len(input_network_geo), -1) for network in networks[4:8]]
        u2, v2, w2, p2 = [solution.cpu().data.numpy() for solution in prediction_values2]
        total_loss2, loss_c2, loss_x2, loss_y2, loss_c_f2, loss_x_f2, loss_y_f2, *loss_z2 = loss_geo_v2(case, prediction_values2, geometry_locations, \
                                                                        means, stds, flag_vtk=True)
    # Read mesh as example of the format to output the data to
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(case.mesh_file)
    reader.Update()
    data_vtk = reader.GetOutput()

    
    # Creating VTK
    if flag_compare:
        velocity = np.zeros((len(x), 3))  # Allocating space
        velocity[:, 0] = u[:, 0] - u2[:, 0]
        velocity[:, 1] = v[:, 0] - v2[:, 0]
        if case.input_dimension == 3:
            velocity[:, 2] = w[:, 0] - w2[:, 0]
        velocity_vtk = VN.numpy_to_vtk(velocity)
        velocity_vtk.SetName('Vel_PINN_diff')  # TAWSS vector
        data_vtk.GetPointData().AddArray(velocity_vtk)

        p_vtk = VN.numpy_to_vtk(p - p2)
        p_vtk.SetName('P_PINN_diff')
        data_vtk.GetPointData().AddArray(p_vtk)

        loss_list = [total_loss, loss_c, loss_x, loss_y, loss_z]
        loss_list2 = [total_loss2, loss_c2, loss_x2, loss_y2, loss_z2]
        loss_names = ['loss_total_diff', 'loss_continuity_diff', 'loss_X_diff', 'loss_Y_diff', 'loss_Z_diff'] if z else \
            ['loss_total_diff', 'loss_continuity_diff', 'loss_X_diff', 'loss_Y_diff']
        for idx in range(len(loss_names)): 
            loss = loss_list[idx]
            loss2 = loss_list2[idx]
            loss_vtk = VN.numpy_to_vtk(loss.data.numpy() - loss2.data.numpy())
            loss_vtk.SetName(loss_names[idx])
            data_vtk.GetPointData().AddArray(loss_vtk) 

        velocity = np.zeros((len(x), 3))  # Allocating space
        velocity[:, 0] = np.divide(u[:, 0] , u2[:, 0]) * 100  
        velocity[:, 1] = np.divide(v[:, 0] , v2[:, 0]) * 100 
        if case.input_dimension == 3:
            velocity[:, 2] = np.divide(w[:, 0], w2[:, 0]) * 100 
        velocity_vtk = VN.numpy_to_vtk(velocity)
        velocity_vtk.SetName('Vel_PINN_diff_perc')  # TAWSS vector
        data_vtk.GetPointData().AddArray(velocity_vtk)

    else:
        velocity = np.zeros((len(x), 3))  # Allocating space
        velocity[:, 0] = u[:, 0]
        velocity[:, 1] = v[:, 0]
        if case.input_dimension == 3:
            velocity[:, 2] = w[:, 0]
        velocity_vtk = VN.numpy_to_vtk(velocity)
        velocity_vtk.SetName('Vel_PINN')  # TAWSS vector
        data_vtk.GetPointData().AddArray(velocity_vtk)

        p_vtk = VN.numpy_to_vtk(p)
        p_vtk.SetName('P_PINN')
        data_vtk.GetPointData().AddArray(p_vtk)

        loss_list = [total_loss, loss_c, loss_x, loss_y, loss_z]
        loss_names = ['loss_total', 'loss_continuity', 'loss_X', 'loss_Y', 'loss_Z'] if z else \
            ['loss_total', 'loss_continuity', 'loss_X', 'loss_Y']
        for idx in range(len(loss_names)): 
            loss = loss_list[idx]
            loss_vtk = VN.numpy_to_vtk(loss.data.numpy())
            loss_vtk.SetName(loss_names[idx])
            data_vtk.GetPointData().AddArray(loss_vtk) 


    myoutput = vtk.vtkDataSetWriter()
    myoutput.SetInputData(data_vtk)
    #print(case.path + "/" + str(run) + "/outputs/" + case.ID + "losses" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".vtk")
    myoutput.SetFileName(case.path + "/" + str(run) + "/outputs/" + case.ID + "losses" + ex_ID + "_" + str(epoch) + "_" + str(run) + "_" + "3" + ".vtk")
    myoutput.Write()

    

    print('output file written!')
  
    
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

def write_loss_vtk(case, total_loss, loss_c, loss_x, loss_y, loss_c_f, loss_x_f, loss_y_f, epoch, ex_ID, run):
    
    # Read mesh as example of the format to output the data to
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(case.mesh_file)
    reader.Update()
    data_vtk = reader.GetOutput()

    loss_list = [total_loss, loss_c, loss_x, loss_y, loss_c_f, loss_x_f, loss_y_f]
    loss_names = ['loss_total', 'loss_continuity', 'loss_X', 'loss_Y', 'loss_c_f', 'loss_X_f', 'loss_Y_f']
    for idx in range(len(loss_names)): 
        loss = loss_list[idx]
        loss_vtk = VN.numpy_to_vtk(loss.cpu().data.numpy())
        loss_vtk.SetName(loss_names[idx])
        data_vtk.GetPointData().AddArray(loss_vtk) 

    myoutput = vtk.vtkDataSetWriter()
    myoutput.SetInputData(data_vtk)
    myoutput.SetFileName(case.path + "/" + str(run) + "/outputs/" + case.ID + "losses" + ex_ID + "_" + str(epoch) + str(run) + ".vtk")
    myoutput.Write()

    print('output file written!')



def write_loss_vtk2(case, geometry_locations, networks, means, stds, epoch, ex_ID, run):
    
    #geometry_locations.detach()
    #networks = [network.detach() for network in networks] 
    #means = [mean.detach() for mean in means]
    #stds = [std.detach() for std in stds]

    input_network_geo = torch.cat(([axis for axis in geometry_locations]), 1)
    prediction_values = [network(input_network_geo).view(len(input_network_geo), -1) for network in networks[0:4]]
    
    u, v, w, p = [solution.cpu().data.numpy() for solution in prediction_values]
    
    x, y, *z = [axis.cpu().data.numpy() for axis in geometry_locations]    
    total_loss, loss_c, loss_x, loss_y, loss_c_f, loss_x_f, loss_y_f, *loss_z = loss_geo_v2(case, prediction_values, geometry_locations, \
                                                                means, stds, flag_vtk=True)
    
    
    # Read mesh as example of the format to output the data to

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(case.mesh_file)
    reader.Update()
    data_vtk = reader.GetOutput()

    loss_list = [total_loss, loss_c, loss_x, loss_y, loss_c_f, loss_x_f, loss_y_f]
    loss_names = ['loss_total', 'loss_continuity', 'loss_X', 'loss_Y', 'loss_c_f', 'loss_X_f', 'loss_Y_f']
    for idx in range(len(loss_names)): 
        loss = loss_list[idx]
        loss_vtk = VN.numpy_to_vtk(loss.cpu().data.numpy())
        loss_vtk.SetName(loss_names[idx])
        data_vtk.GetPointData().AddArray(loss_vtk) 

    myoutput = vtk.vtkDataSetWriter()
    myoutput.SetInputData(data_vtk)
    myoutput.SetFileName(case.path + "/" + str(run) + "/outputs/" + case.ID + "losses" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".vtk")
    myoutput.Write()

    print('output file written!')

def post_experiment(case, hidden_layers, neurons, device, seed, nr_runs): 
    
    root = tk.Tk()
    root.withdraw()
    
    net_u = Net(case.input_dimension, hidden_layers, neurons).to(device)
    net_v = Net(case.input_dimension, hidden_layers, neurons).to(device)
    net_p = Net(case.input_dimension, hidden_layers, neurons).to(device) 
    net_w = Net(case.input_dimension, hidden_layers, neurons).to(device)
    
    
    
    geometry_locations, means_geo, stds_geo  = file_reader_v2(case.mesh_file, seed, nr_runs, mesh=True)
    geometry_locations = [torch.Tensor(geometry_locations[axis]).to(device) for axis in range(len(geometry_locations))]
    x, y = [torch.Tensor(geometry_locations[axis]).to(device) for axis in range(len(geometry_locations))]
    # x, y, *z = [axis.requires_grad_() for axis in geometry_locations]
    
    input_network = torch.cat((x, y), 1)

    storage = np.zeros([len(geometry_locations[0]), case.input_dimension + 1, nr_runs])
    # v_storage = np.zeros([len(geometry_locations[0]), nr_runs])
    # p_storage = np.zeros([len(geometry_locations[0]), nr_runs])
    # w_storage = np.zeros([len(geometry_locations[0]), nr_runs])


    for run in range(nr_runs):
        print('Please specify the NN files (.pt) for run {}'.format(run))
        file_paths = filedialog.askopenfilenames(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
        
        
        net_u.load_state_dict(torch.load(file_paths[0]))
        net_v.load_state_dict(torch.load(file_paths[1]))
        net_p.load_state_dict(torch.load(file_paths[2]))
        networks = [net_u, net_v, net_p]
        print('files loaded')

        predictions = [network(input_network).view(len(input_network), -1) for network in networks] 
        storage[:, 0, run] = torch.squeeze(predictions[0]).detach().numpy()  # pressure
        storage[:, 1, run] = torch.squeeze(predictions[1]).detach().numpy()  # velocity u 
        storage[:, 2, run] = torch.squeeze(predictions[2]).detach().numpy()  # velocity v 

    #mean_prediction_storage = np.zeros([len(geometry_locations[0], case.input_dimension + 1)])  
    p, u, v = predictions 
    u = u.data.numpy()
    v = v.data.numpy()
    mean_prediction_storage = np.sum(storage, axis=2, dtype=np.float64) / nr_runs
    
    u_s1 = storage[:, 1, 0]
    v_s1 = storage[:, 2, 0]
    u_s2 = storage[:, 1, 1]
    v_s2 = storage[:, 2, 1]
    u_s = mean_prediction_storage[:, 1]
    v_s = mean_prediction_storage[:, 2]

    
    #Figures.velocity_prediction_post(Figures, case, geometry_locations, u, v)  # predictions latest run - check if storage is done well
    Figures.velocity_prediction_post(Figures, case, geometry_locations, u_s1, v_s1)  # predictions 1st run 
    Figures.velocity_prediction_post(Figures, case, geometry_locations, u_s2, v_s2)  # predictions 2nd run
    Figures.velocity_prediction_post(Figures, case, geometry_locations, u_s, v_s)  # predictions average two runs 
    
    print('test')
    #predictions_storage = [storage[:, variable.detach().numpy(), run] for variable in predictions]

    return
