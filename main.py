# TODO implement the PINN3.0 Code
from matplotlib.figure import Figure
import torch
import torch.optim as optim
import torch.profiler
import torch.utils.data
import matplotlib.pyplot as plt
from sacred import Experiment
import tkinter as tk
from tkinter import filedialog
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys

import os 
path = os.getcwd()
print(path)

from own_scripts import case_info
from own_scripts.file_readers import data_reader, file_reader
from own_scripts.figure_creation import Figures
from own_scripts.network_builder import Net, init_normal
from own_scripts.loss_functions import loss_geo, loss_bnc, loss_data
from own_scripts.weight_annealing import weight_annealing_algorithm

### naming conventions 
# Function - my_function
# Variable - my_variable
# Class - MyClass
# constant - MY_CONSTANT
# module - my_module.py
# package - mypackage

# If line breaking needs to occur around binary operators, like + and *, it should occur before the operator
# PEP 8 recommends that you always use 4 consecutive spaces to indicate indentation.
# Line up the closing brace with the first non-whitespace character of the previous line 
# Surround docstrings with three double quotes on either side, as in """This is a docstring"""
# - Write them for all public modules, functions, classes, and methods.
# - Put the """ that ends a multiline docstring on a line by itself
# Use .startswith() and .endswith() instead of slicing

ex = Experiment()

# Configuration of the case data, reading of the data files, hyperparameter set up

@ex.config
def config():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        sys.exit('unable to mount to CUDA, please retry. Make sure CUDA is installed') 


    case = case_info.stenose2D_Arzani()
    ex_ID = '002'
    output_filename = "./results/" + case.name + "/outputs/" + case.ID + "PINNoutput_" + ex_ID + ".vtk"
    DEBUG = True
    hidden_layers = 5
    neurons = 128
    
    geometry_locations, minima, maxima  = file_reader(case.mesh_file, mesh=True)
    boundary_locations, _, _ = file_reader(case.bc_file, mesh=False, min=minima, max=maxima)
    solution_locations, solution_values = data_reader(case, min=minima, max=maxima)
    
    # Plot to check the scaling of the geometry and data points
    Figures.scaled_geometry(Figures, case, ex_ID, boundary_locations, solution_locations, solution_values)

    # if not DEBUG:
    #     AppendObserver(case, ex)


    Flag_pretrain = False  # Whether you want to load a pretrained network and train further
    Flag_notrain = False  # Whether you want to load a pretrained network and look at postprocessing
    Flag_lossTerm = True
    epochs = 5500
    epoch_pretrain = 0 if not Flag_pretrain else 5500  # epochs after reading a pretrained network
    epoch_save_loss = 2  # determines how often the loss values are saved
    epoch_save_NN = 100  # determines how often the NN is updated -- 'fail-save' mechanism should training stop before
    epoch_learning_weight = 10
    # final epoch
    remark = "First Arzani run with the PINN3.0 framework"  # note for future reference

    learning_rate = 5e-4  # starting learning rate
    step_epoch = 1200  # after how many iterations the learning rate should be adapted
    decay_rate = 0.1  # 0.1
    alpha = 0.9  # momentum coefficient for learning rate annealing
    batchsize = 256  # common; 2D : 256, 3D : 512
    nr_losses = 3

@ex.main
def geo_train(device, case, epochs, epoch_save_loss, epoch_save_NN, geometry_locations, boundary_locations, solution_locations,
 solution_values, batchsize, nr_losses, learning_rate, Flag_notrain, Flag_pretrain,
              step_epoch, decay_rate, ex_ID, output_filename, epoch_pretrain, alpha, Flag_lossTerm, hidden_layers, neurons, 
              epoch_learning_weight):
    
    # Build the NNs
    net_u = Net(case.input_dimension, hidden_layers, neurons).to(device)
    net_v = Net(case.input_dimension, hidden_layers, neurons).to(device)
    net_p = Net(case.input_dimension, hidden_layers, neurons).to(device) 
    net_w = Net(case.input_dimension, hidden_layers, neurons).to(device)

    networks = [net_u, net_v, net_w, net_p] 
    
    # Reading the pretrained networks if specified 
    if Flag_notrain or Flag_pretrain:
        print('Reading (pretrain) functions first...')
        root = tk.Tk()
        root.withdraw()

        print('Please specify the NN file (.pt) for the velocity u')
        file_path_u = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
        # file_path_u = 'D:/Graduation_project/AA_files/2DSTEN/004/DataReader_adapted/STEN2DTB_data_u.pt'
        print('Please specify the NN file (.pt) for the velocity v')
        file_path_v = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
        # file_path_v = 'D:/Graduation_project/AA_files/2DSTEN/004/DataReader_adapted/STEN2DTB_data_v.pt'
        print('Please specify the NN file (.pt) for the velocity p')
        file_path_p = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
        # file_path_p = 'D:/Graduation_project/AA_files/2DSTEN/004/DataReader_adapted/STEN2DTB_data_p.pt'

        if case.input_dimension == 3:
            print('Please specify the NN file (.pt) for the velocity w')
            file_path_w = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
            net_w.load_state_dict(torch.load(file_path_w))
        # root.destroy()

        net_u.load_state_dict(torch.load(file_path_u))
        net_v.load_state_dict(torch.load(file_path_v))
        net_p.load_state_dict(torch.load(file_path_p))

    # if Flag_notrain:
    #     post(case, epochs, net_u, net_v, net_w, net_p, output_filename, ex_ID, plot_vecfield=False, plot_streamline=False)

        # calculating gradients wrt losses, plotting for each layer
        # TODO : Zorgen dat dit weer werkt voor de no_train flag
        # epoch = 0
        # loss_weight_list = []
        # GradLosses(case, device, ex_ID, epoch, epoch_pretrain, x, y, z, xb, yb, zb, xd, yd, zd, ud, vd, wd, net_u,
        #            net_v, net_w, net_p, nr_losses, alpha, loss_weight_list, Flag_grad=True, Flag_notrain=True)

        # sys.exit('postprocessing executed')
    print('device used is', device)

    # if case.input_dimension == 3:  # TODO data size afhankelijk maken?
    #     x_plot = x_plot[::4]
    #     y_plot = y_plot[::4]
    #     z_plot = z_plot[::4]

    # Build arrays as Tensors and move to device, require_grad = True 
    x_plot, y_plot, *z_plot = [torch.Tensor(geometry_locations[axis]).to(device) for axis in range(len(geometry_locations))]
    geometry_locations = [torch.Tensor(geometry_locations[axis]).to(device) for axis in range(len(geometry_locations))]
    x, y, *z = [axis.requires_grad_() for axis in geometry_locations]
    
    xb_plot, yb_plot, *zb_plot = [torch.Tensor(boundary_locations[axis]).to(device) for axis in range(len(boundary_locations))]
    boundary_locations = [torch.Tensor(boundary_locations[axis]).to(device) for axis in range(len(boundary_locations))]
    # xb, yb, *zb = [axis.requires_grad_() for axis in boundary_locations]
    
    xd_plot, yd_plot, *zd_plot = [torch.Tensor(solution_locations[axis]).to(device) for axis in range(len(solution_locations))]
    solution_locations = [torch.Tensor(solution_locations[axis]).to(device) for axis in range(len(solution_locations))]
    # xd, yd, *zd = [axis.requires_grad_() for axis in solution_locations]
    
    ud_plot, vd_plot, *wd_plot = [torch.Tensor(solution_values[axis]).to(device) for axis in range(len(solution_values))]
    solution_values = [torch.Tensor(solution_values[axis]).to(device) for axis in range(len(solution_values))]
    # ud, vd, *wd = [axis.requires_grad_() for axis in solution_values]
    
    if not z:  # Freeze network if 2D for reduced computation time 
        for param in net_w.parameters():
            param.requires_grad = False  

    dataset = TensorDataset(x, y, *z) 
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=True)

    nr_layers = hidden_layers + 3
    loss_list = [[] for _ in range(nr_losses)]
    loss_weight_list = [[1.] for _ in range(nr_losses-1)]
    NSgrads_list = []
    NSterms_list = []
   
    tic = time.time()

    if not Flag_pretrain:
        net_u.apply(init_normal)  # TODO init Xavier?
        net_v.apply(init_normal)
        net_w.apply(init_normal)
        net_p.apply(init_normal)

    # Optimizer determines algorithm used to calculate update of NN parameters. 
    optimizer_u = optim.Adam(net_u.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=10 ** -15)
    optimizer_v = optim.Adam(net_v.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=10 ** -15)
    optimizer_w = optim.Adam(net_w.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=10 ** -15)
    optimizer_p = optim.Adam(net_p.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=10 ** -15)

    # Scheduler determines the algorithm to update the learning rate during training. 
    scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
    scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
    scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
    scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)

    grad_list = [0, 10, 20, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]  # Epochs calculation gradient distributions
    lambda_bc = case.lambda_bc
    lambda_data = case.lambda_data
    for epoch in range(epochs):  # TODO epoch + epoch_pretrain
        flag_grad_plot = False
        # flag_grad_plot = True if epoch in grad_list else flag_grad_plot
        
        eqn_loss_tot = 0.
        bnc_loss_tot = 0.
        data_loss_tot = 0.
        n = 0
        
        for batch_idx, (x_in, y_in, *z_in) in enumerate(dataloader):  
            # networks = [network.zero_grad() for network in networks]
            net_u.zero_grad()
            net_v.zero_grad()
            net_w.zero_grad()
            net_p.zero_grad()

            batch_locations = [x_in, y_in, z_in] if z_in else [x_in, y_in]  # Differentiate between 2D and 3D 
            
            # Create input data: Coordinates within domain (geo), at boundary (bnc) and sensor points (data)
            input_dimensionetwork_geo = torch.cat(([axis for axis in batch_locations]), 1)
            input_dimensionetwork_bnc = torch.cat(([axis for axis in boundary_locations]), 1)
            input_dimensionetwork_data = torch.cat(([axis for axis in solution_locations]), 1)

            # Current predictions of the PINN for velocity and pressure
            predictions_geo = [network(input_dimensionetwork_geo).view(len(input_dimensionetwork_geo), -1) for network in networks]   
            predictions_bnc = [network(input_dimensionetwork_bnc).view(len(input_dimensionetwork_bnc), -1) for network in networks]  
            predictions_data = [network(input_dimensionetwork_data).view(len(input_dimensionetwork_data), -1) for network in networks]  

            eqn_loss, grads = loss_geo(case, predictions_geo, batch_locations)  # Compare prediction to Navier-Stokes eq. TODO: grads houden?
            bnc_loss = loss_bnc(case, predictions_bnc)  # Compare prediction to no-slip condition
            data_loss = loss_data(case, predictions_data, solution_values)  # Compare prediction to data solution

            if epoch % epoch_learning_weight == 0 and len(loss_weight_list[0]) == epoch/epoch_learning_weight:
                loss_weight_list, lambda_bc, lambda_data, *gradients = weight_annealing_algorithm(networks, eqn_loss, bnc_loss, 
                                                                                data_loss, lambda_bc, lambda_data, 
                                                                                loss_weight_list, nr_layers, alpha, flag_grad_plot)
                if flag_grad_plot: 
                    Figures.gradient_distribution(Figures, case, gradients, epoch+epoch_pretrain, ex_ID, nr_layers)

            loss = eqn_loss + lambda_bc * bnc_loss + lambda_data * data_loss
            eqn_loss_tot += eqn_loss.detach() 
            bnc_loss_tot += bnc_loss.detach()  # Detach to prevent storage of computation graph which leads to Out of Memory error.     
            data_loss_tot += data_loss.detach()
            n += 1

            # Optimizer to determine direction of weight and bias update 
            loss.backward()  # Compute the gradients 
            optimizer_u.step()  # Update parameters 
            optimizer_v.step()
            optimizer_p.step()
            if case.input_dimension == 3:
                optimizer_w.step()

            # End of batch - Verbose: 
            if batch_idx % 40 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} Loss BC {:.8f} Loss data {:.8f}'.format(
                    epoch, batch_idx * len(x_in), len(dataloader.dataset),
                           100. * batch_idx / len(dataloader), eqn_loss.item(), bnc_loss.item(), data_loss.item()))


        NSgrads_list.append(grads)
        # NSterms_list.append(loss_terms)

        # Scheduler follows learning rate adapting scheme as specified at beginning 
        scheduler_u.step()
        scheduler_v.step()
        scheduler_p.step()
        if case.input_dimension == 3:
            scheduler_w.step()
        
        eqn_loss_tot /= n
        bnc_loss_tot /= n
        data_loss_tot /= n
        
        # Store loss data every epoch_save_loss iteration, add to MongoDB server if observer is added (DEBUG=False)
        if epoch % epoch_save_loss == 0:
            loss_list[0].append(eqn_loss_tot.item())
            loss_list[1].append(bnc_loss_tot.item())
            loss_list[2].append(data_loss_tot.item())
            if ex.observers:  # check whether an observer is added to the run
                ex.log_scalar("training.eqn_loss", eqn_loss_tot.item(), epoch)
                ex.log_scalar("training.bnc_loss", bnc_loss_tot.item(), epoch)
                ex.log_scalar("training.data_loss", data_loss_tot.item(), epoch)

        # # Save NNs and PINN performance plots every epoch_save_NN
        if epoch % epoch_save_NN == 0:  
            torch.save(net_u.state_dict(), case.path + "/NNfiles/" + case.ID + "data_u_" + ex_ID + ".pt")
            torch.save(net_v.state_dict(), case.path + "/NNfiles/" + case.ID + "data_v_" + ex_ID + ".pt")
            torch.save(net_p.state_dict(), case.path + "/NNfiles/" + case.ID + "data_p_" + ex_ID + ".pt")
            if case.input_dimension == 3:
                torch.save(net_w.state_dict(), case.path + "/NNfiles/" + case.ID + "data_w_" + ex_ID + ".pt")
            
        #     print('NN saved')

        #     steps = np.linspace(epoch_pretrain, epoch_pretrain + epoch, len(loss_weight_list[0]))
        #     plt.figure(9)
        #     for i in range(nr_losses - 1):
        #         plt.plot(steps, loss_weight_list[i])
        #     plt.xlabel('epochs')
        #     # plt.ylabel('Loss')
        #     # plt.yscale('log')
        #     plt.legend(['lambda_bc', 'lambda_data'])
        #     plt.title(case.name)
        #     # plt.show()
        #     plt.savefig(case.path + "/plots/" + case.ID + "learningWeight_plot_" + ex_ID + ".png")
        #     plt.close(9)

        #     # End of epoch - Verbose 
        #     print('*****Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f} Loss data {:.10f} ****'.format(eqn_loss_tot,
        #                                                                                              bnc_loss_tot,
        #                                                                                              data_loss_tot))
        #     print('learning rate is ', optimizer_u.param_groups[0]['lr'], optimizer_v.param_groups[0]['lr'])


            steps = np.linspace(epoch_pretrain, epoch_pretrain+epoch, len(loss_list[0]))
            colors = ['b', 'g', 'r']
            plt.figure(5)
            for i in range(nr_losses):
                plt.plot(steps, loss_list[i], colors[i])
            plt.xlabel('epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend(['eq', 'bc', 'data'])
            plt.title(case.name)
            # plt.show()
            plt.savefig(case.path + "/plots/" + case.ID + "loss_plot_" + ex_ID + ".png")
            plt.close(5)

            net_in = torch.cat((x_plot.requires_grad_(), y_plot.requires_grad_(), z_plot.requires_grad_()),
                                1) if case.input_dimension == 3 else \
                torch.cat((x_plot.requires_grad_(), y_plot.requires_grad_()), 1)
            # net_in = torch.cat((x.requires_grad_(), y.requires_grad_()), 1)
            output_u = net_u(net_in)  # evaluate model
            output_v = net_v(net_in)  # evaluate model
            output_u = output_u.cpu().data.numpy()  # need to convert to cpu before converting to numpy
            output_v = output_v.cpu().data.numpy()

            xd, yd, *zd = solution_locations

            net_in_data = torch.cat((xd.requires_grad_(), yd.requires_grad_(), zd.requires_grad_()), 1) \
                if case.input_dimension ==3 else torch.cat((xd.requires_grad_(), yd.requires_grad_()), 1)
            output_ud = net_u(net_in_data)
            output_vd = net_v(net_in_data)
            output_ud = output_ud.cpu().data.numpy()  # need to convert to cpu before converting to numpy
            output_vd = output_vd.cpu().data.numpy()

            if case.input_dimension == 3:
                output_w = net_w(net_in)
                output_w = output_w.cpu().data.numpy()
                output_wd = net_w(net_in_data)
                output_wd = output_wd.cpu().data.numpy()

            x_plot2 = x_plot.cpu()
            y_plot2 = y_plot.cpu()
            # z_plot2 = z_plot.cpu()
            plt.figure(6)
            plt.subplot(2, 1, 1)
            plt.scatter(x_plot2.detach().numpy(), y_plot2.detach().numpy(), c=output_u, cmap='rainbow')
            plt.title('NN results, u (top) & v (bot), - epoch' + str(epoch_pretrain + epoch))
            plt.colorbar()
            plt.subplot(2, 1, 2)
            plt.scatter(x_plot2.detach().numpy(), y_plot2.detach().numpy(), c=output_v, cmap='rainbow')
            plt.colorbar()
            plt.savefig(case.path + "/plots/" + case.ID + "plotWU_" + ex_ID + "_" + str(epoch + epoch_pretrain)) if case.input_dimension == 3 \
                else plt.savefig(case.path + "/plots/" + case.ID + "plotUV_" + ex_ID + "_" + str(epoch + epoch_pretrain))
            plt.close(6)
            # plt.show()

            xd_plot2 = xd_plot.cpu()
            yd_plot2 = yd_plot.cpu()
            # zd_plot2 = zd_plot.cpu()
            xb_plot2 = xb_plot.cpu()
            yb_plot2 = yb_plot.cpu()
            # zb_plot2 = zb_plot.cpu()
            ud_plot2 = ud_plot.cpu().data.numpy()
            vd_plot2 = vd_plot.cpu().data.numpy()
            ud_diff = (output_ud - ud_plot2) / ud_plot2 * 100
            vd_diff = (output_vd - vd_plot2) / vd_plot2 * 100
            if case.input_dimension == 3:
                wd_plot2 = wd_plot.cpu().data.numpy()
                wd_diff = (output_wd - wd_plot2) / wd_plot2 * 100
            # ud_diff_geo = (output_u / ud_geo)
            # vd_diff_geo = (output_v / vd_geo)

        # plt.figure(7)
        # plt.subplot(2, 1, 1)
        # plt.scatter(xb_plot2.detach().numpy(), yb_plot2.detach().numpy())
        # plt.scatter(xd_plot2.detach().numpy(), yd_plot2.detach().numpy(), c=ud_diff, cmap='rainbow')
        # plt.title('% error; ud (top) & vd (bot), - epoch' + str(epoch_pretrain + epoch))
        # plt.colorbar()
        # plt.subplot(2, 1, 2)
        # plt.scatter(xb_plot2.detach().numpy(), yb_plot2.detach().numpy())
        # plt.scatter(xd_plot2.detach().numpy(), yd_plot2.detach().numpy(), c=vd_diff, cmap='rainbow')
        # plt.colorbar()
        # plt.savefig(case.path + "/plots/" + case.ID + "plotD_UV_" + ex_ID + "_" + str(epoch + epoch_pretrain))
        # plt.close(7)
        # plt.show()

        # plt.figure(8)
        # plt.subplot(2, 1, 1)
        # plt.scatter(xb_plot2.detach().numpy(), yb_plot2.detach().numpy())
        # plt.scatter(x_plot2.detach().numpy(), y_plot2.detach().numpy(), c=ud_diff_geo, cmap='rainbow')
        # plt.title('% error; ud (top) & vd (bot), - epoch' + str(epoch_pretrain + epoch))
        # plt.colorbar()
        # plt.subplot(2, 1, 2)
        # plt.scatter(xb_plot2.detach().numpy(), yb_plot2.detach().numpy())
        # plt.scatter(x_plot2.detach().numpy(), y_plot2.detach().numpy(), c=vd_diff_geo, cmap='rainbow')
        # plt.colorbar()
        # plt.savefig(case.path + "/plots/" + case.ID + "plotD_UV_geo_" + ex_ID + "_" + str(epoch + epoch_pretrain))
        # plt.close(8)
        # # plt.show()


            steps = np.linspace(epoch_pretrain, epoch_pretrain + epoch, len(loss_weight_list[0]))
            plt.figure(9)
            for i in range(nr_losses - 1):
                plt.plot(steps, loss_weight_list[i])
            plt.xlabel('epochs')
            # plt.ylabel('Loss')
            # plt.yscale('log')
            plt.legend(['lambda_bc', 'lambda_data'])
            plt.title(case.name)
            # plt.show()
            plt.savefig(case.path + "/plots/" + case.ID + "learningWeight_plot_" + ex_ID + ".png")
            plt.close(9)

    toc = time.time()
    elapseTime = toc - tic
    print("elapse time in parallel = ", elapseTime)

    # Save the NNs 
    torch.save(net_u.state_dict(), case.path + "/NNfiles/" + case.ID + "data_u_" + ex_ID + ".pt")
    torch.save(net_v.state_dict(), case.path + "/NNfiles/" + case.ID + "data_v_" + ex_ID + ".pt")
    torch.save(net_p.state_dict(), case.path + "/NNfiles/" + case.ID + "data_p_" + ex_ID + ".pt")
    if case.input_dimension == 3:
        torch.save(net_w.state_dict(), case.path + "/NNfiles/" + case.ID + "data_w_" + ex_ID + ".pt")

    print("Data saved!")
    epochs = epoch + epoch_pretrain  

    # # steps = np.linspace(0, epoch - epoch_save_loss.np.int(np.divide(epoch, epoch_save_loss)))
    # steps = np.linspace(epoch_pretrain, epoch_pretrain+epochs - epoch_save_loss, np.int(np.divide(epochs, epoch_save_loss)))
    steps = np.linspace(epoch_pretrain, epoch_pretrain+epochs - epoch_save_loss, len(loss_list[0]))
    colors = ['b', 'g', 'r']
    plt.figure(10)
    for i in range(nr_losses):
        plt.plot(steps, loss_list[i], colors[i])
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(['eq', 'bc', 'data'])
    plt.title(case.name)
    # plt.show()
    plt.savefig(case.path + "/plots/" + case.ID + "loss_plot_" + ex_ID + ".png")


    print('jobs done!')
ex.run()





