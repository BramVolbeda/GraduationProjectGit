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
from sacred.observers import MongoObserver
import os 
path = os.getcwd()
print(path)

from scripts import case_info
from scripts.file_readers import data_reader, file_reader, file_reader_v2, data_reader_v2, create_dir
from scripts.figure_creation import Figures
from scripts.network_builder import Net, init_normal, Net2, Net2P, Net3
from scripts.loss_functions import loss_geo, loss_bnc, loss_data, loss_geo_v2, loss_geo_v3, loss_wss, evo
from scripts.weight_annealing import weight_annealing_algorithm
from scripts.post_processing import post_run, write_loss_vtk, write_loss_vtk2, post_experiment

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
case = case_info.stenose2D()
DEBUG = True
if not DEBUG: 
    ex.observers.append(MongoObserver(url='mongodb+srv://dbBram:t42F7C828K!!@experiments.aakys.mongodb.net'
                                              '/test_stenose?ssl=true&ssl_cert_reqs=CERT_NONE',
                                          db_name=case.ID))

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

    ex_ID = '009'
    DEBUG = True
    hidden_layers = 5
    neurons = 128
    nr_data_points = 5  # amount of data points if flag_lhs or flag_random is set to True in the file_readers
    nr_runs = 1  # divides the collocation points with replacement
    
    # if not DEBUG:
    #     AppendObserver(case, ex)

    flag_pretrain = False  # Whether you want to load a pretrained network and train further
    flag_notrain = True  # Whether you want to load a pretrained network and look at postprocessing
    flag_compare = False  # if flag_notrain = True, setting this one to true allows for the comparison between 2 runs
    flag_bagging = False  # Currently nothing
    
    epochs = 5500
    epoch_pretrain = 0 if not flag_pretrain else 5500  # epochs after reading a pretrained network
    epoch_save_loss = 2  # determines how often the loss values are saved
    epoch_save_NN = 5  # determines how often the NN is updated -- 'fail-save' mechanism should training stop before
    epoch_learning_weight = 10
    # final epoch
    remark = "2DTUBE frequent logging MSE loss"  # note for future reference

    learning_rate = 5e-4  # starting learning rate
    step_epoch = 1200  # after how many iterations the learning rate should be adapted
    decay_rate = 0.1  # 0.1
    alpha = 0.9  # momentum coefficient for learning rate annealing
    batchsize = 256  # common; 2D : 256, 3D : 512
    nr_losses = 3

@ex.main
def geo_train(device, case, epochs, epoch_save_loss, epoch_save_NN, batchsize, nr_losses, learning_rate, flag_notrain, 
                flag_pretrain, flag_compare, flag_bagging, step_epoch, decay_rate, ex_ID, epoch_pretrain, alpha, 
                hidden_layers, neurons, epoch_learning_weight, nr_data_points, nr_runs):
    
    # if flag_notrain:  # Stelt in staat meerdere runs te vergelijken 
    #     device = ('cpu')
    #     seed = 0  # TODO optional maken 
    #     post_experiment(case, hidden_layers, neurons, device, seed, nr_runs)
        
    if flag_notrain: 
        device = 'cpu'

    # Build the NNs
    # net_u = Net(case.input_dimension, hidden_layers, neurons).to(device)
    # net_v = Net(case.input_dimension, hidden_layers, neurons).to(device)
    # net_p = Net(case.input_dimension, hidden_layers, neurons).to(device) 
    # net_w = Net(case.input_dimension, hidden_layers, neurons).to(device)

    net_u = Net2(2, 128).to(device)
    net_v = Net2(2, 128).to(device)
    net_p = Net2(2, 128).to(device) 
    net_w = Net2(2, 128).to(device)

    networks = [net_u, net_v, net_w, net_p] 
    
    if flag_compare:
        net_u2 = Net(2, hidden_layers, neurons).to(device)
        net_v2 = Net(2, hidden_layers, neurons).to(device)
        net_p2 = Net(2, hidden_layers, neurons).to(device) 
        net_w2 = Net(2, hidden_layers, neurons).to(device)

        networks = [net_u, net_v, net_w, net_p, net_u2, net_v2, net_w2, net_p2]
            
    # Reading the pretrained networks if specified 
    if flag_notrain or flag_pretrain:
        print('Reading (pretrain) functions first...')
        root = tk.Tk()
        root.withdraw()

        print('Please specify the NN file (.pt) for the velocity u')
        file_path_u = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
        #file_path_u = 'D:/Graduation_project/code_v2/results/stenose2DA/NNfiles/STEN2DA_data_u_002.pt'
        print('Please specify the NN file (.pt) for the velocity v')
        file_path_v = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
        #file_path_v = 'D:/Graduation_project/code_v2/results/stenose2DA/NNfiles/STEN2DA_data_v_002.pt'
        print('Please specify the NN file (.pt) for the velocity p')
        file_path_p = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
        #file_path_p = 'D:/Graduation_project/code_v2/results/stenose2DA/NNfiles/STEN2DA_data_p_002.pt'

        if case.input_dimension == 3:
            print('Please specify the NN file (.pt) for the velocity w')
            file_path_w = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
            net_w.load_state_dict(torch.load(file_path_w))
        # root.destroy()

        net_u.load_state_dict(torch.load(file_path_u))
        net_v.load_state_dict(torch.load(file_path_v))
        net_p.load_state_dict(torch.load(file_path_p))

        if flag_compare:
            print('Please specify the NN file (.pt) for the velocity u2')
            file_path_u2 = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
            print('Please specify the NN file (.pt) for the velocity v2')
            file_path_v2 = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
            print('Please specify the NN file (.pt) for the velocity p2')
            file_path_p2 = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
            
            net_u2.load_state_dict(torch.load(file_path_u2))
            net_v2.load_state_dict(torch.load(file_path_v2))
            net_p2.load_state_dict(torch.load(file_path_p2)) 

        # calculating gradients wrt losses, plotting for each layer
        # TODO : Zorgen dat dit weer werkt voor de no_train flag
        # epoch = 0
        # loss_weight_list = []
        # GradLosses(case, device, ex_ID, epoch, epoch_pretrain, x, y, z, xb, yb, zb, xd, yd, zd, ud, vd, wd, net_u,
        #            net_v, net_w, net_p, nr_losses, alpha, loss_weight_list, flag_grad=True, flag_notrain=True)

        # sys.exit('postprocessing executed'
        
    for run in range(1, nr_runs + 1): 
        
        create_dir(case, run)
        seed = run
        torch.manual_seed(seed)
        np.random.seed(seed)
        geometry_locations, means_geo, stds_geo  = file_reader_v2(case.mesh_file, seed, nr_runs, mesh=True, flag_lhs=False)
        # Take half the points for 3D 
        #geometry_locations = [geometry_locations[axis][::20] for axis in range(len(geometry_locations))]

        boundary_locations, means_bnc, stds_bnc = file_reader_v2(case.bc_file, seed, nr_runs, mesh=False, flag_lhs=False)
        solution_locations, solution_values = data_reader_v2(case, geometry_locations, nr_data_points, seed,\
                                                                flag_random=False, flag_exact=False, flag_lhs=False)

        # Figures.scaled_geometry(Figures, case, ex_ID, boundary_locations, geometry_locations, geometry_locations, run, show=True)

        # Build arrays as Tensors and move to device, require_grad = True 
        geometry_locations = [torch.Tensor(geometry_locations[axis]).to(device) for axis in range(len(geometry_locations))]
        x, y, *z = [axis.requires_grad_() for axis in geometry_locations]  # TODO uitzoeken hoe zelfde vorm als x en y 
         # Plot to check the scaling of the geometry and data points
        Figures.scaled_geometry(Figures, case, ex_ID, boundary_locations, solution_locations, solution_values, run, show=False)
        if z: 
            Figures.scatter3D_data(Figures, case, solution_locations, solution_values, show=False)                                                                    #te krijgen terwijl optioneel te houden
        
        boundary_locations = [torch.Tensor(boundary_locations[axis]).to(device) for axis in range(len(boundary_locations))]
        solution_locations = [torch.Tensor(solution_locations[axis]).to(device) for axis in range(len(solution_locations))]
        solution_values = [torch.Tensor(solution_values[axis]).to(device) for axis in range(len(solution_values))]

        if flag_notrain:
            post_run(case, ex_ID, geometry_locations, networks, means_geo, stds_geo, epoch_pretrain, run, flag_compare)
            sys.exit('postprocessing executed')

        if not z:  # Freeze network if 2D for reduced computation time 
            print('Freezing training of NN_W - No Z axis detected')
            for param in net_w.parameters():
                param.requires_grad = False  
            dataset = TensorDataset(x, y)
        else: 
            z = z[0]
            dataset = TensorDataset(x, y, z)
            
        #dataset = TensorDataset(x, y, z) if z_ else TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=True)

        nr_layers = hidden_layers + 2  # TODO: +3 ? 
        loss_list = [[] for _ in range(nr_losses)]
        loss_weight_list = [[1.] for _ in range(nr_losses-1)]
        ns_grads_list = []
        ns_values_list = []
    
        tic = time.time()

        if not flag_pretrain:
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
        # scheduler_u = torch.optim.lr_scheduler.MultiStepLR(optimizer_u, milestones=[1200, 2400, 3600, 4800], gamma = decay_rate) 
        # scheduler_v = torch.optim.lr_scheduler.MultiStepLR(optimizer_v, milestones=[1200, 2400, 3600, 4800], gamma = decay_rate) 
        # scheduler_w = torch.optim.lr_scheduler.MultiStepLR(optimizer_w, milestones=[1200, 2400, 3600, 4800], gamma = decay_rate) 
        # scheduler_p = torch.optim.lr_scheduler.MultiStepLR(optimizer_p, milestones=[1200, 2400, 3600, 4800], gamma = decay_rate) 
        scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
        scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
        scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
        scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)

        grad_list = [0, 10, 20, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]  # Epochs calculation gradient distributions
        vtk_list = [0, 5, 10, 25, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000, 5000]
        
        batch_evo_0 = [1, 2, 3, 4, 5, 10, 25, 50, 75]
        batch_evo = [20, 40, 60]
        
        lambda_bc = case.lambda_bc
        lambda_data = case.lambda_data

        # batch_retain = [torch.tensor([0, 0])]
        for epoch in range(epoch_pretrain, epochs):  # TODO epoch + epoch_pretrain
            flag_grad_plot = False
            flag_grad_plot = True if epoch in grad_list else flag_grad_plot
            
            eqn_loss_tot = 0.
            bnc_loss_tot = 0.
            data_loss_tot = 0.
            n = 0
            
            batch_len = []
            # if epoch in vtk_list: 
            #     write_loss_vtk2(case, geometry_locations, networks, means_geo, stds_geo, epoch, ex_ID, run)

            for batch_idx, batch_locations in enumerate(dataloader):  
                
                # if batch_idx != 0: 

                # #     # plt.figure()  # Check which points are being replaced
                # #     # xplot = batch_locations[0][0:len(batch_retain[0]), :].cpu().detach().numpy()
                # #     # yplot = batch_locations[1][0:len(batch_retain[0]), :].cpu().detach().numpy()
                # #     # plt.scatter(xplot, yplot)
                # #     # plt.show()
                    
                #     batch_locations[0][0:len(batch_retain[0]), :] = batch_retain[0]
                #     batch_locations[1][0:len(batch_retain[0]), :] = batch_retain[1]
                #     batch_len.append(len(batch_retain[0]))

                net_u.zero_grad()
                net_v.zero_grad()
                net_w.zero_grad()
                net_p.zero_grad()

                # Create input data: Coordinates within domain (geo), at boundary (bnc) and sensor points (data)
                input_network_geo = torch.cat(([axis for axis in batch_locations]), 1)
                input_network_bnc = torch.cat(([axis for axis in boundary_locations]), 1)
                input_network_data = torch.cat(([axis for axis in solution_locations]), 1)

                # Current predictions of the PINN for velocity and pressure
                predictions_geo = [network(input_network_geo).view(len(input_network_geo), -1) for network in networks]   
                predictions_bnc = [network(input_network_bnc).view(len(input_network_bnc), -1) for network in networks]  
                predictions_data = [network(input_network_data).view(len(input_network_data), -1) for network in networks]  

                # eqn_loss, grads, ns_values_epoch = loss_geo_v2(case, predictions_geo, batch_locations, means_geo, stds_geo, flag_values=True)  # Compare prediction to Navier-Stokes eq. TODO: grads houden?
                eqn_loss, grads, ns_values_epoch, total_loss, loss_x, loss_y, loss_c = loss_geo_v2(case, predictions_geo, batch_locations, \
                                                            means_geo, stds_geo, flag_values=True) 
                                                             # Compare prediction to Navier-Stokes eq. TODO: verschillende returns obv flag
                                                             # loss_x en loss_y vangen in principe total_loss
                bnc_loss = loss_bnc(case, predictions_bnc)  # Compare prediction to no-slip condition
                data_loss = loss_data(case, predictions_data, solution_values)  # Compare prediction to data solution
                
                # EVO algorithm - lim parameter tunable
                #batch_retain, total_loss_retain, loss_x_retain, loss_y_retain, loss_c_retain = evo(total_loss, loss_x, loss_y, loss_c, batch_locations, lim=0.6)
                
                # if epoch == 0: 
                #     if batch_idx in batch_evo_0: 
                #         Figures.evo_plot(Figures, case, run, ex_ID, epoch, batch_idx, boundary_locations, batch_locations, batch_retain, total_loss_retain, loss_x_retain, loss_y_retain, loss_c_retain, batch_retain_old)
                #         Figures.velocity_prediction(Figures, case, ex_ID, geometry_locations, networks, epoch, run, batch_idx)
                # elif epoch in grad_list:
                #     if batch_idx in batch_evo: 
                #         Figures.evo_plot(Figures, case, run, ex_ID, epoch, batch_idx, boundary_locations, batch_locations, batch_retain, total_loss_retain, loss_x_retain, loss_y_retain, loss_c_retain, batch_retain_old)
                #         Figures.velocity_prediction(Figures, case, ex_ID, geometry_locations, networks, epoch, run, batch_idx)
                # batch_retain_old = batch_retain    

                # if epoch % epoch_save_NN == 0: 
                #     if batch_idx == 0:
                #         loss_wss(case, predictions_bnc, networks, xb, yb, *zb)
                #         Figures.gradient_losses(Figures, case, ex_ID, epoch, epoch_pretrain, nr_layers, eqn_loss, bnc_loss, data_loss, networks)
            

                if epoch % epoch_learning_weight == 0 and len(loss_weight_list[0]) == epoch/epoch_learning_weight:
                    loss_weight_list, lambda_bc, lambda_data, *gradients = weight_annealing_algorithm(networks, eqn_loss, bnc_loss, 
                                                                                    data_loss, lambda_bc, lambda_data, 
                                                                                    loss_weight_list, alpha)
                    ns_values_list.append(ns_values_epoch.detach())
                    if ex.observers:
                        ex.log_scalar("training.loss_convU_ux" + str(run), ns_values_epoch[0].item(), epoch)
                        ex.log_scalar("training.loss_convV_uy" + str(run), ns_values_epoch[1].item(), epoch)
                        ex.log_scalar("training.loss_convU_vx" + str(run), ns_values_epoch[2].item(), epoch)
                        ex.log_scalar("training.loss_convV_vy" + str(run), ns_values_epoch[3].item(), epoch)
                        ex.log_scalar("training.loss_diffXu" + str(run), ns_values_epoch[4].item(), epoch)
                        ex.log_scalar("training.loss_diffYu" + str(run), ns_values_epoch[5].item(), epoch)
                        ex.log_scalar("training.loss_diffXv" + str(run), ns_values_epoch[6].item(), epoch)
                        ex.log_scalar("training.loss_diffYv" + str(run), ns_values_epoch[7].item(), epoch)
                        ex.log_scalar("training.loss_forceX" + str(run), ns_values_epoch[8].item(), epoch)
                        ex.log_scalar("training.loss_forceY" + str(run), ns_values_epoch[9].item(), epoch)
                        ex.log_scalar("training.loss_ux" + str(run), ns_values_epoch[10].item(), epoch)
                        ex.log_scalar("training.loss_uy" + str(run), ns_values_epoch[11].item(), epoch)
                    # if flag_grad_plot: 
                        # Figures.gradient_distribution(Figures, case, gradients, epoch+epoch_pretrain, ex_ID, nr_layers)
                
                #ns_grads_list.append(grads)  # Out of Memory problem als elke grad wordt opgeslagen
                loss = eqn_loss + lambda_bc * bnc_loss + lambda_data * data_loss
                eqn_loss_tot += eqn_loss.detach() 
                bnc_loss_tot += bnc_loss.detach()  # Detach to prevent storage of computation graph which leads to Out of Memory error.     
                data_loss_tot += data_loss.detach()
                n += 1

                # Optimizer to determine direction of weight and bias update 
                #loss.backward(retain_graph=True)  # Compute the gradients # TODO retain-graph geintroduceerd door evo
                loss.backward()  # Compute the gradients # TODO retain-graph geintroduceerd door evo
                               
                optimizer_u.step()  # Update parameters 
                optimizer_v.step()
                optimizer_p.step()
                if case.input_dimension == 3:
                    optimizer_w.step()
                
                # End of batch - Verbose: 
                if batch_idx % 40 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} Loss BC {:.8f} Loss data {:.8f}'.format(
                        epoch, batch_idx * len(batch_locations[0]), len(dataloader.dataset),
                            100. * batch_idx / len(dataloader), eqn_loss.item(), bnc_loss.item(), data_loss.item()))


            #Figures.hist_batch_retain(Figures, case, ex_ID, epoch, run, batch_len, batchsize) #TODO uit voor 3D 
            
            # Scheduler follows learning rate adapting scheme as specified at beginning 
            scheduler_u.step()
            scheduler_v.step()
            scheduler_p.step()
            if case.input_dimension == 3:
                scheduler_w.step()
            
            eqn_loss_tot /= n
            bnc_loss_tot /= n
            data_loss_tot /= n
            
            # if epoch in vtk_list: 
            #     print('writing vtk')
            #     write_loss_vtk(case, total_loss, loss_c, loss_x, loss_y, loss_c_f, loss_x_f, loss_y_f, epoch)

            # Store loss data every epoch_save_loss iteration, add to MongoDB server if observer is added (DEBUG=False)
            if epoch % epoch_save_loss == 0:
                loss_list[0].append(eqn_loss_tot.detach().item())
                loss_list[1].append(bnc_loss_tot.detach().item())
                loss_list[2].append(data_loss_tot.detach().item())
                if ex.observers:  # check whether an observer is added to the run
                    ex.log_scalar("training.eqn_loss" + str(run), eqn_loss_tot.detach().item(), epoch)
                    ex.log_scalar("training.bnc_loss" + str(run), bnc_loss_tot.detach().item(), epoch)
                    ex.log_scalar("training.data_loss" + str(run), data_loss_tot.detach().item(), epoch)

            # # Save NNs and PINN performance plots every epoch_save_NN
            if epoch % epoch_save_NN == 0:  
                torch.save(net_u.state_dict(), case.path + "/" + str(run) + "/NNfiles/" + case.ID + "data_u_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".pt")
                torch.save(net_v.state_dict(), case.path + "/" + str(run) + "/NNfiles/" + case.ID + "data_v_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".pt")
                torch.save(net_p.state_dict(), case.path + "/" + str(run) + "/NNfiles/" + case.ID + "data_p_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".pt")
                if case.input_dimension == 3:
                    torch.save(net_w.state_dict(), case.path + "/" + str(run) + "/NNfiles/" + case.ID + "data_w_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".pt")

                if epoch != 0:  
                    Figures.loss_plot(Figures, case, ex_ID, loss_list, epoch, epoch_pretrain, run)
                    #Figures.ns_grad_values(Figures, case, ex_ID, epoch, epoch_pretrain, ns_grads_list)
                Figures.velocity_prediction(Figures, case, ex_ID, geometry_locations, networks, epoch, run, batch_idx, show=False)
                #Figures.velocity_prediction3D(Figures, case, ex_ID, geometry_locations, networks, epoch, run, batch_idx, show=True)
                # Figures.data_error(Figures, case, ex_ID, boundary_locations, solution_locations, solution_values, networks, epoch, epoch_pretrain)
                Figures.weight_factors_loss(Figures, case, ex_ID, epoch, epochs, loss_weight_list, run)
                
            # End of epoch - Verbose 
            print('*****Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f} Loss data {:.10f} ****'.format(eqn_loss_tot,
                                                                                                        bnc_loss_tot,
                                                                                                        data_loss_tot))
            print('learning rate is ', optimizer_u.param_groups[0]['lr'], optimizer_v.param_groups[0]['lr'])

        toc = time.time()
        elapseTime = toc - tic
        print("elapse time in parallel = ", elapseTime)

        
        # Save the NNs 
        torch.save(net_u.state_dict(), case.path + "/" + str(run) + "/NNfiles/" + case.ID + "data_u_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".pt")
        torch.save(net_v.state_dict(), case.path + "/" + str(run) + "/NNfiles/" + case.ID + "data_v_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".pt")
        torch.save(net_p.state_dict(), case.path + "/" + str(run) + "/NNfiles/" + case.ID + "data_p_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".pt")
        if case.input_dimension == 3:
            torch.save(net_w.state_dict(), case.path + "/" + str(run) + "/NNfiles/" + case.ID + "data_w_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".pt")

        print("Data saved!")
        
        Figures.loss_plot(Figures, case, ex_ID, loss_list, epoch+1, epoch_pretrain, run)
        # Figures.ns_grad_values(Figures, case, ex_ID, epoch, epoch_pretrain, ns_grads_list)
        Figures.velocity_prediction(Figures, case, ex_ID, geometry_locations, networks, epoch, run, batch_idx, show=False)
        # Figures.data_error(Figures, case, ex_ID, boundary_locations, solution_locations, solution_values, networks, epoch)
        Figures.weight_factors_loss(Figures, case, ex_ID, epoch, epoch_pretrain, loss_weight_list, run)
        

        # Only contains creating VTK for now
        # write_loss_vtk2(case, geometry_locations, networks, means_geo, stds_geo, epoch, ex_ID, run)  # TODO uitzoeken hoe niet 
                                                                            # OOM te gaan door extra iteratie computation graph 
     
        # post(case, ex_ID, geometry_locations, networks, means_geo, stds_geo, output_filename, flag_compare)
        print('jobs done!')
    

ex.run()

#write_loss_vtk2(case, geometry_locations, networks, means_geo, stds_geo, epoch, ex_ID, run)




