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



from scripts.file_readers import data_reader, file_reader
from scripts import case_info
from scripts.figure_creation import Figures


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
        sys.exit('unable to mount to CUDA, please retry. fMake sure CUDA is installed') 


    case = case_info.stenose2D()
    ex_ID = '003'
    output_filename = "./results/" + case.name + "/outputs/" + case.ID + "PINNoutput_" + ex_ID + ".vtk"
    DEBUG = True

    
    geometry_locations, minima, maxima  = file_reader(case.mesh_file, mesh=True)
    boundary_locations, _, _ = file_reader(case.bc_file, mesh=False, min=minima, max=maxima)
    solution_locations, solution_values = data_reader(case, min=minima, max=maxima)


    Figures.scaled_geometry(Figures, boundary_locations, solution_locations, solution_values, show=False)

    # if not DEBUG:
    #     AppendObserver(case, ex)


    Flag_pretrain = False  # Whether you want to load a pretrained network and train further
    Flag_notrain = False  # Whether you want to load a pretrained network and look at postprocessing
    Flag_lossTerm = True
    epochs = 5500
    epoch_pretrain = 0 if not Flag_pretrain else 5500  # epochs after reading a pretrained network
    epoch_save_loss = 2  # determines how often the loss values are saved
    epoch_save_NN = 250  # determines how often the NN is updated -- 'fail-save' mechanism should training stop before
    # final epoch
    remark = "This test is to compare X to Y"  # note for future reference

    learning_rate = 5e-4  # starting learning rate
    step_epoch = 1200  # after how many iterations the learning rate should be adapted
    decay_rate = 0.1  # 0.1
    alpha = 0.9  # momentum coefficient for learning rate annealing
    batchsize = 256  # common; 2D : 256, 3D : 512
    nr_losses = 3

@ex.main
def geo_train(device, case, epochs, epoch_save_loss, epoch_save_NN, geometry_locations, boundary_locations, solution_locations,
 solution_values, batchsize, nr_losses, learning_rate, Flag_notrain, Flag_pretrain,
              step_epoch, decay_rate, ex_ID, output_filename, epoch_pretrain, alpha, Flag_lossTerm):
    
    # # Build the NNs
    # net2_u = Net3(case.input_n, case.h_n).to(device)
    # net2_v = Net3(case.input_n, case.h_n).to(device)
    # net2_p = Net3(case.input_n, case.h_n).to(device) 
    # net2_w = Net3(case.input_n, case.h_n).to(device)

    # # Reading the pretrained networks if specified 
    # if Flag_notrain or Flag_pretrain:
    #     print('Reading (pretrain) functions first...')
    #     root = tk.Tk()
    #     root.withdraw()

    #     print('Please specify the NN file (.pt) for the velocity u')
    #     file_path_u = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
    #     # file_path_u = 'D:/Graduation_project/AA_files/2DSTEN/004/DataReader_adapted/STEN2DTB_data_u.pt'
    #     print('Please specify the NN file (.pt) for the velocity v')
    #     file_path_v = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
    #     # file_path_v = 'D:/Graduation_project/AA_files/2DSTEN/004/DataReader_adapted/STEN2DTB_data_v.pt'
    #     print('Please specify the NN file (.pt) for the velocity p')
    #     file_path_p = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
    #     # file_path_p = 'D:/Graduation_project/AA_files/2DSTEN/004/DataReader_adapted/STEN2DTB_data_p.pt'

    #     if case.input_n == 3:
    #         print('Please specify the NN file (.pt) for the velocity w')
    #         file_path_w = filedialog.askopenfilename(filetypes=(("NN files", "*.pt"), ("all files", "*.*")))
    #         net2_w.load_state_dict(torch.load(file_path_w))
    #     # root.destroy()

    #     net2_u.load_state_dict(torch.load(file_path_u))
    #     net2_v.load_state_dict(torch.load(file_path_v))
    #     net2_p.load_state_dict(torch.load(file_path_p))

    # if Flag_notrain:
    #     post(case, epochs, net2_u, net2_v, net2_w, net2_p, output_filename, ex_ID, plot_vecfield=False, plot_streamline=False)

    #     # calculating gradients wrt losses, plotting for each layer
    #     # TODO : Zorgen dat dit weer werkt voor de no_train flag
    #     # epoch = 0
    #     # anneal_weight = []
    #     # GradLosses(case, device, ex_ID, epoch, epoch_pretrain, x, y, z, xb, yb, zb, xd, yd, zd, ud, vd, wd, net2_u,
    #     #            net2_v, net2_w, net2_p, nr_losses, alpha, anneal_weight, Flag_grad=True, Flag_notrain=True)

    #     sys.exit('postprocessing executed')
    # print('device used is', device)



    # x = torch.Tensor(x).to(device)
    # y = torch.Tensor(y).to(device)
    # z = torch.Tensor(z).to(device)
    # x_plot = torch.clone(x)
    # y_plot = torch.clone(y)
    # z_plot = torch.clone(z)
    # xb = torch.Tensor(xb).to(device)
    # yb = torch.Tensor(yb).to(device)
    # zb = torch.Tensor(zb).to(device)
    # xb_plot = torch.clone(xb)
    # yb_plot = torch.clone(yb)
    # zb_plot = torch.clone(zb)
    # xd = torch.Tensor(xd).to(device)
    # yd = torch.Tensor(yd).to(device)
    # zd = torch.Tensor(zd).to(device)
    # xd_plot = torch.clone(xd)
    # yd_plot = torch.clone(yd)
    # zd_plot = torch.clone(zd)
    # ud = torch.Tensor(ud).to(device)
    # vd = torch.Tensor(vd).to(device)
    # wd = torch.Tensor(wd).to(device)
    # ud_plot = torch.clone(ud)
    # vd_plot = torch.clone(vd)
    # wd_plot = torch.clone(wd)
    # if case.input_n == 3:  # TODO data size afhankelijk maken?
    #     x_plot = x_plot[::4]
    #     y_plot = y_plot[::4]
    #     z_plot = z_plot[::4]

    # Build arrays as Tensors and move to device 
    x, y, z = [torch.Tensor(geometry_locations[axis]).to(device) for axis in range(len(geometry_locations))]
    xb, yb, zb = [torch.Tensor(boundary_locations[axis]).to(device) for axis in range(len(boundary_locations))]
    xd, yd, zd = [torch.Tensor(solution_locations[axis]).to(device) for axis in range(len(solution_locations))]
    ud, vd, wd = [torch.Tensor(solution_values[axis]).to(device) for axis in range(len(solution_values))]



    dataset = TensorDataset(x, y, z)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=True)

    loss_list = [[] for _ in range(nr_losses)]
    anneal_weight = [[1.] for _ in range(nr_losses-1)]
    anneal_weight2 = [[1.] for _ in range(nr_losses - 1)]
    NSgrads_list = []
    NSterms_list = []
   
    tic = time.time()

    if not Flag_pretrain:
        net2_u.apply(init_normal)  # TODO init Xavier?
        net2_v.apply(init_normal)
        net2_w.apply(init_normal)
        net2_p.apply(init_normal)

    optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=10 ** -15)
    optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=10 ** -15)
    optimizer_w = optim.Adam(net2_w.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=10 ** -15)
    optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=10 ** -15)

    scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
    scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
    if case.input_n == 3:
        scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
    scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)

    # scheduler_u = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_u, factor=0.5, verbose=True)
    # scheduler_v = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_v, factor=0.5, verbose=True)
    # scheduler_p = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p, factor=0.5, verbose=True)

    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=1, warmup=0, active=2, repeat=2),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
    #         record_shapes=True,
    #         profile_memory=False,
    #         with_stack=True
    # ) as prof:
    grad_list = [0, 10, 20, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    Lambda_BC = case.Lambda_BC
    Lambda_data = case.Lambda_data
    Flag_grad = False
    for epoch in range(epochs):

        if epoch in grad_list:
            Flag_grad = True
            print('Executing gradient loss algorithm this iteration')

        loss_eqn_tot = 0.
        loss_bc_tot = 0.
        loss_data_tot = 0.
        n = 0
        for batch_idx, (x_in, y_in, z_in) in enumerate(dataloader):  # standaard is 3D, maar als Z=0 heeft dat geen
            # invloed op loss functies
            # prof.step()
            net2_u.zero_grad()
            net2_v.zero_grad()
            net2_w.zero_grad()
            net2_p.zero_grad()
            loss_eqn, grads, loss_terms = criterion2(x_in, y_in, z_in, net2_u, net2_v, net2_w, net2_p, case.X_scale, case.Y_scale,
                                 case.Z_scale, case.U_scale, case.Diff, case.rho, case.input_n) # TODO verschillende criterions
            loss_bc = Loss_BC(xb, yb, zb, net2_u, net2_v, net2_w, case.input_n)
            loss_data = Loss_data(xd, yd, zd, ud, vd, wd, net2_u, net2_v, net2_w, case.input_n)

            if epoch % 10 == 0 and len(anneal_weight[0]) == epoch/10:
                anneal_weight, Lambda_BC, Lambda_data = GradLosses(case, device, ex_ID, epoch, epoch_pretrain,
                                                                    x_in, y_in, z_in, xb, yb, zb, xd, yd, zd, ud, vd, wd,
                                                                    net2_u, net2_v, net2_w, net2_p, alpha,
                                                                    anneal_weight, Flag_grad, Lambda_BC, Lambda_data,
                                                                             Flag_notrain=False)

                Flag_grad = False

            loss = loss_eqn + Lambda_BC * loss_bc + Lambda_data * loss_data
            loss.backward()
            optimizer_u.step()
            optimizer_v.step()
            if case.input_n == 3:
                optimizer_w.step()
            optimizer_p.step()

            loss_eqn_tot += loss_eqn.detach()  # Detach essentieel om te voorkomen dat de hele computation graph ook hier wordt opgeslagen, leidt tot OOM
            # loss_eqn_tot.detach()
            loss_bc_tot += loss_bc.detach()
            # loss_bc_tot.detach()
            loss_data_tot += loss_data.detach()
            n += 1
            if batch_idx % 40 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} Loss BC {:.8f} Loss data {:.8f}'.format(
                    epoch, batch_idx * len(x_in), len(dataloader.dataset),
                           100. * batch_idx / len(dataloader), loss_eqn.item(), loss_bc.item(), loss_data.item()))

        NSgrads_list.append(grads)
        NSterms_list.append(loss_terms)
        scheduler_u.step()
        scheduler_v.step()
        if case.input_n == 3:
            scheduler_w.step()
        scheduler_p.step()
        loss_eqn_tot /= n
        loss_bc_tot /= n
        loss_data_tot /= n
        print('*****Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f} Loss data {:.10f} ****'.format(loss_eqn_tot,
                                                                                                     loss_bc_tot,
                                                                                                     loss_data_tot))
        print('learning rate is ', optimizer_u.param_groups[0]['lr'], optimizer_v.param_groups[0]['lr'])

        if epoch % epoch_save_loss == 0:
            loss_list[0].append(loss_eqn_tot.item())
            loss_list[1].append(loss_bc_tot.item())
            loss_list[2].append(loss_data_tot.item())
            if ex.observers:  # check whether an observer is added to the run
                ex.log_scalar("training.loss_eqn", loss_eqn_tot.item(), epoch)
                ex.log_scalar("training.loss_bc", loss_bc_tot.item(), epoch)
                ex.log_scalar("training.loss_data", loss_data_tot.item(), epoch)
                if case.input_n == 2 and Flag_lossTerm:
                    ex.log_scalar("training.loss_convU_ux", loss_terms[0].item(), epoch)
                    ex.log_scalar("training.loss_convV_uy", loss_terms[1].item(), epoch)
                    ex.log_scalar("training.loss_convU_vx", loss_terms[2].item(), epoch)
                    ex.log_scalar("training.loss_convV_vy", loss_terms[3].item(), epoch)
                    ex.log_scalar("training.loss_diffXu", loss_terms[4].item(), epoch)
                    ex.log_scalar("training.loss_diffYu", loss_terms[5].item(), epoch)
                    ex.log_scalar("training.loss_diffXv", loss_terms[6].item(), epoch)
                    ex.log_scalar("training.loss_diffYv", loss_terms[7].item(), epoch)
                    ex.log_scalar("training.loss_forceX", loss_terms[8].item(), epoch)
                    ex.log_scalar("training.loss_forceY", loss_terms[9].item(), epoch)
                    # TODO kan 3D nog toevoegn, let er wel op dat bovenstaande volgorde niet voor 3D geldt

        if epoch % epoch_save_NN == 0:  # save network and loss plot
            torch.save(net2_u.state_dict(), case.path + "/NNfiles/" + case.ID + "data_u_" + ex_ID + ".pt")
            torch.save(net2_v.state_dict(), case.path + "/NNfiles/" + case.ID + "data_v_" + ex_ID + ".pt")
            torch.save(net2_p.state_dict(), case.path + "/NNfiles/" + case.ID + "data_p_" + ex_ID + ".pt")
            if case.input_n == 3:
                torch.save(net2_w.state_dict(), case.path + "/NNfiles/" + case.ID + "data_w_" + ex_ID + ".pt")
            print('NN saved')

            # steps = np.linspace(0, epochs - epoch_save_loss, np.int(np.divide(epochs, epoch_save_loss)))
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
                               1) if case.input_n == 3 else \
                torch.cat((x_plot.requires_grad_(), y_plot.requires_grad_()), 1)
            # net_in = torch.cat((x.requires_grad_(), y.requires_grad_()), 1)
            output_u = net2_u(net_in)  # evaluate model
            output_v = net2_v(net_in)  # evaluate model
            output_u = output_u.cpu().data.numpy()  # need to convert to cpu before converting to numpy
            output_v = output_v.cpu().data.numpy()

            net_in_data = torch.cat((xd.requires_grad_(), yd.requires_grad_(), zd.requires_grad_()), 1) \
                if case.input_n ==3 else torch.cat((xd.requires_grad_(), yd.requires_grad_()), 1)
            output_ud = net2_u(net_in_data)
            output_vd = net2_v(net_in_data)
            output_ud = output_ud.cpu().data.numpy()  # need to convert to cpu before converting to numpy
            output_vd = output_vd.cpu().data.numpy()

            if case.input_n == 3:
                output_w = net2_w(net_in)
                output_w = output_w.cpu().data.numpy()
                output_wd = net2_w(net_in_data)
                output_wd = output_wd.cpu().data.numpy()

            x_plot2 = x_plot.cpu()
            y_plot2 = y_plot.cpu()
            z_plot2 = z_plot.cpu()
            plt.figure(6)
            plt.subplot(2, 1, 1)
            plt.scatter(z_plot2.detach().numpy(), x_plot2.detach().numpy(), c=output_w, cmap='rainbow') if case.input_n == 3 \
                else plt.scatter(x_plot2.detach().numpy(), y_plot2.detach().numpy(), c=output_u, cmap='rainbow')
            plt.title('NN results, u (top) & v (bot), - epoch' + str(epoch_pretrain + epoch))
            plt.colorbar()
            plt.subplot(2, 1, 2)
            plt.scatter(z_plot2.detach().numpy(), x_plot2.detach().numpy(), c=output_u, cmap='rainbow') if case.input_n == 3 \
                 else plt.scatter(x_plot2.detach().numpy(), y_plot2.detach().numpy(), c=output_v, cmap='rainbow')
            plt.colorbar()
            plt.savefig(case.path + "/plots/" + case.ID + "plotWU_" + ex_ID + "_" + str(epoch + epoch_pretrain)) if case.input_n == 3 \
                else plt.savefig(case.path + "/plots/" + case.ID + "plotUV_" + ex_ID + "_" + str(epoch + epoch_pretrain))
            plt.close(6)
            # plt.show()

            xd_plot2 = xd_plot.cpu()
            yd_plot2 = yd_plot.cpu()
            zd_plot2 = zd_plot.cpu()
            xb_plot2 = xb_plot.cpu()
            yb_plot2 = yb_plot.cpu()
            zb_plot2 = zb_plot.cpu()
            ud_plot2 = ud_plot.cpu().data.numpy()
            vd_plot2 = vd_plot.cpu().data.numpy()
            ud_diff = (output_ud - ud_plot2) / ud_plot2 * 100
            vd_diff = (output_vd - vd_plot2) / vd_plot2 * 100
            if case.input_n == 3:
                wd_plot2 = wd_plot.cpu().data.numpy()
                wd_diff = (output_wd - wd_plot2) / wd_plot2 * 100
            # ud_diff_geo = (output_u / ud_geo)
            # vd_diff_geo = (output_v / vd_geo)

            plt.figure(7)
            plt.subplot(2, 1, 1)
            plt.scatter(zb_plot2.detach().numpy(), xb_plot2.detach().numpy()) if case.input_n == 3 \
                else  plt.scatter(xb_plot2.detach().numpy(), yb_plot2.detach().numpy())
            plt.scatter(xd_plot2.detach().numpy(), yd_plot2.detach().numpy(), c=ud_diff, cmap='rainbow')
            plt.title('% error; ud (top) & vd (bot), - epoch' + str(epoch_pretrain + epoch))
            plt.colorbar()
            plt.subplot(2, 1, 2)
            plt.scatter(xb_plot2.detach().numpy(), yb_plot2.detach().numpy())
            plt.scatter(xd_plot2.detach().numpy(), yd_plot2.detach().numpy(), c=vd_diff, cmap='rainbow')
            plt.colorbar()
            plt.savefig(case.path + "/plots/" + case.ID + "plotD_UV_" + ex_ID + "_" + str(epoch + epoch_pretrain))
            plt.close(7)
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


            steps = np.linspace(epoch_pretrain, epoch_pretrain + epoch, len(anneal_weight[0]))
            plt.figure(9)
            for i in range(nr_losses - 1):
                plt.plot(steps, anneal_weight[i])
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

    torch.save(net2_u.state_dict(), case.path + "/NNfiles/" + case.ID + "data_u_" + ex_ID + ".pt")
    torch.save(net2_v.state_dict(), case.path + "/NNfiles/" + case.ID + "data_v_" + ex_ID + ".pt")
    torch.save(net2_p.state_dict(), case.path + "/NNfiles/" + case.ID + "data_p_" + ex_ID + ".pt")
    if case.input_n == 3:
        torch.save(net2_w.state_dict(), case.path + "/NNfiles/" + case.ID + "data_w_" + ex_ID + ".pt")

    print("Data saved!")

    # # steps = np.linspace(0, epoch - epoch_save_loss.np.int(np.divide(epoch, epoch_save_loss)))
    steps = np.linspace(epoch_pretrain, epoch_pretrain+epochs - epoch_save_loss, np.int(np.divide(epochs, epoch_save_loss)))
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


    plt.figure(11)
    bins = 50
    #### X
    min_ux = [list[0][0].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(min_ux)
    n, bins_data_min_ux, patches = plt.hist(min_ux, bins)
    min_ux_data = stats.norm.pdf(bins_data_min_ux, mu, sigma)

    max_ux = [list[0][1].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(max_ux)
    n, bins_data_max_ux, patches = plt.hist(max_ux, bins)
    max_ux_data = stats.norm.pdf(bins_data_max_ux, mu, sigma)

    min_uxx = [list[0][2].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(min_uxx)
    n, bins_data_min_uxx, patches = plt.hist(min_uxx, bins)
    min_uxx_data = stats.norm.pdf(bins_data_min_uxx, mu, sigma)

    max_uxx = [list[0][3].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(max_uxx)
    n, bins_data_max_uxx, patches = plt.hist(max_uxx, bins)
    max_uxx_data = stats.norm.pdf(bins_data_max_uxx, mu, sigma)

    min_vx = [list[1][0].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(min_vx)
    n, bins_data_min_vx, patches = plt.hist(min_vx, bins)
    min_vx_data = stats.norm.pdf(bins_data_min_vx, mu, sigma)

    max_vx = [list[1][1].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(max_vx)
    n, bins_data_max_vx, patches = plt.hist(max_vx, bins)
    max_vx_data = stats.norm.pdf(bins_data_max_vx, mu, sigma)

    min_vxx = [list[1][2].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(min_vxx)
    n, bins_data_min_vxx, patches = plt.hist(min_vxx, bins)
    min_vxx_data = stats.norm.pdf(bins_data_min_vxx, mu, sigma)

    max_vxx = [list[1][3].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(max_vxx)
    n, bins_data_max_vxx, patches = plt.hist(max_vxx, bins)
    max_vxx_data = stats.norm.pdf(bins_data_max_vxx, mu, sigma)

    ######  Y
    min_uy = [list[2][0].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(min_uy)
    n, bins_data_min_uy, patches = plt.hist(min_uy, bins)
    min_uy_data = stats.norm.pdf(bins_data_min_ux, mu, sigma)

    max_uy = [list[2][1].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(max_uy)
    n, bins_data_max_uy, patches = plt.hist(max_uy, bins)
    max_uy_data = stats.norm.pdf(bins_data_max_uy, mu, sigma)

    min_uyy = [list[2][2].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(min_uyy)
    n, bins_data_min_uyy, patches = plt.hist(min_uyy, bins)
    min_uyy_data = stats.norm.pdf(bins_data_min_uyy, mu, sigma)

    max_uyy = [list[2][3].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(max_uyy)
    n, bins_data_max_uyy, patches = plt.hist(max_uyy, bins)
    max_uyy_data = stats.norm.pdf(bins_data_max_uyy, mu, sigma)

    min_vy = [list[3][0].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(min_vy)
    n, bins_data_min_vy, patches = plt.hist(min_vy, bins)
    min_vy_data = stats.norm.pdf(bins_data_min_vy, mu, sigma)

    max_vy = [list[3][1].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(max_vy)
    n, bins_data_max_vy, patches = plt.hist(max_vy, bins)
    max_vy_data = stats.norm.pdf(bins_data_max_vy, mu, sigma)

    min_vyy = [list[3][2].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(min_vyy)
    n, bins_data_min_vyy, patches = plt.hist(min_vyy, bins)
    min_vyy_data = stats.norm.pdf(bins_data_min_vyy, mu, sigma)

    max_vyy = [list[3][3].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(max_vyy)
    n, bins_data_max_vyy, patches = plt.hist(max_vyy, bins)
    max_vyy_data = stats.norm.pdf(bins_data_max_vyy, mu, sigma)

    ####### P
    min_px = [list[4][0].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(min_px)
    n, bins_data_min_px, patches = plt.hist(min_px, bins)
    min_px_data = stats.norm.pdf(bins_data_min_px, mu, sigma)

    max_px = [list[4][1].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(max_px)
    n, bins_data_max_px, patches = plt.hist(max_px, bins)
    max_px_data = stats.norm.pdf(bins_data_max_px, mu, sigma)

    min_py = [list[4][2].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(min_py)
    n, bins_data_min_py, patches = plt.hist(min_py, bins)
    min_py_data = stats.norm.pdf(bins_data_min_py, mu, sigma)

    max_py = [list[4][3].item() for list in NSgrads_list]
    (mu, sigma) = stats.norm.fit(max_py)
    n, bins_data_max_py, patches = plt.hist(max_py, bins)
    max_py_data = stats.norm.pdf(bins_data_max_py, mu, sigma)


    plt.subplot(3,2,1)
    plt.plot(bins_data_min_ux, min_ux_data)
    plt.plot(bins_data_max_ux, max_ux_data)
    plt.plot(bins_data_min_uxx, min_uxx_data)
    plt.plot(bins_data_max_uxx, max_uxx_data)
    plt.legend(['min_ux', 'max_ux', 'min_uxx', 'max_uxx'])


    plt.subplot(3,2,2)
    plt.plot(bins_data_min_vx, min_vx_data)
    plt.plot(bins_data_max_vx, max_vx_data)
    plt.plot(bins_data_min_vxx, min_vxx_data)
    plt.plot(bins_data_max_vxx, max_vxx_data)
    plt.legend(['min_vx', 'max_vx', 'min_vxx', 'max_vxx'])

    plt.subplot(3,2,3)
    plt.plot(bins_data_min_uy, min_uy_data)
    plt.plot(bins_data_max_uy, max_uy_data)
    plt.plot(bins_data_min_uyy, min_uyy_data)
    plt.plot(bins_data_max_uyy, max_uyy_data)
    plt.legend(['min_uy', 'max_uy', 'min_uyy', 'max_uyy'])

    plt.subplot(3,2,4)
    plt.plot(bins_data_min_vy, min_vy_data)
    plt.plot(bins_data_max_vy, max_vy_data)
    plt.plot(bins_data_min_vyy, min_vyy_data)
    plt.plot(bins_data_max_vyy, max_vyy_data)
    plt.legend(['min_vy', 'max_vy', 'min_vyy', 'max_vyy'])

    plt.subplot(3,2,5)
    plt.plot(bins_data_min_px, min_px_data)
    plt.plot(bins_data_max_px, max_px_data)
    plt.plot(bins_data_min_py, min_py_data)
    plt.plot(bins_data_max_py, max_py_data)
    plt.legend(['min_px', 'max_px', 'min_py', 'max_py'])

    # plt.show()

    epochs += epoch_pretrain
    post(case, epochs, net2_u, net2_v, net2_w, net2_p, output_filename, ex_ID, plot_vecfield=False, plot_streamline=False,
         writeVTK=True)

    print('jobs done!')
ex.run()





