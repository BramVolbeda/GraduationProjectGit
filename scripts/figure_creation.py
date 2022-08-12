import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import numpy as np
import torch
import scipy.stats as stats
from scipy.interpolate import griddata
import plotly.express as px 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class Figures: 
    def __init__(self, inplace=True):
        super(Figures, self).__init__()
        self.inplace = inplace
    

    def scaled_geometry(self, case, ex_ID, boundary_locations, solution_locations, solution_values, run, show=False): 
        """
        Plots boundary of the geometry plus the data points' locations and values within this geometry. 

        Parameters: 
        case (class) : The case to which the PINN is applied, contains case-specific information.
        ex_ID (string) : Contains the name of the current experiment, added to the filename during storage. 
        boundary_locations (tensor) : The x, y(, z) coordinates of the walls of the geometry.
        solution_locations (tensor) : The x, y(, z) coordinates of the data points. 
        solution_values (tensor) :  The u, v(, w) values for the data points. 
        networks (list) : List with all the neural networks (u, v, w and P). 
        epoch (int) : Current epoch, used together with epoch_pretrain to determine the total number of epochs. Used in the title. 
        epoch_pretrain (int) : Amount of epochs the networks have been trained with in case of a pretrained network. 

        """
        
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.scatter(boundary_locations[0], boundary_locations[1])  # xb, yb
        plt.scatter(solution_locations[0], solution_locations[1], c=solution_values[0], cmap='autumn_r')
        plt.title('scaled data points, u (top) and v (bot')
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.scatter(boundary_locations[0], boundary_locations[1])
        plt.scatter(solution_locations[0], solution_locations[1], c=solution_values[1], cmap='autumn_r')
        plt.colorbar()
        plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "scaling" + ex_ID + "_" + str(run))
        if show: 
            plt.show()
        plt.close()


    def loss_plot(self, case, ex_ID, loss_list, epoch, epoch_pretrain, run, show=False): 
        """
        Plots the prediction for the axial velocity (top) and radial velocity (bot) for the current epoch. 

        Parameters: 
        case (class) : The case to which the PINN is applied, contains case-specific information.
        ex_ID (string) : Contains the name of the current experiment, added to the filename during storage. 
        loss_list (list) : List that stores the values of the loss functions. 
        epoch (int) : Current epoch, used together with epoch_pretrain to determine the total number of epochs. Used in the title. 
        epoch_pretrain (int) : Amount of epochs the networks have been trained with in case of a pretrained network. 
        """
        steps = np.linspace(epoch_pretrain, epoch_pretrain+epoch, len(loss_list[0]))
        colors = ['b', 'g', 'r']
        plt.figure()
        for i in range(len(loss_list)):
            plt.plot(steps, loss_list[i], colors[i])
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend(['eq', 'bc', 'data'])
        plt.title(case.name)
        if show:
            plt.show()
        plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "loss_plot_" + ex_ID + "_" + str(run) + ".png")
        plt.close()


    def velocity_prediction(self, case, ex_ID, geometry_locations, networks, epoch, run, batch_idx, show=False): # TODO: werkt nog niet goed voor 3D 
        """
        Plots the prediction for the axial velocity (top) and radial velocity (bot) for the current epoch. 

        Parameters: 
        case (class) : The case to which the PINN is applied, contains case-specific information.
        ex_ID (string) : Contains the name of the current experiment, added to the filename during storage. 
        geometry_locations (tensor) : The x, y(, z) coordinates within the geometry. 
        networks (list) : List with all the neural networks (u, v, w and P). 
        epoch (int) : Current epoch, used together with epoch_pretrain to determine the total number of epochs. Used in the title. 
        epoch_pretrain (int) : Amount of epochs the networks have been trained with in case of a pretrained network. 

        
        """
        input_network_geo = torch.cat(([axis for axis in geometry_locations]), 1)
        prediction_values = [network(input_network_geo).view(len(input_network_geo), -1) for network in networks]
        
        u, v, *w = [solution.cpu().data.numpy() for solution in prediction_values]
        x, y, *z = [axis.cpu().data.numpy() for axis in geometry_locations]
        
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.scatter(z, x, c=w, cmap='rainbow') if case.input_dimension == 3 else plt.scatter(x, y, c=u, cmap='rainbow')
        plt.title('NN results, u (top) & v (bot), - epoch' + str(epoch))
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.scatter(z, x, c=u, cmap='rainbow') if case.input_dimension == 3 else plt.scatter(x, y, c=v, cmap='rainbow')
        plt.colorbar()
        plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "plotWU_" + ex_ID + "_" + str(epoch) + "_" + str(batch_idx) + "_" + str(run)) if case.input_dimension == 3 \
            else plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "plotUV_" + ex_ID + "_" + str(epoch) +"_" + str(batch_idx) + "_" + str(run))
        if show: 
            plt.show()
        plt.close()

    def velocity_prediction3D(self, case, ex_ID, geometry_locations, networks, epoch, run, batch_idx, show=False): # TODO: werkt nog niet goed voor 3D 
        """
        Plots the prediction for the axial velocity (top) and radial velocity (bot) for the current epoch. 

        Parameters: 
        case (class) : The case to which the PINN is applied, contains case-specific information.
        ex_ID (string) : Contains the name of the current experiment, added to the filename during storage. 
        geometry_locations (tensor) : The x, y(, z) coordinates within the geometry. 
        networks (list) : List with all the neural networks (u, v, w and P). 
        epoch (int) : Current epoch, used together with epoch_pretrain to determine the total number of epochs. Used in the title. 
        epoch_pretrain (int) : Amount of epochs the networks have been trained with in case of a pretrained network. 

        
        """
        input_network_geo = torch.cat(([axis for axis in geometry_locations]), 1)
        prediction_values = [network(input_network_geo).view(len(input_network_geo), -1) for network in networks]
        
        u = prediction_values[0].cpu().data.numpy()
        v = prediction_values[1].cpu().data.numpy()
        w = prediction_values[2].cpu().data.numpy()
        x, y, z = [axis.cpu().data.numpy() for axis in geometry_locations]
        
        coordinates = np.concatenate((x, y, z), axis=1)
        grid_y, grid_x = np.mgrid[-0.9:0.9:10j, -0.9:0.9:10j]
        equation = lambda x,y : x-y
        grid_z = equation(grid_x, grid_y)
        grid_z2 = np.zeros_like(grid_x)


        interp = griddata(coordinates, w, (grid_x, grid_y, grid_z), method='linear')
        interp2 = griddata(coordinates, w, (grid_x, grid_y, grid_z2), method='linear')

        plt.subplot(121)
        plt.imshow(interp,  origin='lower')
        plt.subplot(122)
        plt.contourf(grid_x, grid_y, interp2,  origin='lower')
        plt.title('temperature along 0.8*(1-x)')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title('NN results, u (top) & v (bot), - epoch' + str(epoch))
        plt.colorbar()
        
        # ax = plt.subplot(122, projection=Axes3D.name)
        # ax.scatter(coordinates[:,0], coordinates[:,1], coordinates[:,2], c=w)
        # ax.set_zlim(-.1,1.1)
        # ax.plot_surface(grid_x,grid_y,grid_z, facecolors=plt.cm.viridis(interp),
        #                 linewidth=0, antialiased=False, shade=False)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "plotWU_" + ex_ID + "_" + str(epoch) + str(batch_idx) + "_" + str(run)) if case.input_dimension == 3 \
            else plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "plotUV_" + ex_ID + "_" + str(epoch) + str(batch_idx) + "_" + str(run))
        if show: 
            plt.show()
        plt.close()
        


    def data_error(self, case, ex_ID, boundary_locations, solution_locations, solution_values, networks, epoch, epoch_pretrain, run, show=False):
        """
        Plots the difference between the predicted velocity and the 'ground truth' velocity for the given data points for current epoch. 
        Axial velocity (top) and radial velocity (bot). 

        Parameters: 
        case (class) : The case to which the PINN is applied, contains case-specific information.
        ex_ID (string) : Contains the name of the current experiment, added to the filename during storage. 
        boundary_locations (tensor) : The x, y(, z) coordinates of the walls of the geometry.
        solution_locations (tensor) : The x, y(, z) coordinates of the data points. 
        solution_values (tensor) :  The u, v(, w) values for the data points. 
        networks (list) : List with all the neural networks (u, v, w and P). 
        epoch (int) : Current epoch, used together with epoch_pretrain to determine the total number of epochs. Used in the title. 
        epoch_pretrain (int) : Amount of epochs the networks have been trained with in case of a pretrained network. 

        """
        input_network_data = torch.cat(([axis for axis in solution_locations]), 1)
        prediction_values = [network(input_network_data).view(len(input_network_data), -1) for network in networks]

        u, v, *w = [solution.cpu().data.numpy() for solution in prediction_values]
        solution_values = [axis.cpu().detach().numpy() for axis in solution_values]
        ud = solution_values[0]
        vd = solution_values[1].transpose()

        xb, yb, *zb = [axis.cpu().detach().numpy() for axis in boundary_locations]
        xd, yd, *zd = [axis.cpu().detach().numpy() for axis in solution_locations]

        ud_diff = (u - ud) / ud * 100
        vd_diff = (v - vd) / vd * 100
        # if case.input_n == 3:
            # wd_diff = (w - wd) / wd * 100
        
        plt.figure()
        plt.subplot(2, 1, 1)
        # plt.scatter(zb_plot2.detach().numpy(), xb_plot2.detach().numpy()) if case.input_n == 3 \
        #     else  plt.scatter(xb_plot2.detach().numpy(), yb_plot2.detach().numpy())
        plt.scatter(xb, yb)
        plt.scatter(xd, yd, c=ud_diff, cmap='rainbow')
        plt.title('% error; ud (top) & vd (bot), - epoch' + str(epoch))
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.scatter(xb, yb)
        plt.scatter(xd, yd, c=vd_diff, cmap='rainbow')
        plt.colorbar()
        plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "plotD_UV_" + ex_ID + "_" + str(epoch + epoch_pretrain) + "_" + str(run))
        if show:
            plt.show()
        plt.close()


    def weight_factors_loss(self, case, ex_ID, epoch, epochs, loss_weight_list, run): 
        """
        Plots the evolution of the weight factor for the boundary_condition loss and the data loss. 

        Parameters: 
        case (class) : The case to which the PINN is applied, contains case-specific information.
        ex_ID (string) : Contains the name of the current experiment, added to the filename during storage. 
        epoch (int) : Current epoch, used together with epoch_pretrain to determine the total number of epochs. Used in the title. 
        epoch_pretrain (int) : Amount of epochs the networks have been trained with in case of a pretrained network. 
        loss_weight_list (list) : List to store the values of the weight factors after the weight annealing algorithm is applied.

        """
        steps = np.linspace(epoch, epochs, len(loss_weight_list[0]))
        plt.figure()
        for i in range(len(loss_weight_list)):
            plt.plot(steps, loss_weight_list[i])
        plt.xlabel('epochs')
        # plt.ylabel('Loss')
        # plt.yscale('log')
        plt.legend(['lambda_bc', 'lambda_data'])
        plt.title(case.name)
        # plt.show()
        plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "learningWeight_plot_" + ex_ID + "_" + str(run) + ".png")
        plt.close()

    def gradient_losses(self, case, ex_ID, epoch, epoch_pretrain, nr_layers, eqn_loss, bnc_loss, data_loss, networks, run):

        net_u = networks[0]
        # eqn_loss, _, _ = loss_geo_v3(case, predictions_geo, batch_locations, means_geo, stds_geo, flag_values=True)  # Compare prediction to Navier-Stokes eq. TODO: grads houden?
        # bnc_loss = loss_bnc(case, predictions_bnc)  # Compare prediction to no-slip condition
        # data_loss = loss_data(case, predictions_data, solution_values)  # Compare prediction to data solution

        max_grad_r_list = []
        mean_grad_bc_list = []
        mean_grad_data_list = []
        bins_res_list = [[] for _ in range(nr_layers-1)]
        y_res_list = [[] for _ in range(nr_layers-1)]
        bins_bc_list = [[] for _ in range(nr_layers - 1)]
        y_bc_list = [[] for _ in range(nr_layers - 1)]
        bins_data_list = [[] for _ in range(nr_layers - 1)]
        y_data_list = [[] for _ in range(nr_layers - 1)]

        eqn_loss.backward(retain_graph=True)

        fig_nr = epoch

        layer = 0
        
        bins = 50
        plt.figure(fig_nr)
        print('Analyzing gradients wrt residual loss')
        for name, param in net_u.named_parameters(prefix=''):
            if "weight" in name and layer < nr_layers - 1:
                layer += 1
                print(f"Layer: {name} | Size: {param.size()}")
                data = param.grad.cpu().detach().numpy()

                (mu, sigma) = stats.norm.fit(data)
                n, bins_res, patches = plt.hist(data, bins)
                y_res = stats.norm.pdf(bins_res, mu, sigma)
                bins_res_list[layer-1] = bins_res
                y_res_list[layer-1] = y_res
                max_grad_r_list.append(torch.max(abs(param.grad)))


        bnc_loss.backward(retain_graph=True)
        print('Analyzing gradients wrt boundary loss')
        plt.figure(fig_nr)
        layer = 0
        for name, param in net_u.named_parameters(prefix=''):
            if "weight" in name and layer < nr_layers - 1:
                layer += 1
        

                print(f"Layer: {name} | Size: {param.size()}")
                data = param.grad.cpu().detach().numpy()

                (mu, sigma) = stats.norm.fit(data)
                n, bins_bc, patches = plt.hist(data, bins)
                y_bc = stats.norm.pdf(bins_bc, mu, sigma)
                bins_bc_list[layer - 1] = bins_bc
                y_bc_list[layer - 1] = y_bc

                mean_grad_bc_list.append(torch.mean(abs(param.grad)))
        
        data_loss.backward(retain_graph=True)
    
        print('Analyzing gradients wrt data loss')
        plt.figure(fig_nr)
        layer = 0
        for name, param in net_u.named_parameters(prefix=''):
            if "weight" in name and layer < nr_layers - 1:
                layer += 1
               
                print(f"Layer: {name} | Size: {param.size()}")
                data = param.grad.cpu().detach().numpy()

                (mu, sigma) = stats.norm.fit(data)
                n, bins_data, patches = plt.hist(data, bins)
                y_data = stats.norm.pdf(bins_data, mu, sigma)

                bins_data_list[layer - 1] = bins_data
                y_data_list[layer - 1] = y_data
                # plt.xlim([-0.1, 0.1])

                mean_grad_data_list.append(torch.mean(abs(param.grad)))

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
        plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "LossGrad_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".png")
        plt.close(420)
        # plt.show()

    def velocity_prediction_post(self, case, geometry_locations, u, v, show=True): # TODO: werkt nog niet goed voor 3D 
        """
        Plots the prediction for the axial velocity (top) and radial velocity (bot) for the current epoch. 

        Parameters: 
        case (class) : The case to which the PINN is applied, contains case-specific information.
        ex_ID (string) : Contains the name of the current experiment, added to the filename during storage. 
        geometry_locations (tensor) : The x, y(, z) coordinates within the geometry. 
        networks (list) : List with all the neural networks (u, v, w and P). 
        epoch (int) : Current epoch, used together with epoch_pretrain to determine the total number of epochs. Used in the title. 
        epoch_pretrain (int) : Amount of epochs the networks have been trained with in case of a pretrained network. 

        
        """
        x, y = geometry_locations 
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.scatter(z, x, c=w, cmap='rainbow') if case.input_dimension == 3 else plt.scatter(x, y, c=u, cmap='rainbow')
        plt.title('NN results, u (top) & v (bot)')
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.scatter(z, x, c=u, cmap='rainbow') if case.input_dimension == 3 else plt.scatter(x, y, c=v, cmap='rainbow')
        plt.colorbar()
        # plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "plotWU_" + ex_ID + "_" + str(epoch + epoch_pretrain) + "_" + str(run)) if case.input_dimension == 3 \
        #     else plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "plotUV_" + ex_ID + "_" + str(epoch + epoch_pretrain) + "_" + str(run))
        if show: 
            plt.show()
        plt.close()

        

    def evo_plot(self, case, run, ex_ID, epoch, batch_idx, boundary_locations, batch_locations, batch_retain, retained_loss, loss_x_retained, loss_y_retained, loss_c_retained, batch_retain_old): 

    
        x = batch_locations[0].cpu().detach().numpy()
        y = batch_locations[1].cpu().detach().numpy()
        xr = batch_retain[0].cpu().detach().numpy()
        yr = batch_retain[1].cpu().detach().numpy()
        xr_old = batch_retain_old[0].cpu().detach().numpy()
        yr_old = batch_retain_old[1].cpu().detach().numpy()

        xr_old2 = np.reshape(xr_old, (1, len(xr_old))).squeeze()
        yr_old2 = np.reshape(yr_old, (1, len(yr_old))).squeeze()


        data = {'X':xr_old2, 'Y': yr_old2}
        df = pd.DataFrame(data)
        #df = np.array([xr_old3, yr_old3]).squeeze()
        fig = px.density_heatmap(df, x='X', y='Y', marginal_x='histogram', marginal_y='histogram', nbinsx=50, nbinsy=50, range_x=[-1,1], range_y=[-0.15,0.15])
        fig.write_image(case.path + "/" + str(run) + "/plots/" + case.ID + "hist2DBatch_" + ex_ID + "_" + str(epoch) + "_" + str(batch_idx) + "_" + str(run) + ".png")
        #fig.show()
        plt.close()

        retained_loss = retained_loss.cpu().detach().numpy()
        xb, yb, *zb = [axis.cpu().detach().numpy() for axis in boundary_locations]
        plt.figure()
        plt.scatter(xb, yb)
        plt.scatter(x, y, s = 80, color='r')
        # plt.scatter(x, y, c=retained_loss)
        plt.scatter(xr, yr, s=40, color='blue')
        plt.scatter(xr_old, yr_old, s=20, color='yellow')
        plt.legend(['', 'batch', 'batch_r_i', 'batch_r_i-1'], loc='lower left')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "scatterBatch_" + ex_ID + "_" + str(epoch) + "_" + str(batch_idx) + "_" + str(run) + ".png")
        #plt.show()
        plt.close()

        loss_x_retained = loss_x_retained.cpu().detach().numpy()
        loss_y_retained = loss_y_retained.cpu().detach().numpy()
        loss_c_retained = loss_c_retained.cpu().detach().numpy()
        mean_x = np.mean(loss_x_retained)
        # ratio = loss_x_retained / loss_y_retained
        ratio_x = abs(loss_x_retained) / retained_loss * 100
        ratio_y = abs(loss_y_retained) / retained_loss * 100
        ratio_c = abs(loss_c_retained) / retained_loss * 100


        plt.figure()
        plt.subplot(121)
        plt.scatter(xb, yb)
        plt.scatter(xr, yr, c=retained_loss, cmap='plasma')
        plt.colorbar()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('NS eq. residuals')
        plt.subplot(122)
        plt.hist(ratio_x, bins=20)
        plt.hist(ratio_y, bins=20)
        plt.hist(ratio_c, bins=20)
        plt.legend(['X', 'Y', 'C'], loc='upper right')
        plt.xlabel('% contribution loss components NS eq.')
        plt.xlim([0, 100])
        plt.ylim([0, 25])
        fig = plt.gcf()
        fig.set_size_inches((11, 8.5), forward=False)
        fig.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "scatterBatchLoss_" + ex_ID + "_" + str(epoch) + "_" + str(batch_idx) + "_" + str(run) + ".png", dpi=500)
        #plt.show()
        plt.close()


        H, xedges, yedges = np.histogram2d(xr_old2, yr_old2, bins=40)
        H = H.T

        # plt.figure()
        # plt.imshow(H, cmap='hot', interpolation='nearest', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "BlocksBatch_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".png")
        # #plt.show()
        # plt.close()

        fig = plt.figure()
        ax=fig.add_subplot(xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
        im = NonUniformImage(ax, interpolation='bilinear')
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        im.set_data(xcenters, ycenters, H)
        ax.images.append(im)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "SprayedBatch_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".png")
        #plt.show()
        plt.close()

        fig2 = plt.figure()
        ax = fig2.add_subplot()
        X, Y = np.meshgrid(xedges, yedges)
        ax.pcolormesh(X, Y, H)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "EdgesBatch_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".png")
        #plt.show()
        plt.close()
       
        print('evo plots made - batch ' + str(batch_idx))

    def hist_batch_retain(self, case, ex_ID, epoch, run, batch_len, batch_size):

        batch_len = np.divide(batch_len, batch_size) * 100

        plt.figure()
        plt.hist(batch_len, bins=40)
        plt.xlabel('% of retained batch')
        plt.ylabel('frequency')
        plt.title('% retained distribution for epoch ' + str(epoch))
        plt.xlim([0, 100])
        plt.ylim([0, 8])
        plt.savefig(case.path + "/" + str(run) + "/plots/" + case.ID + "HistBatch_" + ex_ID + "_" + str(epoch) + "_" + str(run) + ".png")
        #plt.show()
        plt.close()


    def scatter3D_data(self, case, solution_locations, solution_values, show=False): 
        
        #x = solution_locations[0]
        x = solution_locations[0]
        y = solution_locations[1]
        z = solution_locations[2]

        vel_z = solution_values[2]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, c=vel_z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if show:
            plt.show()
        plt.close()
        return 


    def ns_grad_values(self, case, ex_ID, epoch, epoch_pretrain, grad_list, run): 
        """
        
        """
        bins = 50
        #### X
        min_ux = [list[0][0].cpu().detach().numpy() for list in grad_list]
        (mu, sigma) = stats.norm.fit(min_ux)
        _, bins_data_min_ux, patches = plt.hist(min_ux, bins)
        min_ux_data = stats.norm.pdf(bins_data_min_ux, mu, sigma)

        max_ux = [list[0][1].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(max_ux)
        _, bins_data_max_ux, patches = plt.hist(max_ux, bins)
        max_ux_data = stats.norm.pdf(bins_data_max_ux, mu, sigma)

        min_uxx = [list[0][2].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(min_uxx)
        _, bins_data_min_uxx, patches = plt.hist(min_uxx, bins)
        min_uxx_data = stats.norm.pdf(bins_data_min_uxx, mu, sigma)

        max_uxx = [list[0][3].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(max_uxx)
        _, bins_data_max_uxx, patches = plt.hist(max_uxx, bins)
        max_uxx_data = stats.norm.pdf(bins_data_max_uxx, mu, sigma)

        min_vx = [list[1][0].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(min_vx)
        _, bins_data_min_vx, patches = plt.hist(min_vx, bins)
        min_vx_data = stats.norm.pdf(bins_data_min_vx, mu, sigma)

        max_vx = [list[1][1].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(max_vx)
        _, bins_data_max_vx, patches = plt.hist(max_vx, bins)
        max_vx_data = stats.norm.pdf(bins_data_max_vx, mu, sigma)

        min_vxx = [list[1][2].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(min_vxx)
        _, bins_data_min_vxx, patches = plt.hist(min_vxx, bins)
        min_vxx_data = stats.norm.pdf(bins_data_min_vxx, mu, sigma)

        max_vxx = [list[1][3].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(max_vxx)
        _, bins_data_max_vxx, patches = plt.hist(max_vxx, bins)
        max_vxx_data = stats.norm.pdf(bins_data_max_vxx, mu, sigma)

        ######  Y
        min_uy = [list[2][0].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(min_uy)
        _, bins_data_min_uy, patches = plt.hist(min_uy, bins)
        min_uy_data = stats.norm.pdf(bins_data_min_ux, mu, sigma)

        max_uy = [list[2][1].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(max_uy)
        _, bins_data_max_uy, patches = plt.hist(max_uy, bins)
        max_uy_data = stats.norm.pdf(bins_data_max_uy, mu, sigma)

        min_uyy = [list[2][2].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(min_uyy)
        _, bins_data_min_uyy, patches = plt.hist(min_uyy, bins)
        min_uyy_data = stats.norm.pdf(bins_data_min_uyy, mu, sigma)

        max_uyy = [list[2][3].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(max_uyy)
        _, bins_data_max_uyy, patches = plt.hist(max_uyy, bins)
        max_uyy_data = stats.norm.pdf(bins_data_max_uyy, mu, sigma)

        min_vy = [list[3][0].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(min_vy)
        _, bins_data_min_vy, patches = plt.hist(min_vy, bins)
        min_vy_data = stats.norm.pdf(bins_data_min_vy, mu, sigma)

        max_vy = [list[3][1].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(max_vy)
        _, bins_data_max_vy, patches = plt.hist(max_vy, bins)
        max_vy_data = stats.norm.pdf(bins_data_max_vy, mu, sigma)

        min_vyy = [list[3][2].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(min_vyy)
        _, bins_data_min_vyy, patches = plt.hist(min_vyy, bins)
        min_vyy_data = stats.norm.pdf(bins_data_min_vyy, mu, sigma)

        max_vyy = [list[3][3].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(max_vyy)
        _, bins_data_max_vyy, patches = plt.hist(max_vyy, bins)
        max_vyy_data = stats.norm.pdf(bins_data_max_vyy, mu, sigma)

        ####### P
        min_px = [list[4][0].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(min_px)
        _, bins_data_min_px, patches = plt.hist(min_px, bins)
        min_px_data = stats.norm.pdf(bins_data_min_px, mu, sigma)

        max_px = [list[4][1].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(max_px)
        _, bins_data_max_px, patches = plt.hist(max_px, bins)
        max_px_data = stats.norm.pdf(bins_data_max_px, mu, sigma)

        min_py = [list[4][2].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(min_py)
        _, bins_data_min_py, patches = plt.hist(min_py, bins)
        min_py_data = stats.norm.pdf(bins_data_min_py, mu, sigma)

        max_py = [list[4][3].item() for list in grad_list]
        (mu, sigma) = stats.norm.fit(max_py)
        _, bins_data_max_py, patches = plt.hist(max_py, bins)
        max_py_data = stats.norm.pdf(bins_data_max_py, mu, sigma)

        plt.figure()
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
  