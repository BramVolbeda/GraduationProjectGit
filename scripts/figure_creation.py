import matplotlib.pyplot as plt
import numpy as np
import torch

class Figures: 
    def __init__(self, inplace=True):
        super(Figures, self).__init__()
        self.inplace = inplace
    
    def scaled_geometry(self, boundary_locations, solution_locations, prediction_values, show=False): 
        
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.scatter(boundary_locations[0], boundary_locations[1])  # xb, yb
        plt.scatter(solution_locations[0], solution_locations[1], c=prediction_values[0], cmap='autumn_r')
        plt.title('scaled data points, u (top) and v (bot')
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.scatter(boundary_locations[0], boundary_locations[1])
        plt.scatter(solution_locations[0], solution_locations[1], c=prediction_values[1], cmap='autumn_r')
        plt.colorbar()
        # plt.savefig(case.path + "/plots/" + case.ID + "scaling" + ex_ID)
        if show: 
            plt.show()
        plt.close()


    def loss_plot(self, loss_list, epoch, epoch_pretrain, show=False): 

        steps = np.linspace(epoch_pretrain, epoch_pretrain+epoch, len(loss_list[0]))
        colors = ['b', 'g', 'r']
        plt.figure()
        for i in range(len(loss_list)):
            plt.plot(steps, loss_list[i], colors[i])
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend(['eq', 'bc', 'data'])
        # plt.title(case.name)
        if show:
            plt.show()
        # plt.savefig(case.path + "/plots/" + case.ID + "loss_plot_" + ex_ID + ".png")
        plt.close()


    def velocity_prediction(self, geometry_locations, networks, epochs, show=False): # TODO: werkt nog niet goed voor 3D 
        
        input_network_geo = torch.cat(([axis for axis in geometry_locations]), 1)
        prediction_values = [network(input_network_geo).view(len(input_network_geo), -1) for network in networks]
        
        u, v, *w = [solution.cpu().data.numpy() for solution in prediction_values]
        x, y, *z = [axis.cpu() for axis in input_network_geo]
    
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.scatter(z.detach().numpy(), x.detach().numpy(), c=w, cmap='rainbow') if case.input_n == 3 \
            else plt.scatter(x.detach().numpy(), y.detach().numpy(), c=u, cmap='rainbow')
        plt.title('NN results, u (top) & v (bot), - epoch' + str(epochs))
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.scatter(z.detach().numpy(), x.detach().numpy(), c=u, cmap='rainbow') if case.input_n == 3 \
                else plt.scatter(x.detach().numpy(), y.detach().numpy(), c=v, cmap='rainbow')
        plt.colorbar()
        # plt.savefig(case.path + "/plots/" + case.ID + "plotWU_" + ex_ID + "_" + str(epoch + epoch_pretrain)) if case.input_n == 3 \
        #     else plt.savefig(case.path + "/plots/" + case.ID + "plotUV_" + ex_ID + "_" + str(epoch + epoch_pretrain))
        if show: 
            plt.show()
        plt.close()
        

    def data_error(self, boundary_locations, solution_locations, solution_values, networks, epochs, show=False):

        input_network_data = torch.cat(([axis for axis in solution_locations]), 1)
        prediction_values = [network(input_network_data).view(len(input_network_data), -1) for network in networks]

        u, v, *w = [solution.cpu().data.numpy() for solution in prediction_values]
        ud, vd, *wd = solution_values
        print(ud.size())

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
        plt.title('% error; ud (top) & vd (bot), - epoch' + str(epochs))
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.scatter(xb, yb)
        plt.scatter(xd, yd, c=vd_diff, cmap='rainbow')
        plt.colorbar()
        # plt.savefig(case.path + "/plots/" + case.ID + "plotD_UV_" + ex_ID + "_" + str(epoch + epoch_pretrain))
        if show:
            plt.show()
        plt.close()
            