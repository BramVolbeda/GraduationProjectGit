import matplotlib.pyplot as plt

class Figures: 
    def __init__(self, inplace=True):
        super(Figures, self).__init__()
        self.inplace = inplace
    
    def scaled_geometry(self, boundary_locations, solution_locations, solution_values, show=False): 
        
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
        # plt.savefig(case.path + "/plots/" + case.ID + "scaling" + ex_ID)
        if show: 
            plt.show()

        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.scatter(zb_cross, yb_cross) if case.input_n == 3 else plt.scatter(xb, yb)
        # plt.scatter(zd, xd, c=wd, cmap='autumn_r') if case.input_n == 3 else plt.scatter(xd, yd, c=ud, cmap='autumn_r')
        # plt.title('scaled data points, u (top) & v (bot)')
        # plt.colorbar()
        # # plt.savefig(case.path + "/plots/plotU" + ex_ID)
        # plt.subplot(2, 1, 2)
        # plt.scatter(zb_cross, yb_cross) if case.input_n == 3 else plt.scatter(xb, yb)
        # plt.scatter(zd, xd, c=ud, cmap='autumn_r') if case.input_n == 3 else plt.scatter(xd, yd, c=vd, cmap='autumn_r')
        # plt.colorbar()
        # plt.savefig(case.path + "/plots/" + case.ID + "scaling" + ex_ID)
        # # plt.show()

