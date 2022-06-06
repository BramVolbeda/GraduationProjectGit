import os

class stenose2D:
    def __init__(self, inplace=True):
        super(stenose2D, self).__init__()
        self.inplace = inplace

        self.name = "stenose2D"
        self.input_dimension = 2
        # self.h_n = 10 # 128
        self.path = "./results/" + self.name  # where results are saved
        self.directory = "./data/2DSTEN/"
        self.mesh_file = self.directory + "2DSTEN_mesh.vtu"
        self.bc_file = self.directory + "2DSTEN_bnc.vtk"
        self.vel_file = self.directory + "2DSTEN_solutionRe150.vtu"
        self.ID = "STEN2D_"

        self.X_scale = 2.0 # 3.0
        self.Y_scale = 0.5
        self.Z_scale = 1.0  # 2D, but for generalizing script still needed
        self.U_scale = 10.
        self.Reynolds = 150. 
        self.Diff = 0.001
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'velocity'  # The velocity field name in the vtk file (see from ParaView)
        self.lambda_bc = 16
        self.lambda_data = 1
        #self.ml_name = '2Dstenosis.h5'
        #self.output_filename = self.path + "outputs/" + self.pretrain + "Own_test2.vtk"
        self.x_data = [1., 1.2, 1.22, 1.31, 1.39]
        self.y_data = [0.0, -0.08, 0.07, -0.114, 0.11]
<<<<<<< Updated upstream
        self.z_data = [0., 0., 0., 0., 0.]
=======
        self.z_data = [0., 0., 0., 0., 0.]

class stenose2D_Arzani:
    def __init__(self, inplace=True):
        super(stenose2D_Arzani, self).__init__()
        self.inplace = inplace

        self.name = "stenose2DA"
        self.input_dimension = 2
        # self.h_n = 128
        self.path = "./results/" + self.name  # where results are saved
        self.directory = "./data/2DSTEN/"
        self.mesh_file = self.directory + "sten_mesh000000_T.vtu"
        self.bc_file = self.directory + "wall_BC_T.vtk"
        self.vel_file = self.directory + "velocity_sten_steady_T.vtu"
        self.ID = "STEN2DA_"

        self.X_scale = 2.0  # 3  TODO schaling consequent maken:  / of *
        self.Y_scale = 1.0  # 1
        self.Z_scale = 1.0  # 2D, but for generalizing script still needed
        self.U_scale = 1.0
        self.Reynolds = 150  # Based on peak velocity
        self.Diff = 0.001
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'f_5-0'  # The velocity field name in the vtk file (see from ParaView)
        # self.Lambda_BC = 20
        self.lambda_bc = 1.
        self.lambda_data = 1.
        # self.output_filename = self.path + "outputs/" + self.ID + "Arzani_test3.vtk" # TODO niet universeel over cases
        self.x_data = [1., 1.2, 1.22, 1.31, 1.39]
        self.y_data = [0.15, 0.07, 0.22, 0.036, 0.26]
        self.z_data = [0., 0., 0., 0., 0.]
        # self.x_data = [0.5, 0.6, 0.61, 0.65, 0.7]
        # self.y_data = [0.075, 0.035, 0.11, 0.018, 0.13]
        
>>>>>>> Stashed changes
