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
        self.mesh_file = self.directory + "2DSTEN_meshResX2_C.vtu"
        self.bc_file = self.directory + "2DSTEN_bnc_T.vtk"
        self.vel_file = self.directory + "2DSTEN_solutionRe150_T.vtu"
        self.ID = "PINN3_STEN2D_"

        self.X_scale = 2.0 # 3.0
        self.Y_scale = 1.0
        self.Z_scale = 1.0  # 2D, but for generalizing script still needed
        self.U_scale = 10.
        self.L_scale = 0.3  # Diameter inlet 
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
        self.x_data = [0., 0.2, 0.22, 0.31, 0.39]
        self.y_data = [0.0, -0.08, 0.07, -0.114, 0.11]
        self.z_data = [0., 0., 0., 0., 0.]

class stenose2D_Arzani:
    def __init__(self, inplace=True):
        super(stenose2D_Arzani, self).__init__()
        self.inplace = inplace

        self.name = "stenose2DA"
        self.input_dimension = 2
        # self.h_n = 128
        self.path = "./results/" + self.name  # where results are saved
        self.directory = "./data/2DSTENA/"
        self.mesh_file = self.directory + "sten_mesh000000.vtu"
        self.bc_file = self.directory + "wall_BC.vtk"
        self.vel_file = self.directory + "velocity_sten_steady.vtu"
        self.ID = "STEN2DA_"

        self.X_scale = 2.0  # 3  TODO schaling consequent maken:  / of *
        self.L_scale = 0.3  # Diameter inlet 
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
        self.x_data = [0., 0.2, 0.22, 0.31, 0.39]
        self.y_data = [0., -0.08, 0.07, -0.114, 0.11]
        self.z_data = [0., 0., 0., 0., 0.]
        # self.x_data = [0.5, 0.6, 0.61, 0.65, 0.7]
        # self.y_data = [0.075, 0.035, 0.11, 0.018, 0.13]

        
class TUBE2D:
    def __init__(self, inplace=True):
        super(TUBE2D, self).__init__()
        self.inplace = inplace

        self.name = "tube2D"
        self.input_dimension = 2
        # self.h_n = 10 # 128
        self.path = "./results/" + self.name  # where results are saved
        self.directory = "./data/2DTUBE/"
        self.mesh_file = self.directory + "2DTUBE_mesh_C.vtu"
        self.bc_file = self.directory + "2DTUBE_bnc_C.vtk"
        self.vel_file = self.directory + "2DTUBE_solutionRe150_C.vtu"
        self.ID = "PINN3_TUBE2D_"

        self.X_scale = 2.0 # 3.0
        self.Y_scale = 0.5
        self.Z_scale = 1.0  # 2D, but for generalizing script still needed
        self.U_scale = 10.
        self.L_scale = 0.3  # Diameter inlet 
        self.vmax = 0.5
        self.center = 0  # y coordinate of centerline
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
        self.x_data = [-0.8, -0.4, 0, 0.4, 0.8]
        self.y_data = [0., 0., 0., 0., 0.]
        # self.x_data = [0., 0.2, 0.22, 0.31, 0.39, 0.1]
        # self.y_data = [0.0, -0.08, 0.07, -0.114, 0.11, 0.]
        self.z_data = [0., 0., 0., 0., 0., 0.]

class TUBE3D:
    def __init__(self, inplace=True):
        super(TUBE3D, self).__init__()
        self.inplace = inplace

        self.name = "tube3D"
        self.input_dimension = 3
        # self.h_n = 10 # 128
        self.path = "./results/" + self.name  # where results are saved
        self.directory = "./data/3DTUBE/"
        self.mesh_file = self.directory + "3DTUBE_mesh_CS.vtu"
        self.bc_file = self.directory + "3DTUBE_bnc_CS.vtk"
        self.vel_file = self.directory + "3DTUBE_solutionRe1_500K_CS.vtu"
        self.ID = "PINN3_TUBE3D_"

        self.X_scale = 2.0 # 3.0
        self.Y_scale = 0.5
        self.Z_scale = 1.0  # 2D, but for generalizing script still needed
        self.U_scale = 10.
        self.L_scale = 0.3  # Diameter inlet 
        self.vmax = 0.5
        self.center = 0  # y coordinate of centerline
        self.Reynolds = 1. 
        self.Diff = 0.001
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'velocity'  # The velocity field name in the vtk file (see from ParaView)
        self.lambda_bc = 16
        self.lambda_data = 1
        #self.ml_name = '2Dstenosis.h5'
        #self.output_filename = self.path + "outputs/" + self.pretrain + "Own_test2.vtk"
        self.x_data = [-0.8, -0.4, 0, 0.4, 0.8]
        self.y_data = [0., 0., 0., 0., 0.]
        # self.x_data = [0., 0.2, 0.22, 0.31, 0.39, 0.1]
        # self.y_data = [0.0, -0.08, 0.07, -0.114, 0.11, 0.]
        self.z_data = [0., 0.1, 0., 0.2, 0.]