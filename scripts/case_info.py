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
        self.z_data = [0., 0., 0., 0., 0.]