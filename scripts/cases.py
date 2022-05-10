import os

class stenose2D:
    def __init__(self, inplace=True):
        super(stenose2D, self).__init__()
        self.inplace = inplace

        self.name = "stenose2D"
        self.input_n = 2
        self.h_n = 128
        self.path = "./results/" + self.name  # where results are saved
        self.directory = "./data/2D-stenosis/"
        self.mesh_file = self.directory + "2DSTEN_meshS.vtu"
        self.bc_file = self.directory + "2DSTEN_bncS.vtk"
        self.vel_file = self.directory + "2DSTEN_solutionRe150.vtu"
        self.ID = "STEN2D_"

        self.X_scale = 2.0 # 3.0
        self.Y_scale = 0.5
        self.Z_scale = 1.0  # 2D, but for generalizing script still needed
        self.U_scale = 10.
        self.Diff = 0.001
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'velocity'  # The velocity field name in the vtk file (see from ParaView)
        self.Lambda_BC = 16
        self.Lambda_data = 1
        #self.ml_name = '2Dstenosis.h5'
        #self.output_filename = self.path + "outputs/" + self.pretrain + "Own_test2.vtk"
        self.x_data = [1., 1.2, 1.22, 1.31, 1.39]
        self.y_data = [0.0, -0.08, 0.07, -0.114, 0.11]
        self.z_data = [0., 0., 0., 0., 0.]

class stenose2D_Arzani:
    def __init__(self, inplace=True):
        super(stenose2D_Arzani, self).__init__()
        self.inplace = inplace

        self.name = "stenose2D"
        self.input_n = 2
        self.h_n = 128
        self.path = "./results/" + self.name  # where results are saved
        self.directory = "./data/2D-stenosis/"
        self.mesh_file = self.directory + "sten_mesh000000.vtu"
        self.bc_file = self.directory + "wall_BC.vtk"
        self.vel_file = self.directory + "velocity_sten_steady.vtu"
        self.ID = "STEN2DA_"

        self.X_scale = 2.0  # 3  TODO schaling consequent maken:  / of *
        self.Y_scale = 1.0  # 1
        self.Z_scale = 1.0  # 2D, but for generalizing script still needed
        self.U_scale = 1.0
        self.Diff = 0.001
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'f_5-0'  # The velocity field name in the vtk file (see from ParaView)
        # self.Lambda_BC = 20
        self.Lambda_BC = 1.
        self.Lambda_data = 1.
        # self.output_filename = self.path + "outputs/" + self.ID + "Arzani_test3.vtk" # TODO niet universeel over cases
        self.x_data = [1., 1.2, 1.22, 1.31, 1.39]
        self.y_data = [0.15, 0.07, 0.22, 0.036, 0.26]
        self.z_data = [0., 0., 0., 0., 0.]


class aneurysm2D:
    def __init__(self, inplace=True):
        super(aneurysm2D, self).__init__()
        self.inplace = inplace

        self.name = "aneurysm2D"
        self.input_n = 2
        self.h_n = 128
        self.path = "./results/" + self.name  # where results are saved
        self.directory = "./data/2D-aneurysm/"
        self.mesh_file = self.directory + "IA_mesh_correct_crop.vtu"
        self.bc_file = self.directory + "wall_BC_crop_correct.vtk"
        self.vel_file = self.directory + "velocity_IA_steady.vtu"
        self.ID = "IA2D_"

        self.X_scale = 3.0
        self.Y_scale = 2.0
        self.Z_scale = 1.0  # 2D, but for generalizing script still needed
        self.U_scale = 1.0
        self.Diff = 0.00125  # Such that Re = 320
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'f_5-0'  # The velocity field name in the vtk file (see from ParaView)
        self.Lambda_BC = 20
        self.Lambda_data = 1
        self.ml_name = '2Daneurysm.h5'
        # self.output_filename = self.path + "Outputs/" + self.NNprefix + ".vtk"
        self.radius = 0.5
        self.xM = 1.8
        self.yM = 0.7
        self.theta = [-0.60, 3.70]
        self.x_data = [1.8, 2.0, 1.5, 1.75, 2.1]
        self.y_data = [0.4, 0.5, 0.2, 0.9, 0.75]
        self.z_data = [0., 0., 0., 0., 0.]

class aneurysm3D:
    def __init__(self, inplace=True):
        super(aneurysm3D, self).__init__()
        self.inplace = inplace

        self.name = "aneurysm3D"
        self.input_n = 3
        self.h_n = 200
        self.path = "./results/" + self.name  # where results are saved
        self.directory = "./data/3D-aneurysm/"
        self.mesh_file = self.directory + "IA_mesh3D_nearwall_small_physical.vtu"
        self.bc_file = self.directory + "IA_nearwall_outer_small.vtk"
        self.bc_file_inner = self.directory + "IA_nearwall_wall_small.vtk"
        self.vel_file = self.directory + "IA_3D_unsteady3.vtu"
        self.ID = "IA3D_"

        self.X_scale = 3.0
        self.Y_scale = 2.0
        self.Z_scale = 2.0
        self.U_scale = 1.0
        self.Diff = 0.00125  # Such that Re = 320
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'f_17'  # The velocity field name in the vtk file (see from ParaView)
        self.Lambda_BC = 20
        self.Lambda_data = 1
        # self.output_filename = self.path + "Outputs/" + self.NNprefix + "test2.vtk"
        self.radius = 0.4
        self.xM = 1.8
        self.yM = 0.7
        self.theta = [-0.8, 3.95]
        self.x_data = [1., 1.2, 1.22, 1.31, 1.39 ]
        self.y_data =[0.15, 0.07, 0.22, 0.036, 0.26 ]
        self.z_data  = [0.,0.,0.,0.,0. ]

class stenose2DRescale:
    def __init__(self, inplace=True):
        super(stenose2DRescale, self).__init__()
        self.inplace = inplace

        self.input_n = 2
        self.h_n = 128
        # self.path = "C:/Users\s163213\OneDrive\Graduation project\Code\Results/custom/"  # where results are saved
        self.cd = os.getcwd()
        # self.directory = "C:/Users/s163213/OneDrive/Graduation project/Code/PINN/Data/custom/"
        self.path = self.cd + "/results/"
        self.directory = self.cd + "/data/2D-stenosis/"
        self.mesh_file = self.directory + "sten_mesh000000.vtu"
        self.bc_file = self.directory + "wall_BC.vtk"
        self.vel_file = self.directory + "velocity_sten_steady.vtu"
        self.name = "stenose2D"
        self.pretrain = "sten_"

        self.X_scale = 3.0
        self.Y_scale = 2.0
        self.Z_scale = 2.0
        self.U_scale = 1.0
        self.Diff = 0.00125  # Such that Re = 320
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'f_5-0'  # The velocity field name in the vtk file (see from ParaView)
        self.Lambda_BC = 20
        self.Lambda_data = 1
        self.ml_name = '2DstenosisRescale.h5'
        self.x_data = [1., 1.2, 1.22, 1.31, 1.39]
        self.y_data = [0.15, 0.07, 0.22, 0.036, 0.26]
        self.z_data = [0., 0., 0., 0., 0.]

class NOZ3D:
    def __init__(self, inplace=True):
        super(NOZ3D, self).__init__()
        self.inplace = inplace

        self.name = "nozzle3D"
        self.input_n = 3
        self.h_n = 200
        # self.path = "C:/Users\s163213\OneDrive\Graduation_project\Code\Results/3Daneurysm/"  # where results are saved
        self.path = "./results/" + self.name  # where results are saved
        # self.directory = "C:/Users/s163213/OneDrive/Graduation_project/Code/PINN/Data/3D-aneurysm/"
        self.directory = './data/3DNOZ/'
        self.mesh_file = self.directory + "3DNOZ_meshS.vtu"
        self.bc_file = self.directory + "3DNOZ_bncS.vtk"
        # self.bc_file_inner = self.directory + "IA_nearwall_wall_small.vtk"
        self.vel_file = self.directory + "3DNOZ_solRe285.vtu"
        self.ID = "NOZ3D_"

        self.X_scale = 1
        self.Y_scale = 1
        self.Z_scale = 5
        self.U_scale = 1
        self.Diff = 0.00035  # Such that Re = 10
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'velocity_scaled'  # The velocity field name in the vtk file (see from ParaView)
        self.Lambda_BC = 20
        self.Lambda_data = 1
        # self.output_filename = self.path + "outputs/" + self.ID + "test.vtk"
        # self.radius = 0.4
        # self.xM = 1.8
        # self.yM = 0.7
        # self.theta = [-0.8, 3.95]
        self.x_data = [0., 0.1, 0., 0.07, 0.02, 0.1, 0.003, 0.075] # data points in original geometry
        self.y_data = [0., 0.1, 0., 0.07, 0.02, 0.1, 0.003, 0.075]
        self.z_data = [2.56, 2.6, 2.7, 2.8, 2.9, 3, 3.2, 3.3]

class NOZ2D:
    def __init__(self, inplace=True):
        super(NOZ2D, self).__init__()
        self.inplace = inplace

        self.name = "nozzle2D"
        self.input_n = 2
        self.h_n = 200
        # self.path = "C:/Users\s163213\OneDrive\Graduation_project\Code\Results/3Daneurysm/"  # where results are saved
        self.path = "./results/" + self.name  # where results are saved
        # self.directory = "C:/Users/s163213/OneDrive/Graduation_project/Code/PINN/Data/3D-aneurysm/"
        self.directory = './data/2DNOZ/'
        self.mesh_file = self.directory + "mesh2D_scaled.vtu"
        self.bc_file = self.directory + "bnc_mesh2D_scaled.vtk"
        # self.bc_file_inner = self.directory + "IA_nearwall_wall_small.vtk"
        self.vel_file = self.directory + "sol_Re100.vtu"
        self.ID = "NOZ2D_"

        self.X_scale = 1/5.0  # TODO: scale is over het algemeen / ipv *
        self.Y_scale = 1/25.0
        self.Z_scale = 1.0
        # self.U_scale = 20.
        self.U_scale = 160.
        # self.X_scale = 5.
        # self.Y_scale = 25.
        # self.Z_scale = 1.
        # self.U_scale = 1.
        self.Diff = 0.035  # Such that Re = 10 # TODO uitzoeken hoe dit precies werkt
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'velocity'  # The velocity field name in the vtk file (see from ParaView)
        self.Lambda_BC = 20
        self.Lambda_data = 1
        self.ml_name = '2DNOZ.h5'
        # self.x_data = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]  # data points within original domain
        # self.y_data = [-0.001, 0.003, 0.001, 0, 0.0005, -0.005, 0.004, -0.0008]
        # self.z_data = [0, 0, 0, 0, 0, 0, 0, 0]
        # self.x_data = [0.5, 0.55, 0.575, 0.585, 0.585, 0.585, 0.65, 0.65]  # data points within original domain
        # self.x_data[:] = [x * self.X_scale for x in self.x_data]
        # self.y_data = [0, -0.05, 0.05, -0.1, 0.1, 0, 0.03, -0.025]
        # self.y_data[:] = [y * self.Y_scale for y in self.y_data]
        # self.z_data = [0, 0, 0, 0, 0, 0, 0, 0]
        # self.z_data[:] = [z * self.Z_scale for z in self.z_data]
        self.x_data = [0.1, 0.11, 0.115, 0.117, 0.117, 0.117, 0.13, 0.13]
        self.y_data = [0., -0.002, 0.002, -0.004, 0.004, 0., 0.0012, -0.001]
        self.z_data = [0, 0, 0, 0, 0, 0, 0, 0]

class NOZ2D_R:
    def __init__(self, inplace=True):
        super(NOZ2D_R, self).__init__()
        self.inplace = inplace

        self.name = "nozzle2D-R"
        self.input_n = 2
        self.h_n = 200
        # self.path = "C:/Users\s163213\OneDrive\Graduation_project\Code\Results/3Daneurysm/"  # where results are saved
        self.path = "./results/" + self.name  # where results are saved
        # self.directory = "C:/Users/s163213/OneDrive/Graduation_project/Code/PINN/Data/3D-aneurysm/"
        self.directory = './data/2DNOZ/'
        self.mesh_file = self.directory + "2DNOZ-R_meshS.vtu"
        self.bc_file = self.directory + "2DNOZ-R_bncS.vtk"
        # self.bc_file_inner = self.directory + "IA_nearwall_wall_small.vtk"
        self.vel_file = self.directory + "2DNOZ-R_solutionRe100.vtu"
        self.ID = "NOZ2D_R_"

        self.X_scale = 1/5.0  # TODO: scale is over het algemeen / ipv *
        self.Y_scale = 1/25.0
        self.Z_scale = 1.0
        self.U_scale = 20.
        # self.X_scale = 5.
        # self.Y_scale = 25.
        # self.Z_scale = 1.
        # self.U_scale = 1.
        self.Diff = 0.035  # Such that Re = 10 # TODO uitzoeken hoe dit precies werkt
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'velocity'  # The velocity field name in the vtk file (see from ParaView)
        self.Lambda_BC = 20
        self.Lambda_data = 1
        self.ml_name = '2DNOZ-R.h5'
        # self.output_filename = self.path + "outputs/" + self.ID + "test2.vtk"
        # self.radius = 0.4
        # self.xM = 1.8
        # self.yM = 0.7
        # self.theta = [-0.8, 3.95]
        # self.x_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # data points within original domain
        # self.y_data = [-0.1, 0.1, 0.01, 0, 0.05, -0.05, 0.12, -0.08]
        # self.z_data = [0, 0, 0, 0, 0, 0, 0, 0]
        self.x_data = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]  # data points within original domain
        self.y_data = [-0.001, 0.003, 0.001, 0, 0.0005, -0.005, 0.004, -0.0008]
        self.z_data = [0, 0, 0, 0, 0, 0, 0, 0]

class NOZ2D_G:
    def __init__(self, inplace=True):
        super(NOZ2D_G, self).__init__()
        self.inplace = inplace

        self.name = "nozzle2D-G"
        self.input_n = 2
        self.h_n = 200
        # self.path = "C:/Users\s163213\OneDrive\Graduation_project\Code\Results/3Daneurysm/"  # where results are saved
        self.path = "./results/" + self.name  # where results are saved
        # self.directory = "C:/Users/s163213/OneDrive/Graduation_project/Code/PINN/Data/3D-aneurysm/"
        self.directory = './data/2DNOZ/'
        self.mesh_file = self.directory + "2DNOZ-G_meshS.vtu"
        self.bc_file = self.directory + "2DNOZ-G_bncS.vtk"
        # self.bc_file_inner = self.directory + "IA_nearwall_wall_small.vtk"
        self.vel_file = self.directory + "2DNOZ-G_solutionRe100.vtu"
        self.ID = "NOZ2D_G_"

        self.X_scale = 1/5.0  # TODO: scale is over het algemeen / ipv *
        self.Y_scale = 1/25.0
        self.Z_scale = 1.0
        self.U_scale = 20.
        # self.X_scale = 5.
        # self.Y_scale = 25.
        # self.Z_scale = 1.
        # self.U_scale = 1.
        self.Diff = 0.035  # Such that Re = 10 # TODO uitzoeken hoe dit precies werkt
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'velocity'  # The velocity field name in the vtk file (see from ParaView)
        self.Lambda_BC = 20
        self.Lambda_data = 1
        self.ml_name = '2DNOZ-G.h5'
        # self.output_filename = self.path + "outputs/" + self.ID + "test2.vtk"
        # self.radius = 0.4
        # self.xM = 1.8
        # self.yM = 0.7
        # self.theta = [-0.8, 3.95]
        # self.x_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # data points within original domain
        # self.y_data = [-0.1, 0.1, 0.01, 0, 0.05, -0.05, 0.12, -0.08]
        # self.z_data = [0, 0, 0, 0, 0, 0, 0, 0]
        self.x_data = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]  # data points within original domain
        self.y_data = [-0.001, 0.003, 0.001, 0, 0.0005, -0.005, 0.004, -0.0008]
        self.z_data = [0, 0, 0, 0, 0, 0, 0, 0]


class NOZ2D_L:
    def __init__(self, inplace=True):
        super(NOZ2D_L, self).__init__()
        self.inplace = inplace

        self.name = "nozzle2D-L"
        self.input_n = 2
        self.h_n = 200
        # self.path = "C:/Users\s163213\OneDrive\Graduation_project\Code\Results/3Daneurysm/"  # where results are saved
        self.path = "./results/" + self.name  # where results are saved
        # self.directory = "C:/Users/s163213/OneDrive/Graduation_project/Code/PINN/Data/3D-aneurysm/"
        self.directory = './data/2DNOZ/'
        self.mesh_file = self.directory + "2DNOZ-L_meshS.vtu"
        self.bc_file = self.directory + "2DNOZ-L_bncS.vtk"
        # self.bc_file_inner = self.directory + "IA_nearwall_wall_small.vtk"
        self.vel_file = self.directory + "2DNOZ-L_solutionRe100.vtu"
        self.ID = "NOZ2D_L_"

        self.X_scale = 2.  # TODO: scale is over het algemeen / ipv *
        self.Y_scale = 1.
        self.Z_scale = 1.
        self.U_scale = 0.5
        # self.X_scale = 5.
        # self.Y_scale = 25.
        # self.Z_scale = 1.
        # self.U_scale = 1.
        self.Diff = 0.001  # Such that Re = 10 # TODO uitzoeken hoe dit precies werkt
        self.rho = 1.
        self.Lambda_div = 1.  # penalty factor for continuity eqn (Makes it worse!?)
        self.Lambda_v = 1.  # penalty factor for y-momentum equation
        self.fieldname = 'velocity'  # The velocity field name in the vtk file (see from ParaView)
        self.Lambda_BC = 20
        self.Lambda_data = 1
        self.ml_name = '2DNOZ-L.h5'
        # self.output_filename = self.path + "outputs/" + self.ID + "test2.vtk"
        # self.radius = 0.4
        # self.xM = 1.8
        # self.yM = 0.7
        # self.theta = [-0.8, 3.95]
        # self.x_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # data points within original domain
        # self.y_data = [-0.1, 0.1, 0.01, 0, 0.05, -0.05, 0.12, -0.08]
        # self.z_data = [0, 0, 0, 0, 0, 0, 0, 0]
        self.x_data = [1.2, 1.4, 1.42, 1.51, 1.59]
        self.y_data = [0.0, -0.08, 0.07, -0.114, 0.11]
        self.z_data = [0., 0., 0., 0., 0.]