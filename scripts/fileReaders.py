import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy  # PyCharm compatible code
import vtk
import os
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

def AppendObserver(case, ex):
    ml_name = case.ID
    observer_path = './results/' + ml_name + '/'
    ex.observers.append(MongoObserver(url='mongodb+srv://dbBram:t42F7C828K!!@experiments.aakys.mongodb.net'
                                          '/test_stenose?ssl=true&ssl_cert_reqs=CERT_NONE',
                                      db_name=case.ID))
    # configure local observer
    if not os.path.exists(observer_path):
        os.mkdir(observer_path)
        os.mkdir(os.path.join(observer_path, 'sacred'))
    ex.observers.append(FileStorageObserver(basedir=os.path.join(observer_path, 'sacred')))
    print("progress will be saved")
    return

def fileReader(file, input_n, mesh):
    #classes van maken?
    print('Loading', file)  # mesh_file komt overeen met .vtu file die mesh moet geven
    if mesh:
        reader = vtk.vtkXMLUnstructuredGridReader() # Get the Reader's output
    # elif input_n == 3:
    #     reader = vtk.vtkPolyDataReader()
    else:
        reader = vtk.vtkUnstructuredGridReader()  # for the boundary conditions

    reader.SetFileName(file) # Sets file name if exported
    reader.Update() # Fail-save mechanisme? Zeker weten dat de SetFileName werkt?
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints() # n_points 39238 van sten_mesh
    print('n_points of the mesh:', n_points)
    x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    z_vtk_mesh = np.zeros((n_points, 1))
    VTKpoints = vtk.vtkPoints() #Represent and manipulate 3D points
    for i in range(n_points): # voor alle punten (3D, i.e. [0,0.3,0] in de mesh, schrijf weg naar x array en y array. Kan hier ook z
        # aan toevoegen door pt_iso[2] weg te gaan schrijven
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = pt_iso[0] # x_coordinaat van punt uit sten_mesh file
        y_vtk_mesh[i] = pt_iso[1] # y_coordinaat van punt uit sten_mesh file
        z_vtk_mesh[i] = pt_iso[2]
        VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2]) #voegt ook lijnsegmenten toe aan aanliggende punten

    point_data = vtk.vtkUnstructuredGrid() #dataset representeert arbitraire combinatie van mogelijke celtypes
    point_data.SetPoints(VTKpoints) # niet zeker, maar vermoed dat alle mesh_punten hier toegevoegd worden na de for loop hierboven

    x = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh[:]), 1)) # (n_points,1)
    y = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1)) # (n_points,1)
    z = np.reshape(z_vtk_mesh, (np.size(z_vtk_mesh[:]), 1))

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)

    return x, y, z, data_vtk
def dataReader(case, random_flag=False):
    # !!specify pts location here:

    x_data = np.asarray(case.x_data)  # convert to numpy
    y_data = np.asarray(case.y_data)  # convert to numpy
    z_data = np.asarray(case.z_data)
    print(x_data, y_data)
    print('Loading', case.vel_file) # mesh met X_scale = 2, velocity_sten_steady (is dit dan al de oplossing van 2D stenose?)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(case.vel_file)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints() # 21140
    print('n_points of the data file read:', n_points)

    if random_flag:
        N_sample = int(input("Please specify the amount of data points randomly sampled from the geometry"))
        sample_idx = n_points / N_sample
        print('Sampling', N_sample, 'points from a total of', n_points, 'points')
        x_vtk_mesh = np.zeros((n_points, 1))
        y_vtk_mesh = np.zeros((n_points, 1))
        z_vtk_mesh = np.zeros((n_points, 1))
        VTKpoints = vtk.vtkPoints()
        N_pts_data = 0
        for i in range(n_points):
            if i % sample_idx == 0:
                pt_iso = data_vtk.GetPoint(i)
                x_vtk_mesh[N_pts_data] = pt_iso[0]
                y_vtk_mesh[N_pts_data] = pt_iso[1]
                z_vtk_mesh[N_pts_data] = pt_iso[2]
                N_pts_data += 1

        print('n_points sampled', N_pts_data)
        x_data = np.zeros((N_pts_data, 1))
        y_data = np.zeros((N_pts_data, 1))
        z_data = np.zeros((N_pts_data, 1))

        x_data[:, 0] = x_vtk_mesh[0:N_pts_data, 0]
        y_data[:, 0] = y_vtk_mesh[0:N_pts_data, 0]
        z_data[:, 0] = z_vtk_mesh[0:N_pts_data, 0]


    VTKpoints = vtk.vtkPoints()
    for i in range(len(x_data)):
        VTKpoints.InsertPoint(i, x_data[i], y_data[i], z_data[i]) # anders dan voorheen; hier worden de arbitraire
        # gegenereerde datapunten ingevoerd, 'sensor'

    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints) # bouw een unstructured grid voor deze random input punten

    probe = vtk.vtkProbeFilter() # sample data values at specified point locations --> sensor punten
    probe.SetInputData(point_data)
    probe.SetSourceData(data_vtk)
    probe.Update()
    array = probe.GetOutput().GetPointData().GetArray(case.fieldname) # vtkFloatArray size tuple (5,3)
    # data_vel = vtkToNumpy(array)
    data_vel = vtk_to_numpy(array) # [[0. 0. 0.]x 5]

    data_vel_u = data_vel[:, 0]
    data_vel_v = data_vel[:, 1]
    data_vel_w = data_vel[:, 2]
    
    x_data = x_data / case.X_scale
    y_data = y_data / case.Y_scale
    z_data = z_data / case.Z_scale
    ud = data_vel_u / case.U_scale
    vd = data_vel_v / case.U_scale
    wd = data_vel_w / case.U_scale
    
    
    xd = x_data.reshape(-1, 1)  # need to reshape to get 2D array # worden 5 losse lijsten
    yd = y_data.reshape(-1, 1)  # [[0.15],[0.07], etc.
    zd = z_data.reshape(-1, 1)
    ud = ud.reshape(-1, 1)  # need to reshape to get 2D array
    vd = vd.reshape(-1, 1)  # need to reshape to get 2D array
    wd = wd.reshape(-1, 1)

    return xd, yd, zd, ud, vd, wd


def dataReader2(case, x_data, y_data, z_data, random_flag=False):
    # !!specify pts location here:

    x_data = np.asarray(x_data)  # convert to numpy
    y_data = np.asarray(y_data)  # convert to numpy
    z_data = np.asarray(z_data)

    print('Loading',
          case.vel_file)  # mesh met X_scale = 2, velocity_sten_steady (is dit dan al de oplossing van 2D stenose?)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(case.vel_file)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()  # 21140
    print('n_points of the data file read:', n_points)

    if random_flag:
        N_sample = int(input("Please specify the amount of data points randomly sampled from the geometry"))
        sample_idx = n_points / N_sample
        print('Sampling', N_sample, 'points from a total of', n_points, 'points')
        x_vtk_mesh = np.zeros((n_points, 1))
        y_vtk_mesh = np.zeros((n_points, 1))
        z_vtk_mesh = np.zeros((n_points, 1))
        VTKpoints = vtk.vtkPoints()
        N_pts_data = 0
        for i in range(n_points):
            if i % sample_idx == 0:
                pt_iso = data_vtk.GetPoint(i)
                x_vtk_mesh[N_pts_data] = pt_iso[0]
                y_vtk_mesh[N_pts_data] = pt_iso[1]
                z_vtk_mesh[N_pts_data] = pt_iso[2]
                N_pts_data += 1

        print('n_points sampled', N_pts_data)
        x_data = np.zeros((N_pts_data, 1))
        y_data = np.zeros((N_pts_data, 1))
        z_data = np.zeros((N_pts_data, 1))

        x_data[:, 0] = x_vtk_mesh[0:N_pts_data, 0]
        y_data[:, 0] = y_vtk_mesh[0:N_pts_data, 0]
        z_data[:, 0] = z_vtk_mesh[0:N_pts_data, 0]

    VTKpoints = vtk.vtkPoints()
    for i in range(len(x_data)):
        VTKpoints.InsertPoint(i, x_data[i], y_data[i], z_data[i])  # anders dan voorheen; hier worden de arbitraire
        # gegenereerde datapunten ingevoerd, 'sensor'

    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)  # bouw een unstructured grid voor deze random input punten

    probe = vtk.vtkProbeFilter()  # sample data values at specified point locations --> sensor punten
    probe.SetInputData(point_data)
    probe.SetSourceData(data_vtk)
    probe.Update()
    array = probe.GetOutput().GetPointData().GetArray(case.fieldname)  # vtkFloatArray size tuple (5,3)
    # data_vel = vtkToNumpy(array)
    data_vel = vtk_to_numpy(array)  # [[0. 0. 0.]x 5]

    data_vel_u = data_vel[:, 0]  # * 12.
    data_vel_v = data_vel[:, 1]  # * 12.
    data_vel_w = data_vel[:, 2]  # * 12.

    x_data = x_data / case.X_scale
    y_data = y_data / case.Y_scale
    z_data = z_data / case.Z_scale
    ud = data_vel_u / case.U_scale
    vd = data_vel_v / case.U_scale
    wd = data_vel_w / case.U_scale

    xd = x_data.reshape(-1, 1)  # need to reshape to get 2D array # worden 5 losse lijsten
    yd = y_data.reshape(-1, 1)  # [[0.15],[0.07], etc.
    zd = z_data.reshape(-1, 1)
    ud = ud.reshape(-1, 1)  # need to reshape to get 2D array
    vd = vd.reshape(-1, 1)  # need to reshape to get 2D array
    wd = wd.reshape(-1, 1)

    return xd, yd, zd, ud, vd, wd

def planeReader(file, input_n, axis_idx, value, mesh=True):
    #classes van maken?
    print('Loading', file)  # mesh_file komt overeen met .vtu file die mesh moet geven
    if mesh:
        reader = vtk.vtkXMLUnstructuredGridReader() # Get the Reader's output
    elif input_n == 3:
        reader = vtk.vtkPolyDataReader()
    else:
        reader = vtk.vtkUnstructuredGridReader()  # for the boundary conditions

    reader.SetFileName(file) # Sets file name if exported
    reader.Update() # Fail-save mechanisme? Zeker weten dat de SetFileName werkt?
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints() # n_points 39238 van sten_mesh
    print('n_points of the mesh:', n_points)

    x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    z_vtk_mesh = np.zeros((n_points, 1))
    VTKpoints = vtk.vtkPoints() #Represent and manipulate 3D points
    for i in range(n_points): # voor alle punten (3D, i.e. [0,0.3,0] in de mesh, schrijf weg naar x array en y array. Kan hier ook z
        # aan toevoegen door pt_iso[2] weg te gaan schrijven
        # pt_iso = data_vtk.GetPoint(i)
        x, y, z = data_vtk.GetPoint(i)

        if axis_idx == 0:
            x = float("{0:.2f}".format(x))
            if x == value:
                # x_vtk_mesh[i] = x + 0.01 if x == 0 else x
                y_vtk_mesh[i] = y + 0.01 if y == 0 else y
                z_vtk_mesh[i] = z + 0.01 if z == 0 else z
                VTKpoints.InsertPoint(i, x, y, z) #voegt ook lijnsegmenten toe aan aanliggende punten
        elif axis_idx == 1:
            y = float("{0:.2f}".format(y))
            if y == value:
                x_vtk_mesh[i] = x + 0.01 if x == 0 else x
                # y_vtk_mesh[i] = y + 0.01 if y == 0 else y
                z_vtk_mesh[i] = z + 0.01 if z == 0 else z
                VTKpoints.InsertPoint(i, x, y, z) #voegt ook lijnsegmenten toe aan aanliggende punten
        elif axis_idx == 2:
            z = float("{0:.2f}".format(z))
            if z == value:
                x_vtk_mesh[i] = x + 0.01 if x == 0 else x
                y_vtk_mesh[i] = y + 0.01 if y == 0 else y
                # z_vtk_mesh[i] = z + 0.01 if z == 0 else z
                VTKpoints.InsertPoint(i, x, y, z) #voegt ook lijnsegmenten toe aan aanliggende punten

    if axis_idx == 0:
        y_vtk_mesh = y_vtk_mesh[y_vtk_mesh != 0]
        z_vtk_mesh = z_vtk_mesh[z_vtk_mesh != 0]
        x_vtk_mesh = np.zeros(len(z_vtk_mesh))
    elif axis_idx == 1:
        x_vtk_mesh = x_vtk_mesh[x_vtk_mesh != 0]
        z_vtk_mesh = z_vtk_mesh[z_vtk_mesh != 0]
        y_vtk_mesh = np.zeros(len(z_vtk_mesh))
    elif axis_idx == 2:
        x_vtk_mesh = x_vtk_mesh[x_vtk_mesh != 0]
        y_vtk_mesh = y_vtk_mesh[y_vtk_mesh != 0]
        z_vtk_mesh = np.zeros(len(x_vtk_mesh))

    point_data = vtk.vtkUnstructuredGrid() #dataset representeert arbitraire combinatie van mogelijke celtypes
    point_data.SetPoints(VTKpoints) # niet zeker, maar vermoed dat alle mesh_punten hier toegevoegd worden na de for loop hierboven

    x = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh[:]), 1)) # (n_points,1)
    y = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1)) # (n_points,1)
    z = np.reshape(z_vtk_mesh, (np.size(z_vtk_mesh[:]), 1))

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)

    return x, y, z

def planeReader2(file, input_n, x_plane, mesh=True):
    #classes van maken?
    print('Loading', file)  # mesh_file komt overeen met .vtu file die mesh moet geven
    if mesh:
        reader = vtk.vtkXMLUnstructuredGridReader() # Get the Reader's output
    elif input_n == 3:
        reader = vtk.vtkPolyDataReader()
    else:
        reader = vtk.vtkUnstructuredGridReader()  # for the boundary conditions

    reader.SetFileName(file) # Sets file name if exported
    reader.Update() # Fail-save mechanisme? Zeker weten dat de SetFileName werkt?
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints() # n_points 39238 van sten_mesh
    print('n_points of the mesh:', n_points)
    # x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    z_vtk_mesh = np.zeros((n_points, 1))
    VTKpoints = vtk.vtkPoints() #Represent and manipulate 3D points
    for i in range(n_points): # voor alle punten (3D, i.e. [0,0.3,0] in de mesh, schrijf weg naar x array en y array. Kan hier ook z
        # aan toevoegen door pt_iso[2] weg te gaan schrijven
        # pt_iso = data_vtk.GetPoint(i)
        x, y, z = data_vtk.GetPoint(i)
        z = float("{0:.2f}".format(z))
        if x == x_plane:
            z_vtk_mesh[i] = z + 0.01 if z == 0 else z
            y_vtk_mesh[i] = y + 0.01 if y == 0 else y
            # z_vtk_mesh[i] = z
            VTKpoints.InsertPoint(i, x, y, z) #voegt ook lijnsegmenten toe aan aanliggende punten

    z_vtk_mesh = z_vtk_mesh[z_vtk_mesh != 0]
    y_vtk_mesh = y_vtk_mesh[y_vtk_mesh != 0]
    x_vtk_mesh = np.zeros(len(z_vtk_mesh))

    point_data = vtk.vtkUnstructuredGrid() #dataset representeert arbitraire combinatie van mogelijke celtypes
    point_data.SetPoints(VTKpoints) # niet zeker, maar vermoed dat alle mesh_punten hier toegevoegd worden na de for loop hierboven

    x = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh[:]), 1)) # (n_points,1)
    y = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1)) # (n_points,1)
    z = np.reshape(z_vtk_mesh, (np.size(z_vtk_mesh[:]), 1))

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)

    return x, y, z
